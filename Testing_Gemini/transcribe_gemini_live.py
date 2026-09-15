#!/usr/bin/env python3
"""
Transcribe an EXISTING audio/video file with real utterance-level segments by
streaming it through Gemini 3.5 Transcribe's Live endpoint
(gemini-3.5-transcribe-live-preview on Vertex/Enterprise), rather than the
one-shot file endpoint (gemini-3.5-transcribe-preview).

Why: per Google's docs, the one-shot file endpoint does NOT support
utterance-level timestamps/segmentation -- it just returns one continuous
blob of text, even for multilingual audio. The Live/streaming endpoint DOES
support utterance-level segmentation, it just expects audio to arrive as a
stream of small chunks rather than one upload. This script simulates that by
reading your existing file, converting it to the required raw PCM format,
and feeding it into the Live session in small chunks -- it's not "live"
audio, just the same wire format the Live API expects. You get back a
properly segmented transcript, one entry per utterance, as the server
actually finalizes them.

Trade-offs vs. the one-shot file endpoint:
  - No speaker diarization on this endpoint (only on the file endpoint).
  - Word-level timestamps are not available here (only utterance-level).
  - Session cap is 10 minutes of streamed audio (vs 15 min on the file
    endpoint).
  - The API does not label *which* language each utterance is in -- it just
    transcribes correctly across languages. If you want a language label per
    utterance, use --split-languages (local langdetect heuristic, same
    caveat as before: it's a heuristic, not something the model reports).

Setup:
    pip install google-genai
    # ffmpeg must be installed and on PATH (used to convert your file to
    # 16-bit PCM / 16kHz / mono, which is what the Live API requires)
    # Optional, only for --split-languages:
    pip install langdetect

Usage:
    python transcribe_gemini_live.py path/to/video.mp4
    python transcribe_gemini_live.py path/to/audio.mp3 --lang en-US,es-ES --split-languages
    python transcribe_gemini_live.py path/to/audio.wav --smart --fast

Output:
    Writes "<basename>.live.json" next to the script (or into --output-dir),
    containing an ordered list of finalized utterance segments plus the full
    concatenated transcript.
"""

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import os
import time
from pathlib import Path

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM
CHANNELS = 1

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\!\?\u3002\uff01\uff1f])\s+')


def split_into_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def detect_language(text: str):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
    except ImportError:
        return None
    try:
        return detect(text)
    except Exception:
        return None


def convert_to_pcm(src_path: Path) -> bytes:
    """Use ffmpeg to convert any audio/video file to raw 16-bit PCM, 16kHz, mono."""
    if not shutil.which("ffmpeg"):
        print(
            "Error: ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg` "
            "or `apt install ffmpeg`) -- it's needed to convert your file to the "
            "raw PCM format the Live API requires.",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        "ffmpeg",
        "-i", str(src_path),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-loglevel", "error",
        "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error: ffmpeg failed to convert {src_path}:\n{result.stderr.decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def chunk_pcm(pcm_bytes: bytes, chunk_ms: int):
    """Yield (chunk_bytes, chunk_duration_seconds) tuples."""
    bytes_per_ms = (SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS) / 1000.0
    chunk_size = int(bytes_per_ms * chunk_ms)
    chunk_size -= chunk_size % BYTES_PER_SAMPLE  # keep sample-aligned
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i + chunk_size]
        if chunk:
            yield chunk, len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS)


async def run_live_transcription(
    client,
    types,
    model: str,
    pcm_bytes: bytes,
    chunk_ms: int,
    pace_realtime: bool,
    language_codes,
    custom_vocabulary,
    smart_mode: bool,
    idle_timeout: float,
    show_interim: bool,
):
    config_kwargs = {"language_codes": language_codes or []}
    if custom_vocabulary:
        config_kwargs["custom_vocabulary"] = custom_vocabulary
    if smart_mode:
        config_kwargs["mode"] = "SMART"

    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(**config_kwargs),
    )

    segments = []
    start_time = time.monotonic()
    send_done = asyncio.Event()

    async def sender(session):
        for chunk, duration in chunk_pcm(pcm_bytes, chunk_ms):
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
            )
            if pace_realtime:
                await asyncio.sleep(duration)
        await session.send_realtime_input(audio_stream_end=True)
        send_done.set()

    async def receiver(session):
        # BUG FIX: the original code called session.receive() fresh on every
        # loop iteration, which creates a brand-new async generator each
        # time instead of continuing to consume the same message stream --
        # that's why messages could be missed or the loop could behave
        # unpredictably. Create the generator ONCE and keep calling
        # __anext__() on that same object.
        message_stream = session.receive()
        while True:
            try:
                # Stop waiting for more messages once sending is done and
                # nothing new has arrived for `idle_timeout` seconds.
                timeout = idle_timeout if send_done.is_set() else None
                if timeout is not None:
                    message = await asyncio.wait_for(message_stream.__anext__(), timeout=timeout)
                else:
                    message = await message_stream.__anext__()
            except (asyncio.TimeoutError, StopAsyncIteration):
                break

            server_content = getattr(message, "server_content", None)
            if not server_content:
                continue

            interim = getattr(server_content, "interim_input_transcription", None)
            if interim and getattr(interim, "text", None) and show_interim:
                print(f"\r[interim] {interim.text[:100]}", end="", flush=True)

            final = getattr(server_content, "input_transcription", None)
            if final and getattr(final, "text", None):
                if show_interim:
                    print()  # newline after interim line
                elapsed = time.monotonic() - start_time
                segments.append({
                    "index": len(segments),
                    "text": final.text,
                    "approx_elapsed_seconds": round(elapsed, 2),
                })
                print(f"[final segment {len(segments)}] {final.text}")

    async with client.aio.live.connect(model=model, config=config) as session:
        try:
            await asyncio.gather(sender(session), receiver(session))
        except Exception as e:
            # Don't lose whatever segments were already collected if the
            # connection drops or errors out partway through -- surface the
            # error but still return what we have.
            print(f"\nWarning: live session ended early ({e!r}). "
                  f"Returning {len(segments)} segment(s) collected so far.", file=sys.stderr)

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an existing audio/video file with real utterance-level "
                    "segments via the Gemini 3.5 Transcribe Live endpoint."
    )
    parser.add_argument("audio_path", help="Path to the source audio/video file")
    parser.add_argument("--credentials", default="google.json",
                         help="Path to GCP service account JSON key (default: google.json)")
    parser.add_argument("--project", default=None,
                         help="GCP project ID. If omitted, read from 'project_id' in the credentials JSON.")
    parser.add_argument("--location", default="global",
                         help="Location (default: global)")
    parser.add_argument("--lang", default=None,
                         help="Comma-separated BCP-47 language codes to hint, e.g. en-US,es-ES. "
                              "Omit for automatic language detection (recommended for multilingual audio).")
    parser.add_argument("--vocab", default=None, help="Comma-separated custom vocabulary terms")
    parser.add_argument("--smart", action="store_true",
                         help="Use Smart transcription mode (removes filler words, formats output)")
    parser.add_argument("--chunk-ms", type=int, default=100,
                         help="Audio chunk size in milliseconds to stream at a time (default: 100)")
    parser.add_argument("--fast", action="store_true",
                         help="Send chunks as fast as possible instead of pacing to real-time playback speed. "
                              "Real-time pacing (the default) is safer for the server's turn-detection logic.")
    parser.add_argument("--idle-timeout", type=float, default=8.0,
                         help="Seconds to wait for trailing messages after all audio has been sent (default: 8.0)")
    parser.add_argument("--show-interim", action="store_true",
                         help="Print live partial (interim) transcriptions to the console as they arrive")
    parser.add_argument("--split-languages", action="store_true",
                         help="Tag each finalized utterance with a local best-guess language via langdetect")
    parser.add_argument("--output-dir", default=".", help="Directory to write the output JSON (default: current dir)")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    creds_path = Path(args.credentials)
    if not creds_path.exists():
        print(f"Error: credentials file not found: {creds_path}", file=sys.stderr)
        sys.exit(1)

    project = args.project
    if not project:
        try:
            with open(creds_path, "r", encoding="utf-8") as f:
                sa_key = json.load(f)
            project = sa_key.get("project_id")
        except (json.JSONDecodeError, OSError):
            project = None
    if not project:
        print(f"Error: could not find 'project_id' in {creds_path}, and no --project given.", file=sys.stderr)
        sys.exit(1)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.resolve())

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(enterprise=True, project=project, location=args.location)

    language_codes = [c.strip() for c in args.lang.split(",") if c.strip()] if args.lang else []
    custom_vocabulary = [v.strip() for v in args.vocab.split(",") if v.strip()] if args.vocab else None

    print(f"Converting {audio_path} to 16kHz mono PCM with ffmpeg ...")
    pcm_bytes = convert_to_pcm(audio_path)
    duration_s = len(pcm_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS)
    print(f"Audio duration: ~{duration_s:.1f}s. Streaming to gemini-3.5-transcribe-live-preview ...")
    if duration_s > 600:
        print(
            "Warning: audio is longer than the 10-minute Live session cap. "
            "The session may be cut off before the end of the file.",
            file=sys.stderr,
        )

    segments = asyncio.run(run_live_transcription(
        client=client,
        types=types,
        model="gemini-3.5-transcribe-live-preview",
        pcm_bytes=pcm_bytes,
        chunk_ms=args.chunk_ms,
        pace_realtime=not args.fast,
        language_codes=language_codes,
        custom_vocabulary=custom_vocabulary,
        smart_mode=args.smart,
        idle_timeout=args.idle_timeout,
        show_interim=args.show_interim,
    ))

    if args.split_languages:
        for seg in segments:
            seg["detected_language"] = detect_language(seg["text"])

    full_transcript = " ".join(s["text"] for s in segments)

    result = {
        "file_name": audio_path.name,
        "model": "gemini-3.5-transcribe-live-preview",
        "language_codes": language_codes,
        "smart_mode": args.smart,
        "transcript": full_transcript,
        "segments": segments,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audio_path.stem}.live.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(segments)} utterance segment(s) saved to {out_path}")


if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# """
# Transcribe an EXISTING audio/video file with real utterance-level segments by
# streaming it through Gemini 3.5 Transcribe's Live endpoint
# (gemini-3.5-transcribe-live-preview on Vertex/Enterprise), rather than the
# one-shot file endpoint (gemini-3.5-transcribe-preview).

# Why: per Google's docs, the one-shot file endpoint does NOT support
# utterance-level timestamps/segmentation -- it just returns one continuous
# blob of text, even for multilingual audio. The Live/streaming endpoint DOES
# support utterance-level segmentation, it just expects audio to arrive as a
# stream of small chunks rather than one upload. This script simulates that by
# reading your existing file, converting it to the required raw PCM format,
# and feeding it into the Live session in small chunks -- it's not "live"
# audio, just the same wire format the Live API expects. You get back a
# properly segmented transcript, one entry per utterance, as the server
# actually finalizes them.

# Trade-offs vs. the one-shot file endpoint:
#   - No speaker diarization on this endpoint (only on the file endpoint).
#   - Word-level timestamps are not available here (only utterance-level).
#   - Session cap is 10 minutes of streamed audio (vs 15 min on the file
#     endpoint).
#   - The API does not label *which* language each utterance is in -- it just
#     transcribes correctly across languages. If you want a language label per
#     utterance, use --split-languages (local langdetect heuristic, same
#     caveat as before: it's a heuristic, not something the model reports).

# Setup:
#     pip install google-genai
#     # ffmpeg must be installed and on PATH (used to convert your file to
#     # 16-bit PCM / 16kHz / mono, which is what the Live API requires)
#     # Optional, only for --split-languages:
#     pip install langdetect

# Usage:
#     python transcribe_gemini_live.py path/to/video.mp4
#     python transcribe_gemini_live.py path/to/audio.mp3 --lang en-US,es-ES --split-languages
#     python transcribe_gemini_live.py path/to/audio.wav --smart --fast

# Output:
#     Writes "<basename>.live.json" next to the script (or into --output-dir),
#     containing an ordered list of finalized utterance segments plus the full
#     concatenated transcript.
# """

# import argparse
# import asyncio
# import json
# import re
# import shutil
# import subprocess
# import sys
# import os
# import time
# from pathlib import Path

# SAMPLE_RATE = 16000
# BYTES_PER_SAMPLE = 2  # 16-bit PCM
# CHANNELS = 1

# _SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\!\?\u3002\uff01\uff1f])\s+')


# def split_into_sentences(text: str):
#     text = text.strip()
#     if not text:
#         return []
#     parts = _SENTENCE_SPLIT_RE.split(text)
#     return [p.strip() for p in parts if p.strip()]


# def detect_language(text: str):
#     try:
#         from langdetect import detect, DetectorFactory
#         DetectorFactory.seed = 0
#     except ImportError:
#         return None
#     try:
#         return detect(text)
#     except Exception:
#         return None


# def convert_to_pcm(src_path: Path) -> bytes:
#     """Use ffmpeg to convert any audio/video file to raw 16-bit PCM, 16kHz, mono."""
#     if not shutil.which("ffmpeg"):
#         print(
#             "Error: ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg` "
#             "or `apt install ffmpeg`) -- it's needed to convert your file to the "
#             "raw PCM format the Live API requires.",
#             file=sys.stderr,
#         )
#         sys.exit(1)

#     cmd = [
#         "ffmpeg",
#         "-i", str(src_path),
#         "-f", "s16le",
#         "-acodec", "pcm_s16le",
#         "-ar", str(SAMPLE_RATE),
#         "-ac", str(CHANNELS),
#         "-loglevel", "error",
#         "-",
#     ]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.returncode != 0:
#         print(f"Error: ffmpeg failed to convert {src_path}:\n{result.stderr.decode(errors='replace')}", file=sys.stderr)
#         sys.exit(1)
#     return result.stdout


# def chunk_pcm(pcm_bytes: bytes, chunk_ms: int):
#     """Yield (chunk_bytes, chunk_duration_seconds) tuples."""
#     bytes_per_ms = (SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS) / 1000.0
#     chunk_size = int(bytes_per_ms * chunk_ms)
#     chunk_size -= chunk_size % BYTES_PER_SAMPLE  # keep sample-aligned
#     for i in range(0, len(pcm_bytes), chunk_size):
#         chunk = pcm_bytes[i:i + chunk_size]
#         if chunk:
#             yield chunk, len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS)


# async def run_live_transcription(
#     client,
#     types,
#     model: str,
#     pcm_bytes: bytes,
#     chunk_ms: int,
#     pace_realtime: bool,
#     language_codes,
#     custom_vocabulary,
#     smart_mode: bool,
#     idle_timeout: float,
#     show_interim: bool,
# ):
#     config_kwargs = {"language_codes": language_codes or []}
#     if custom_vocabulary:
#         config_kwargs["custom_vocabulary"] = custom_vocabulary
#     if smart_mode:
#         config_kwargs["mode"] = "SMART"

#     config = types.LiveConnectConfig(
#         response_modalities=["TEXT"],
#         input_audio_transcription=types.AudioTranscriptionConfig(**config_kwargs),
#     )

#     segments = []
#     start_time = time.monotonic()
#     send_done = asyncio.Event()

#     async def sender(session):
#         for chunk, duration in chunk_pcm(pcm_bytes, chunk_ms):
#             await session.send_realtime_input(
#                 audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
#             )
#             if pace_realtime:
#                 await asyncio.sleep(duration)
#         await session.send_realtime_input(audio_stream_end=True)
#         send_done.set()

#     async def receiver(session):
#         # BUG FIX: the original code called session.receive() fresh on every
#         # loop iteration, which creates a brand-new async generator each
#         # time instead of continuing to consume the same message stream --
#         # that's why messages could be missed or the loop could behave
#         # unpredictably. Create the generator ONCE and keep calling
#         # __anext__() on that same object.
#         message_stream = session.receive()
#         while True:
#             try:
#                 # Stop waiting for more messages once sending is done and
#                 # nothing new has arrived for `idle_timeout` seconds.
#                 timeout = idle_timeout if send_done.is_set() else None
#                 if timeout is not None:
#                     message = await asyncio.wait_for(message_stream.__anext__(), timeout=timeout)
#                 else:
#                     message = await message_stream.__anext__()
#             except (asyncio.TimeoutError, StopAsyncIteration):
#                 break

#             server_content = getattr(message, "server_content", None)
#             if not server_content:
#                 continue

#             interim = getattr(server_content, "interim_input_transcription", None)
#             if interim and getattr(interim, "text", None) and show_interim:
#                 print(f"\r[interim] {interim.text[:100]}", end="", flush=True)

#             final = getattr(server_content, "input_transcription", None)
#             if final and getattr(final, "text", None):
#                 if show_interim:
#                     print()  # newline after interim line
#                 elapsed = time.monotonic() - start_time
#                 segments.append({
#                     "index": len(segments),
#                     "text": final.text,
#                     "approx_elapsed_seconds": round(elapsed, 2),
#                 })
#                 print(f"[final segment {len(segments)}] {final.text}")

#     async with client.aio.live.connect(model=model, config=config) as session:
#         try:
#             await asyncio.gather(sender(session), receiver(session))
#         except Exception as e:
#             # Don't lose whatever segments were already collected if the
#             # connection drops or errors out partway through -- surface the
#             # error but still return what we have.
#             print(f"\nWarning: live session ended early ({e!r}). "
#                   f"Returning {len(segments)} segment(s) collected so far.", file=sys.stderr)

#     return segments


# def main():
#     parser = argparse.ArgumentParser(
#         description="Transcribe an existing audio/video file with real utterance-level "
#                     "segments via the Gemini 3.5 Transcribe Live endpoint."
#     )
#     parser.add_argument("audio_path", help="Path to the source audio/video file")
#     parser.add_argument("--credentials", default="google.json",
#                          help="Path to GCP service account JSON key (default: google.json)")
#     parser.add_argument("--project", default=None,
#                          help="GCP project ID. If omitted, read from 'project_id' in the credentials JSON.")
#     parser.add_argument("--location", default="global",
#                          help="Location (default: global)")
#     parser.add_argument("--lang", default=None,
#                          help="Comma-separated BCP-47 language codes to hint, e.g. en-US,es-ES. "
#                               "Omit for automatic language detection (recommended for multilingual audio).")
#     parser.add_argument("--vocab", default=None, help="Comma-separated custom vocabulary terms")
#     parser.add_argument("--smart", action="store_true",
#                          help="Use Smart transcription mode (removes filler words, formats output)")
#     parser.add_argument("--chunk-ms", type=int, default=100,
#                          help="Audio chunk size in milliseconds to stream at a time (default: 100)")
#     parser.add_argument("--fast", action="store_true",
#                          help="Send chunks as fast as possible instead of pacing to real-time playback speed. "
#                               "Real-time pacing (the default) is safer for the server's turn-detection logic.")
#     parser.add_argument("--idle-timeout", type=float, default=8.0,
#                          help="Seconds to wait for trailing messages after all audio has been sent (default: 8.0)")
#     parser.add_argument("--show-interim", action="store_true",
#                          help="Print live partial (interim) transcriptions to the console as they arrive")
#     parser.add_argument("--split-languages", action="store_true",
#                          help="Tag each finalized utterance with a local best-guess language via langdetect")
#     parser.add_argument("--output-dir", default=".", help="Directory to write the output JSON (default: current dir)")
#     args = parser.parse_args()

#     audio_path = Path(args.audio_path)
#     if not audio_path.exists():
#         print(f"Error: file not found: {audio_path}", file=sys.stderr)
#         sys.exit(1)

#     creds_path = Path(args.credentials)
#     if not creds_path.exists():
#         print(f"Error: credentials file not found: {creds_path}", file=sys.stderr)
#         sys.exit(1)

#     project = args.project
#     if not project:
#         try:
#             with open(creds_path, "r", encoding="utf-8") as f:
#                 sa_key = json.load(f)
#             project = sa_key.get("project_id")
#         except (json.JSONDecodeError, OSError):
#             project = None
#     if not project:
#         print(f"Error: could not find 'project_id' in {creds_path}, and no --project given.", file=sys.stderr)
#         sys.exit(1)

#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.resolve())

#     try:
#         from google import genai
#         from google.genai import types
#     except ImportError:
#         print("Error: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
#         sys.exit(1)

#     client = genai.Client(enterprise=True, project=project, location=args.location)

#     language_codes = [c.strip() for c in args.lang.split(",") if c.strip()] if args.lang else []
#     custom_vocabulary = [v.strip() for v in args.vocab.split(",") if v.strip()] if args.vocab else None

#     print(f"Converting {audio_path} to 16kHz mono PCM with ffmpeg ...")
#     pcm_bytes = convert_to_pcm(audio_path)
#     duration_s = len(pcm_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS)
#     print(f"Audio duration: ~{duration_s:.1f}s. Streaming to gemini-3.5-transcribe-live-preview ...")
#     if duration_s > 600:
#         print(
#             "Warning: audio is longer than the 10-minute Live session cap. "
#             "The session may be cut off before the end of the file.",
#             file=sys.stderr,
#         )

#     segments = asyncio.run(run_live_transcription(
#         client=client,
#         types=types,
#         model="gemini-3.5-transcribe-live-preview",
#         pcm_bytes=pcm_bytes,
#         chunk_ms=args.chunk_ms,
#         pace_realtime=not args.fast,
#         language_codes=language_codes,
#         custom_vocabulary=custom_vocabulary,
#         smart_mode=args.smart,
#         idle_timeout=args.idle_timeout,
#         show_interim=args.show_interim,
#     ))

#     if args.split_languages:
#         for seg in segments:
#             seg["detected_language"] = detect_language(seg["text"])

#     full_transcript = " ".join(s["text"] for s in segments)

#     result = {
#         "file_name": audio_path.name,
#         "model": "gemini-3.5-transcribe-live-preview",
#         "language_codes": language_codes,
#         "smart_mode": args.smart,
#         "transcript": full_transcript,
#         "segments": segments,
#     }

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     out_path = output_dir / f"{audio_path.stem}.live.json"
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)

#     print(f"\nDone. {len(segments)} utterance segment(s) saved to {out_path}")


# if __name__ == "__main__":
#     main()