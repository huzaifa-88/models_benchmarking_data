#!/usr/bin/env python3
"""
Chunk-level transcription with Gemini 3.5 Transcribe (file endpoint,
gemini-3.5-transcribe-preview on Vertex AI / Gemini Enterprise).

Approach being tested: instead of uploading the whole file in one call (which
returns one giant unsegmented transcript for multilingual audio -- see
transcribe_gemini.py's docstring for why), this script:

  1. Detects natural pauses in the audio using ffmpeg's silencedetect filter.
  2. Cuts the audio into chunks at those pauses, aiming for a target chunk
     length but never letting a chunk fall outside [--min-chunk-seconds,
     --max-chunk-seconds].
  3. If no pause is found within the max window, it forces a cut anyway (this
     risks clipping a word -- it's flagged in the output when it happens).
  4. Transcribes each chunk separately via the file endpoint.
  5. Merges the per-chunk results, in order, into one transcript with
     segments tagged by chunk index and offset into real start/end times.

What this fixes vs. a single upload: segmentation resolution becomes "one
segment per chunk" instead of "one segment for the whole file" -- so a
language switch that happens at a pause between chunks will land cleanly on
a chunk boundary.

What this does NOT fully fix: a language switch that happens *inside* a
single chunk (e.g. a code-switch mid-sentence with no pause) will still come
back as one blob containing both languages, same underlying problem at
smaller scale. Silence-based cutting reduces how often that happens; it
doesn't eliminate it.

Important caveat for --diarize: this endpoint's diarization has no memory
across separate API calls. Speaker labels (e.g. "spk_1") are assigned fresh
within each chunk, so "spk_1" in chunk 2 is NOT guaranteed to be the same
person as "spk_1" in chunk 1. This script does not attempt to reconcile
speaker identity across chunks -- treat per-chunk speaker labels as
chunk-local only, not global.

Setup:
    pip install google-genai
    # ffmpeg must be installed and on PATH
    # Optional, only for --split-languages:
    pip install langdetect

Usage:
    python transcribe_gemini_chunked.py audio.mp3
    python transcribe_gemini_chunked.py audio.mp3 --target-chunk-seconds 20 --max-chunk-seconds 30
    python transcribe_gemini_chunked.py audio.mp3 --diarize --split-languages
    python transcribe_gemini_chunked.py audio.mp3 --lang en-US,es-ES

Output:
    Writes "<basename>.chunked.json" containing the merged transcript, a
    global ordered list of segments (each tagged with chunk_index,
    start_seconds, end_seconds, speaker if diarized, detected_language if
    --split-languages was used), and a list of any forced (non-silence)
    cut points so you can see how often boundary-clipping risk occurred.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MIME_TYPES = {
    ".mp3": "audio/mp3",
    ".mpeg": "audio/mpeg",
    ".wav": "audio/wav",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/m4a",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
}

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\!\?\u3002\uff01\uff1f])\s+')
_SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")


def require_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(
                f"Error: {tool} not found on PATH. Install ffmpeg (which includes ffprobe), "
                "e.g. `brew install ffmpeg` or `apt install ffmpeg`.",
                file=sys.stderr,
            )
            sys.exit(1)


def get_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"Error: ffprobe failed on {path}:\n{result.stderr.decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    return float(result.stdout.decode().strip())


def detect_silences(path: Path, noise_db: float, min_duration: float):
    """Return a list of (silence_start, silence_end) tuples in seconds."""
    cmd = [
        "ffmpeg", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = result.stderr.decode(errors="replace")

    silences = []
    pending_start = None
    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            silences.append((pending_start, float(end_match.group(1))))
            pending_start = None
    return silences


def compute_chunk_boundaries(duration, silences, min_chunk, target_chunk, max_chunk):
    """
    Walk through the audio choosing cut points. Prefer cutting at the
    midpoint of a detected silence closest to the target chunk length; if no
    silence falls within [min_chunk, max_chunk] of the current position,
    force a cut at max_chunk (flagged as forced -- risk of clipping).
    """
    midpoints = sorted((s + e) / 2 for s, e in silences)

    boundaries = []
    forced_cuts = []
    current_start = 0.0

    while current_start < duration - 1e-6:
        remaining = duration - current_start
        if remaining <= max_chunk:
            boundaries.append((current_start, duration))
            break

        window_lo = current_start + min_chunk
        window_hi = current_start + max_chunk
        target = current_start + target_chunk

        candidates = [m for m in midpoints if window_lo <= m <= window_hi]
        if candidates:
            cut = min(candidates, key=lambda m: abs(m - target))
        else:
            cut = window_hi
            forced_cuts.append(round(cut, 2))

        boundaries.append((current_start, cut))
        current_start = cut

    return boundaries, forced_cuts


def extract_chunk_wav(src_path: Path, start: float, end: float, out_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-ss", str(start),
        "-to", str(end),
        "-ar", "44100",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-loglevel", "error",
        str(out_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error: ffmpeg failed extracting chunk [{start}, {end}]:\n"
              f"{result.stderr.decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)


def _offset_to_seconds(offset):
    if offset is None:
        return None
    if isinstance(offset, (int, float)):
        return float(offset)
    if isinstance(offset, str):
        try:
            return float(offset.rstrip("s"))
        except ValueError:
            return None
    seconds = getattr(offset, "seconds", None)
    nanos = getattr(offset, "nanos", None)
    if seconds is not None:
        return float(seconds) + (float(nanos) / 1e9 if nanos else 0.0)
    return None

def transcribe_chunk(client, types, chunk_bytes, mime_type, transcription_kwargs):
    print("TRANSCRIPTION KWARGS:")
    print(transcription_kwargs)
    response = client.models.generate_content(
        model="gemini-3.5-transcribe-preview",
        contents=[
            types.Part.from_bytes(
                data=chunk_bytes,
                mime_type=mime_type,
            )
        ],
        config=types.GenerateContentConfig(
            audio_transcription_config=types.AudioTranscriptionConfig(
                **transcription_kwargs,
            ),
        ),
    )
    # Gemini Transcribe returns transcription parts directly
    parts = getattr(response, "parts", None) or []

    chunk_segments = []
    chunk_text_pieces = []

    for part in parts:
        audio_tx = getattr(part, "audio_transcription", None)

        if not audio_tx:
            continue

        speaker = getattr(audio_tx, "speaker_label", None)
        segment_text = getattr(audio_tx, "text", None)

        if segment_text:
            chunk_text_pieces.append(segment_text)

        # Extract word-level timestamps directly from Gemini response
        print("========================================")
        print(f"Word-level timestamps for speaker {speaker}:{getattr(audio_tx, 'words', None)}")
        print("========================================")
        words = []

        for word_info in getattr(audio_tx, "words", None) or []:
            words.append({
                "word": getattr(word_info, "word", None),
                "start_offset": (
                    round(
                        _offset_to_seconds(getattr(word_info, "start_offset", None)),
                        3
                    )
                    if _offset_to_seconds(getattr(word_info, "start_offset", None)) is not None
                    else None
                ),
                "end_offset": (
                    round(
                        _offset_to_seconds(getattr(word_info, "end_offset", None)),
                        3
                    )
                    if _offset_to_seconds(getattr(word_info, "end_offset", None)) is not None
                    else None
                ),
            })

        if segment_text or words:
            chunk_segments.append({
                "speaker": speaker,
                "text": segment_text or "",
                "words": words,
            })

    return chunk_segments, "".join(chunk_text_pieces)


def tag_languages(text: str):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
    except ImportError:
        return None
    try:
        return detect(text)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Chunk-level transcription with Gemini 3.5 Transcribe")
    parser.add_argument("audio_path", help="Path to the source audio/video file")
    parser.add_argument("--credentials", default="google.json",
                         help="Path to GCP service account JSON key (default: google.json)")
    parser.add_argument("--project", default=None,
                         help="GCP project ID. If omitted, read from 'project_id' in the credentials JSON.")
    parser.add_argument("--location", default="global", help="Location (default: global)")
    parser.add_argument("--diarize", action="store_true",
                         help="Enable speaker diarization per chunk (see docstring: labels reset across chunks)")
    parser.add_argument("--timestamps", action="store_true", help="Enable word-level timestamps")
    parser.add_argument("--lang", default=None, help="Comma-separated BCP-47 language codes, e.g. en-US,es-ES")
    parser.add_argument("--vocab", default=None, help="Comma-separated custom vocabulary terms")
    parser.add_argument("--split-languages", action="store_true",
                         help="Tag each chunk with a local best-guess language via langdetect")
    parser.add_argument("--min-chunk-seconds", type=float, default=8.0,
                         help="Minimum chunk length before a silence is eligible as a cut point (default: 8.0)")
    parser.add_argument("--target-chunk-seconds", type=float, default=20.0,
                         help="Preferred chunk length -- cuts nearest this are chosen among eligible silences (default: 20.0)")
    parser.add_argument("--max-chunk-seconds", type=float, default=60.0,
                         help="Maximum chunk length -- forces a cut here if no silence was found first (default: 30.0)")
    parser.add_argument("--silence-threshold-db", type=float, default=-30.0,
                         help="Volume (dB) below which audio counts as silence (default: -30.0)")
    parser.add_argument("--silence-min-duration", type=float, default=0.4,
                         help="Minimum duration (seconds) of quiet to count as a pause (default: 0.4)")
    parser.add_argument("--output-dir", default=".", help="Directory to write the output JSON (default: current dir)")
    parser.add_argument("--keep-chunks", action="store_true",
                         help="Don't delete the temporary per-chunk audio files (useful for debugging)")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
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

    require_ffmpeg()
    client = genai.Client(enterprise=True, project=project, location=args.location)

    print(f"Probing {audio_path} ...")
    duration = get_duration_seconds(audio_path)
    print(f"Duration: {duration:.1f}s. Detecting pauses (silence >= {args.silence_min_duration}s, "
          f"below {args.silence_threshold_db}dB) ...")
    silences = detect_silences(audio_path, args.silence_threshold_db, args.silence_min_duration)
    print(f"Found {len(silences)} candidate pause(s).")

    boundaries, forced_cuts = compute_chunk_boundaries(
        duration, silences, args.min_chunk_seconds, args.target_chunk_seconds, args.max_chunk_seconds
    )
    print(f"Split into {len(boundaries)} chunk(s).")
    if forced_cuts:
        print(
            f"Warning: {len(forced_cuts)} chunk boundary(ies) had no pause available within the max "
            f"window and were force-cut at {forced_cuts} -- these are at higher risk of clipping a word. "
            "Consider raising --max-chunk-seconds or loosening --silence-threshold-db/--silence-min-duration.",
            file=sys.stderr,
        )

    transcription_kwargs = {}
    language_codes = []
    if args.diarize:
        transcription_kwargs["diarization"] = True
    if args.timestamps:
        transcription_kwargs["word_timestamp"] = True
    if args.lang:
        language_codes = [c.strip() for c in args.lang.split(",") if c.strip()]
        transcription_kwargs["language_codes"] = language_codes
    if args.vocab:
        transcription_kwargs["custom_vocabulary"] = [v.strip() for v in args.vocab.split(",") if v.strip()]

    all_segments = []
    full_text_pieces = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for i, (start, end) in enumerate(boundaries):
            chunk_path = tmp_dir_path / f"chunk_{i:03d}.wav"
            print(f"[{i + 1}/{len(boundaries)}] Extracting [{start:.1f}s - {end:.1f}s] ...")
            extract_chunk_wav(audio_path, start, end, chunk_path)

            if args.keep_chunks:
                keep_dir = Path(args.output_dir) / f"{audio_path.stem}_chunks"
                keep_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(chunk_path, keep_dir / chunk_path.name)

            with open(chunk_path, "rb") as f:
                chunk_bytes = f.read()

            print(f"[{i + 1}/{len(boundaries)}] Transcribing ...")
            chunk_segments, chunk_text = transcribe_chunk(
                client, types, chunk_bytes, "audio/wav", transcription_kwargs
            )

            for seg in chunk_segments:
                seg["chunk_index"] = i
                seg["chunk_start_seconds"] = round(start, 2)
                seg["chunk_end_seconds"] = round(end, 2)
                if args.split_languages:
                    seg["detected_language"] = tag_languages(seg["text"])
                all_segments.append(seg)

            if chunk_text:
                full_text_pieces.append(chunk_text)
                print(f"[{i + 1}/{len(boundaries)}] -> {chunk_text[:80]!r}")

    full_transcript = " ".join(full_text_pieces)

    all_words = []
    for segment in all_segments:
        for word in segment.get("words", []):
            all_words.append({
                "word": word.get("word"),
                "start_seconds": (round(segment["chunk_start_seconds"] + word["start_offset"], 3) if word.get("start_offset") is not None else None),
                "end_seconds": (round(segment["chunk_start_seconds"] + word["end_offset"], 3) if word.get("end_offset") is not None else None),
                "speaker": segment.get("speaker"),
                "chunk_index": segment.get("chunk_index"),
            })

    result = {
        "file_name": audio_path.name,
        "model": "gemini-3.5-transcribe-preview",
        "approach": "chunked-file-endpoint",
        "diarization_enabled": bool(args.diarize),
        "diarization_note": (
            "Speaker labels are assigned independently within each chunk and are NOT "
            "consistent across chunks -- treat 'speaker' as chunk-local only."
            if args.diarize else None
        ),
        "timestamps_enabled": bool(args.timestamps),
        "language_codes": language_codes,
        "chunking": {
            "min_chunk_seconds": args.min_chunk_seconds,
            "target_chunk_seconds": args.target_chunk_seconds,
            "max_chunk_seconds": args.max_chunk_seconds,
            "silence_threshold_db": args.silence_threshold_db,
            "silence_min_duration": args.silence_min_duration,
            "num_chunks": len(boundaries),
            "forced_cut_points": forced_cuts,
        },
        "transcript": full_transcript,
        "segments": all_segments,
        "word_timestamps": all_words,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audio_path.stem}.chunked.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print( 
        f"\nDone. {len(all_segments)} segment(s), " 
        f"{len(all_words)} word(s) across {len(boundaries)} chunk(s) " 
        f"saved to {out_path}" 
    )
    if forced_cuts:
        print(f"({len(forced_cuts)} chunk boundary(ies) were force-cut without a pause -- see 'chunking.forced_cut_points')")


if __name__ == "__main__":
    main()
