# #!/usr/bin/env python3
# """
# Transcribe audio using Gemini 3.5 Transcribe on the Gemini Enterprise Agent
# Platform (Vertex AI), authenticated with a GCP service account key (e.g.
# google.json).

# Note: Vertex AI / Enterprise uses a different API shape than the public
# Gemini Developer API docs. It uses client.models.generate_content() with
# types.AudioTranscriptionConfig, and the model id has a "-preview" suffix
# (gemini-3.5-transcribe-preview), not the Interactions API.

# Setup:
#     pip install google-genai

#     Set the AUDIO_PATH variable below to point at your audio file. Make sure
#     google.json (your GCP service account key) is in the same folder as this
#     script -- the project ID is read automatically from its 'project_id'
#     field. Then just run:

#         python transcribe_gemini.py

#     # With speaker diarization + word timestamps:
#     python transcribe_gemini.py --diarize --timestamps

#     # Hint the spoken language(s):
#     python transcribe_gemini.py --lang en-US

#     # Bias recognition toward specific terms:
#     python transcribe_gemini.py --vocab "Kubernetes,BigQuery"

# Output:
#     Writes "<audio_basename>.json" (e.g. sample.mp3 -> sample.json) next to
#     the script (or into --output-dir if given), containing the transcript
#     and, if requested, word-level annotations / speaker labels.

# Limits (per Google's docs for this endpoint):
#     - Audio up to 15 minutes per request.
#     - File processing is capped at 15 minutes when diarization or word
#       timestamps are enabled.
#     - Speaker attribution for 3+ speakers is experimental.
# """

# import argparse
# import json
# import os
# import sys
# from pathlib import Path

# # ---------------------------------------------------------------------------
# # Set the path to your source audio file here.
# # ---------------------------------------------------------------------------
# AUDIO_PATH = "/Users/dev/Media/AndreaPhitzer-[en-US].mp4"

# MIME_TYPES = {
#     ".mp3": "audio/mp3",
#     ".mpeg": "audio/mpeg",
#     ".wav": "audio/wav",
#     ".aiff": "audio/aiff",
#     ".aac": "audio/aac",
#     ".ogg": "audio/ogg",
#     ".flac": "audio/flac",
#     ".m4a": "audio/m4a",
#     ".opus": "audio/opus",
#     ".webm": "audio/webm",
# }


# def main():
#     parser = argparse.ArgumentParser(description="Transcribe audio with Gemini 3.5 Transcribe")
#     parser.add_argument(
#         "audio_path",
#         nargs="?",
#         default=AUDIO_PATH,
#         help="Path to the source audio file to transcribe (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--credentials",
#         default="google.json",
#         help="Path to GCP service account JSON key (default: google.json)",
#     )
#     parser.add_argument("--project", default=None,
#                          help="GCP project ID. If omitted, it's read from the "
#                               "'project_id' field inside the credentials JSON.")
#     parser.add_argument("--location", default="global",
#                          help="Location (default: global -- gemini-3.5-transcribe-preview "
#                               "is only served from 'global', not regional locations)")
#     parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
#     parser.add_argument("--timestamps", action="store_true", help="Enable word-level timestamps")
#     parser.add_argument("--lang", default=None, help="Comma-separated BCP-47 language codes, e.g. en-US or en-US,es-ES")
#     parser.add_argument("--vocab", default=None, help="Comma-separated custom vocabulary terms")
#     parser.add_argument("--output-dir", default=".", help="Directory to write the output JSON (default: current dir)")
#     args = parser.parse_args()

#     audio_path = Path(args.audio_path)
#     if not audio_path.exists():
#         print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
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
#         print(
#             f"Error: could not find 'project_id' in {creds_path}, and no "
#             "--project was given. Pass --project YOUR_PROJECT_ID explicitly.",
#             file=sys.stderr,
#         )
#         sys.exit(1)

#     # Point the Google auth libraries at the service account key file.
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.resolve())

#     try:
#         from google import genai
#         from google.genai import types
#     except ImportError:
#         print("Error: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
#         sys.exit(1)

#     client = genai.Client(enterprise=True, project=project, location=args.location)

#     mime_type = MIME_TYPES.get(audio_path.suffix.lower(), "audio/mp3")

#     print(f"Reading {audio_path} ...")
#     with open(audio_path, "rb") as f:
#         audio_bytes = f.read()

#     transcription_kwargs = {}
#     if args.diarize:
#         transcription_kwargs["diarization"] = True
#     if args.timestamps:
#         transcription_kwargs["word_timestamp"] = True
#     if args.lang:
#         transcription_kwargs["language_codes"] = [c.strip() for c in args.lang.split(",") if c.strip()]
#     if args.vocab:
#         transcription_kwargs["custom_vocabulary"] = [v.strip() for v in args.vocab.split(",") if v.strip()]

#     print("Requesting transcription from gemini-3.5-transcribe-preview ...")
#     response = client.models.generate_content(
#         model="gemini-3.5-transcribe-preview",
#         contents=[
#             types.Part.from_bytes(
#                 data=audio_bytes,
#                 mime_type=mime_type,
#             ),
#         ],
#         config=types.GenerateContentConfig(
#             audio_transcription_config=types.AudioTranscriptionConfig(**transcription_kwargs),
#         ),
#     )

#     print("Transcription response received. Processing ...")
#     print(f"Response: {response}")

#     candidates = getattr(response, "candidates", None) or []
#     content = getattr(candidates[0], "content", None) if candidates else None
#     parts = getattr(content, "parts", None) or []
#     print("===============================================")
#     print("Transcription parts:", parts)
#     print("===============================================")

#     segments = []
#     words = []
#     full_text_chunks = []

#     for part in parts:
#         text = getattr(part, "text", None)
#         audio_tx = getattr(part, "audio_transcription", None)
#         speaker = getattr(audio_tx, "speaker_label", None) if audio_tx else None

#         segment_text = text or (getattr(audio_tx, "text", None) if audio_tx else None)
#         if segment_text:
#             full_text_chunks.append(segment_text)
#             segments.append({"speaker": speaker, "text": segment_text})

#         if audio_tx:
#             for w in getattr(audio_tx, "words", None) or []:
#                 words.append(
#                     {
#                         "word": getattr(w, "word", None),
#                         "start_offset": getattr(w, "start_offset", None),
#                         "end_offset": getattr(w, "end_offset", None),
#                     }
#                 )

#     result = {
#         "file_name": audio_path.name,
#         "model": "gemini-3.5-transcribe-preview",
#         "diarization_enabled": bool(args.diarize),
#         "timestamps_enabled": bool(args.timestamps),
#         "language_codes": transcription_kwargs.get("language_codes", []),
#         "transcript": "".join(full_text_chunks),
#     }
#     if args.diarize:
#         result["segments"] = segments
#     if words:
#         result["words"] = words

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     out_path = output_dir / f"{audio_path.stem}.json"

#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)

#     print(f"Done. Transcript saved to {out_path}")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Transcribe audio using Gemini 3.5 Transcribe on the Gemini Enterprise Agent
Platform (Vertex AI), authenticated with a GCP service account key (e.g.
google.json).

Note: Vertex AI / Enterprise uses a different API shape than the public
Gemini Developer API docs. It uses client.models.generate_content() with
types.AudioTranscriptionConfig, and the model id has a "-preview" suffix
(gemini-3.5-transcribe-preview), not the Interactions API.

--- Why multilingual audio was showing up as ONE segment ---
Per Google's own docs for this model, "Utterance-level Timestamps" are
NOT supported on the synchronous file-processing endpoint
(gemini-3.5-transcribe-preview) -- only on the Live/streaming endpoint
(gemini-3.5-transcribe-live-preview). Language auto-detection (85+ locales,
including mid-utterance code-switching) still runs under the hood on the
file endpoint, but the API gives you no seams to cut the output into
per-language chunks -- it just returns one continuous transcript.

On top of that, the original script had a real bug: it only attached the
`segments` list to the output JSON when --diarize was passed, even though
it built that list unconditionally. So even if the API *did* return more
than one part, it was silently dropped unless --diarize was set.

This version fixes that, and adds an optional local (non-API) heuristic
--split-languages flag that segments the transcript by sentence and tags
each with a best-guess language, as a practical workaround. For *real*
model-driven utterance/language boundaries, you need the Live/streaming
endpoint (gemini-3.5-transcribe-live-preview), which is a different,
async, chunked-audio API shape -- not covered by this script.

Setup:
    pip install google-genai
    # Optional, only needed for --split-languages:
    pip install langdetect

    Set the AUDIO_PATH variable below to point at your audio file. Make sure
    google.json (your GCP service account key) is in the same folder as this
    script -- the project ID is read automatically from its 'project_id'
    field. Then just run:

        python transcribe_gemini.py

    # With speaker diarization + word timestamps:
    python transcribe_gemini.py --diarize --timestamps

    # Hint the spoken language(s):
    python transcribe_gemini.py --lang en-US

    # Bias recognition toward specific terms:
    python transcribe_gemini.py --vocab "Kubernetes,BigQuery"

    # Multilingual audio: local sentence-level language tagging workaround
    python transcribe_gemini.py --lang en-US,es-ES --split-languages

Output:
    Writes "<audio_basename>.json" (e.g. sample.mp3 -> sample.json) next to
    the script (or into --output-dir if given), containing the transcript
    and, if requested, word-level annotations / speaker labels / detected
    per-sentence languages.

Limits (per Google's docs for this endpoint):
    - Audio up to 15 minutes per request.
    - File processing is capped at 15 minutes when diarization or word
      timestamps are enabled.
    - Speaker attribution for 3+ speakers is experimental.
    - Utterance-level segmentation is NOT available on this (file) endpoint,
      only on the live/streaming endpoint.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Set the path to your source audio file here.
# ---------------------------------------------------------------------------
AUDIO_PATH = "/Users/dev/Downloads/Test_Clip.mp4"

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

# Rough sentence splitter that copes with common Latin + CJK terminators.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\!\?\u3002\uff01\uff1f])\s+')


def split_into_sentences(text: str):
    """Best-effort sentence splitter for local language tagging."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def tag_languages(text: str):
    """
    Local, offline, best-effort language tagging per sentence. This is a
    heuristic workaround -- NOT the same as model-driven utterance/language
    boundaries, which this API endpoint does not expose. Requires
    `pip install langdetect`.
    """
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic results
    except ImportError:
        print(
            "Warning: --split-languages requires the 'langdetect' package. "
            "Install it with: pip install langdetect",
            file=sys.stderr,
        )
        return None

    tagged = []
    for sentence in split_into_sentences(text):
        try:
            lang = detect(sentence)
        except Exception:
            lang = None
        tagged.append({"text": sentence, "detected_language": lang})
    return tagged


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Gemini 3.5 Transcribe")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=AUDIO_PATH,
        help="Path to the source audio file to transcribe (default: %(default)s)",
    )
    parser.add_argument(
        "--credentials",
        default="google.json",
        help="Path to GCP service account JSON key (default: google.json)",
    )
    parser.add_argument("--project", default=None,
                         help="GCP project ID. If omitted, it's read from the "
                              "'project_id' field inside the credentials JSON.")
    parser.add_argument("--location", default="global",
                         help="Location (default: global -- gemini-3.5-transcribe-preview "
                              "is only served from 'global', not regional locations)")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--timestamps", action="store_true", help="Enable word-level timestamps")
    parser.add_argument("--lang", default=None, help="Comma-separated BCP-47 language codes, e.g. en-US or en-US,es-ES")
    parser.add_argument("--vocab", default=None, help="Comma-separated custom vocabulary terms")
    parser.add_argument(
        "--split-languages",
        action="store_true",
        help="Locally tag each sentence of the transcript with a best-guess "
             "language (via langdetect). This is a workaround, not true "
             "model-driven utterance segmentation -- see script docstring.",
    )
    parser.add_argument("--output-dir", default=".", help="Directory to write the output JSON (default: current dir)")
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
        print(
            f"Error: could not find 'project_id' in {creds_path}, and no "
            "--project was given. Pass --project YOUR_PROJECT_ID explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Point the Google auth libraries at the service account key file.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.resolve())

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(enterprise=True, project=project, location=args.location)

    mime_type = MIME_TYPES.get(audio_path.suffix.lower(), "audio/mp3")

    print(f"Reading {audio_path} ...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    transcription_kwargs = {}
    language_codes = ["es-US"]
    if args.diarize:
        transcription_kwargs["diarization"] = True
    if args.timestamps:
        transcription_kwargs["word_timestamp"] = True
    if args.lang:
        language_codes = [c.strip() for c in args.lang.split(",") if c.strip()]
        transcription_kwargs["language_codes"] = language_codes
    if args.vocab:
        transcription_kwargs["custom_vocabulary"] = [v.strip() for v in args.vocab.split(",") if v.strip()]

    print("Requesting transcription from gemini-3.5-transcribe-preview ...")
    response = client.models.generate_content(
        model="gemini-3.5-transcribe-preview",
        contents=[
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type,
            ),
        ],
        config=types.GenerateContentConfig(
            audio_transcription_config=types.AudioTranscriptionConfig(**transcription_kwargs),
        ),
    )

    print("Transcription response received. Processing ...")
    print(f"Response: {response}")

    candidates = getattr(response, "candidates", None) or []
    content = getattr(candidates[0], "content", None) if candidates else None
    parts = getattr(content, "parts", None) or []
    print("===============================================")
    print("Transcription parts:", parts)
    print("===============================================")

    segments = []
    words = []
    full_text_chunks = []

    for part in parts:
        text = getattr(part, "text", None)
        audio_tx = getattr(part, "audio_transcription", None)
        speaker = getattr(audio_tx, "speaker_label", None) if audio_tx else None

        segment_text = text or (getattr(audio_tx, "text", None) if audio_tx else None)
        if segment_text:
            full_text_chunks.append(segment_text)
            segments.append({"speaker": speaker, "text": segment_text})

        if audio_tx:
            for w in getattr(audio_tx, "words", None) or []:
                words.append(
                    {
                        "word": getattr(w, "word", None),
                        "start_offset": getattr(w, "start_offset", None),
                        "end_offset": getattr(w, "end_offset", None),
                    }
                )

    full_transcript = "".join(full_text_chunks)

    # --- Diagnostic: explain the "one giant segment" symptom when it happens ---
    if len(parts) <= 1 and (args.diarize or len(language_codes) > 1):
        print(
            "\nNote: the API returned a single part covering the whole clip.\n"
            "This is expected on the file-processing endpoint "
            "(gemini-3.5-transcribe-preview): it does not support "
            "utterance-level segmentation, only the Live/streaming endpoint "
            "(gemini-3.5-transcribe-live-preview) does. Language "
            "auto-detection still runs internally, it's just not exposed as "
            "separate chunks here. Use --split-languages for a local "
            "heuristic workaround, or switch to the live/streaming model for "
            "real per-utterance language segmentation.\n"
        )

    result = {
        "file_name": audio_path.name,
        "model": "gemini-3.5-transcribe-preview",
        "diarization_enabled": bool(args.diarize),
        "timestamps_enabled": bool(args.timestamps),
        "language_codes": language_codes,
        "transcript": full_transcript,
        # Always include segments now -- previously this was dropped
        # silently unless --diarize was passed, even though it was always
        # being built.
        "segments": segments,
    }
    if words:
        result["words"] = words

    if args.split_languages:
        tagged = tag_languages(full_transcript)
        if tagged is not None:
            result["language_segments"] = tagged

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audio_path.stem}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Transcript saved to {out_path}")


if __name__ == "__main__":
    main()