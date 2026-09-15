#!/usr/bin/env python3
"""
Merge the outputs of transcribe_gemini.py (file endpoint, run WITH
--diarize --timestamps) and transcribe_gemini_live.py (live endpoint) into
one transcript that has both speaker labels AND proper utterance/language
segmentation -- since neither endpoint gives you both on its own.

How it works:
  - transcribe_gemini.py --diarize --timestamps gives you speaker turns,
    each with a start_seconds/end_seconds time range (derived from its
    word-level timestamps).
  - transcribe_gemini_live.py gives you cleanly segmented utterances, each
    with an approx_elapsed_seconds value (roughly when that utterance
    finished, based on how much audio had been streamed).
  - For each live utterance, this script finds the diarized speaker segment
    whose time range best contains (or is closest to) that utterance's
    timestamp, and labels the utterance with that speaker.

This is a best-effort time-based alignment, not an exact one -- the two
endpoints don't share a common clock, so matches near speaker-change
boundaries can occasionally be off by one segment. Spot-check the output,
especially around points where speakers change.

Usage:
    # 1. Run the file endpoint with diarization + timestamps:
    python transcribe_gemini.py audio.mp3 --diarize --timestamps

    # 2. Run the live endpoint for proper segmentation:
    python transcribe_gemini_live.py audio.mp3 --split-languages

    # 3. Merge them:
    python merge_transcripts.py audio.json audio.live.json -o audio.merged.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_speaker_timeline(diarized_data):
    """Extract (start, end, speaker, text) tuples from the diarized file-endpoint output."""
    timeline = []
    for seg in diarized_data.get("segments", []):
        start = seg.get("start_seconds")
        end = seg.get("end_seconds")
        if start is None or end is None:
            continue
        timeline.append({
            "start": start,
            "end": end,
            "speaker": seg.get("speaker"),
            "text": seg.get("text"),
        })
    timeline.sort(key=lambda s: s["start"])
    return timeline


def find_speaker_for_time(timeline, t):
    """Return the speaker label whose time range contains t, or the closest one if none does."""
    if not timeline:
        return None

    # Exact containment first.
    for seg in timeline:
        if seg["start"] <= t <= seg["end"]:
            return seg["speaker"]

    # Fall back to whichever segment's range is closest to t.
    def distance(seg):
        if t < seg["start"]:
            return seg["start"] - t
        return t - seg["end"]

    closest = min(timeline, key=distance)
    return closest["speaker"]


def main():
    parser = argparse.ArgumentParser(
        description="Merge diarized file-endpoint output with segmented live-endpoint output."
    )
    parser.add_argument("diarized_json", help="Output of transcribe_gemini.py (run with --diarize --timestamps)")
    parser.add_argument("live_json", help="Output of transcribe_gemini_live.py")
    parser.add_argument("-o", "--output", default=None,
                         help="Output path (default: <live_json_stem>.merged.json)")
    args = parser.parse_args()

    diarized_path = Path(args.diarized_json)
    live_path = Path(args.live_json)

    if not diarized_path.exists():
        print(f"Error: {diarized_path} not found", file=sys.stderr)
        sys.exit(1)
    if not live_path.exists():
        print(f"Error: {live_path} not found", file=sys.stderr)
        sys.exit(1)

    diarized_data = load_json(diarized_path)
    live_data = load_json(live_path)

    if not diarized_data.get("diarization_enabled"):
        print(
            f"Warning: {diarized_path} wasn't generated with --diarize -- "
            "there won't be any speaker labels to merge in.",
            file=sys.stderr,
        )

    timeline = build_speaker_timeline(diarized_data)
    if not timeline:
        print(
            f"Warning: no usable speaker time ranges found in {diarized_path}. "
            "Make sure it was generated with BOTH --diarize AND --timestamps -- "
            "word timestamps are what let us compute per-speaker time ranges.",
            file=sys.stderr,
        )

    merged_segments = []
    for utt in live_data.get("segments", []):
        t = utt.get("approx_elapsed_seconds")
        speaker = find_speaker_for_time(timeline, t) if t is not None else None
        merged_segments.append({
            "index": utt.get("index"),
            "speaker": speaker,
            "text": utt.get("text"),
            "detected_language": utt.get("detected_language"),
            "approx_elapsed_seconds": t,
        })

    result = {
        "file_name": live_data.get("file_name") or diarized_data.get("file_name"),
        "source_diarized_file": str(diarized_path),
        "source_live_file": str(live_path),
        "transcript": live_data.get("transcript"),
        "segments": merged_segments,
    }

    out_path = Path(args.output) if args.output else live_path.with_suffix("").with_suffix(".merged.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(merged_segments)} utterance(s) with speaker labels -> {out_path}")


if __name__ == "__main__":
    main()