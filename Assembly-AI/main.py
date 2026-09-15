import os
import time
import math
import logging
import tempfile
from pathlib import Path

import assemblyai as aai
from dotenv import load_dotenv
from mutagen import File as MutagenFile
from pydub import AudioSegment

# =====================================================
# CONFIG
# =====================================================

# MEDIA_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/test_video/vocals.wav"
# MEDIA_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/Video__On_Demand___68207434c013a___Chinese__Cantonese__Traditional____source/vocals.wav"
# MEDIA_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/BAL_2023_Ep07__2830__source___1/vocals.wav"
MEDIA_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/FBSNBKFCKNUC00OTA901__source___source/vocals.wav"

# chunk if audio > this duration
CHUNK_THRESHOLD_SEC = 1800  # 30 min

# chunk size
CHUNK_SIZE_SEC = 900  # 15 min

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY not found")

aai.settings.api_key = API_KEY

# =====================================================
# HELPERS
# =====================================================

def get_media_duration(path: str) -> float:
    """
    Duration in seconds
    """
    media = MutagenFile(path)

    if media is None or not hasattr(media, "info"):
        raise RuntimeError(f"Cannot determine duration for {path}")

    return media.info.length


def split_audio(path: str):
    """
    Split large audio into chunks
    """
    logger.info("Loading media for chunking...")

    audio = AudioSegment.from_file(path)

    total_ms = len(audio)

    chunk_ms = CHUNK_SIZE_SEC * 1000

    temp_dir = tempfile.mkdtemp(prefix="assembly_chunks_")

    chunk_files = []

    for idx, start in enumerate(range(0, total_ms, chunk_ms)):
        end = min(start + chunk_ms, total_ms)

        chunk = audio[start:end]

        chunk_path = os.path.join(
            temp_dir,
            f"chunk_{idx}.wav"
        )

        chunk.export(chunk_path, format="wav")

        chunk_files.append(chunk_path)

    logger.info(
        "Created %s chunks",
        len(chunk_files)
    )

    return chunk_files


# def transcribe_file(file_path: str):
#     """
#     Transcribe a single file
#     """

#     transcriber = aai.Transcriber()

#     start = time.time()

#     transcript = transcriber.transcribe(file_path)

#     latency = time.time() - start

#     if transcript.status == aai.TranscriptStatus.error:
#         raise RuntimeError(transcript.error)

#     return transcript.text, latency
def get_average_confidence(transcript):
    if not transcript.words:
        return None

    confidences = [
        word.confidence
        for word in transcript.words
        if word.confidence is not None
    ]

    if not confidences:
        return None

    return sum(confidences) / len(confidences)


def transcribe_file(file_path: str):

    transcriber = aai.Transcriber()

    start = time.time()

    transcript = transcriber.transcribe(file_path)

    latency = time.time() - start

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(transcript.error)

    confidence = get_average_confidence(transcript)

    return (
        transcript.text,
        latency,
        confidence,
    )


# =====================================================
# MAIN
# =====================================================

# def main():

    duration = get_media_duration(MEDIA_FILE)

    logger.info(
        "Media Duration: %.2f sec (%.2f min)",
        duration,
        duration / 60
    )

    total_start = time.time()

    full_text = []

    total_api_latency = 0

    if duration > CHUNK_THRESHOLD_SEC:

        logger.info("Chunk mode enabled")

        chunk_files = split_audio(MEDIA_FILE)

        for idx, chunk in enumerate(chunk_files):

            logger.info(
                "Processing chunk %s/%s",
                idx + 1,
                len(chunk_files)
            )

            text, latency = transcribe_file(chunk)

            total_api_latency += latency

            full_text.append(text)

    else:

        logger.info("Single file mode")

        text, latency = transcribe_file(MEDIA_FILE)

        total_api_latency += latency

        full_text.append(text)

    transcript_text = "\n".join(full_text)

    total_runtime = time.time() - total_start

    words = len(transcript_text.split())

    speed_ratio = duration / total_api_latency

    media_name = Path(MEDIA_FILE).stem

    output_file = OUTPUT_DIR / f"{media_name}.txt"

    output_file.write_text(
        transcript_text,
        encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info("ASSEMBLY AI RESULTS")
    logger.info("=" * 60)

    logger.info("Duration: %.2f sec", duration)
    logger.info("Word Count: %s", words)

    logger.info(
        "API Latency: %.2f sec",
        total_api_latency
    )

    logger.info(
        "Total Runtime: %.2f sec",
        total_runtime
    )

    logger.info(
        "Realtime Factor: %.2fx",
        speed_ratio
    )

    logger.info(
        "Transcript saved: %s",
        output_file
    )
def main():

    duration = get_media_duration(MEDIA_FILE)

    logger.info(
        "Media Duration: %.2f sec (%.2f min)",
        duration,
        duration / 60
    )

    total_start = time.time()

    full_text = []

    total_api_latency = 0

    confidences = []

    if duration > CHUNK_THRESHOLD_SEC:

        logger.info("Chunk mode enabled")

        chunk_files = split_audio(MEDIA_FILE)

        for idx, chunk in enumerate(chunk_files):

            logger.info(
                "Processing chunk %s/%s",
                idx + 1,
                len(chunk_files)
            )

            text, latency, confidence = transcribe_file(
                chunk
            )

            total_api_latency += latency

            full_text.append(text)

            if confidence is not None:
                confidences.append(confidence)

    else:

        logger.info("Single file mode")

        text, latency, confidence = transcribe_file(
            MEDIA_FILE
        )

        total_api_latency += latency

        full_text.append(text)

        if confidence is not None:
            confidences.append(confidence)

    transcript_text = "\n".join(full_text)

    total_runtime = time.time() - total_start

    words = len(transcript_text.split())

    avg_confidence = None

    if confidences:
        avg_confidence = (
            sum(confidences)
            / len(confidences)
        )

    rtf = total_api_latency / duration

    speed_multiplier = (
        duration / total_api_latency
    )

    media_name = Path(MEDIA_FILE).stem

    counter = 1
    output_file = OUTPUT_DIR / f"{media_name}{counter}.txt"
    
    while output_file.exists():
        counter += 1
        output_file = OUTPUT_DIR / f"{media_name}{counter}.txt"

    output_file.write_text(
        transcript_text,
        encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info("ASSEMBLYAI RESULTS")
    logger.info("=" * 60)

    logger.info(
        "Duration: %.2f sec",
        duration
    )

    logger.info(
        "Word Count: %s",
        words
    )

    if avg_confidence is not None:
        logger.info(
            "Average Confidence: %.4f",
            avg_confidence
        )

    logger.info(
        "API Latency: %.2f sec",
        total_api_latency
    )

    logger.info(
        "Total Runtime: %.2f sec",
        total_runtime
    )

    logger.info(
        "RTF: %.4f",
        rtf
    )

    logger.info(
        "Speed Multiplier: %.2fx",
        speed_multiplier
    )

    logger.info(
        "Transcript saved: %s",
        output_file
    )

if __name__ == "__main__":
    main()