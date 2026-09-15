import os
import time
import shutil
import logging
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from mutagen import File as MutagenFile
from deepgram import DeepgramClient


# =====================================================
# CONFIG
# =====================================================

MODEL = "nova-3"

# MEDIA_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/test_video/vocals.wav"
# MEDIA_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/Video__On_Demand___68207434c013a___Chinese__Cantonese__Traditional____source/vocals.wav"
# MEDIA_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/BAL_2023_Ep07__2830__source___1/vocals.wav"
MEDIA_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/FBSNBKFCKNUC00OTA901__source___source/vocals.wav"

CHUNK_THRESHOLD_SECONDS = 120
CHUNK_DURATION_SECONDS = 120


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =====================================================
# HELPERS
# =====================================================


def get_duration(path):

    media = MutagenFile(path)

    if media is None:
        raise Exception(
            "Unable to read media duration"
        )

    return float(media.info.length)



def split_media(path):

    chunk_dir = Path("chunks")

    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    chunk_dir.mkdir()


    output = str(
        chunk_dir / "chunk_%03d.mp4"
    )


    command = [
        "ffmpeg",
        "-i",
        path,

        "-c",
        "copy",

        "-f",
        "segment",

        "-segment_time",
        str(CHUNK_DURATION_SECONDS),

        "-reset_timestamps",
        "1",

        output,

        "-y"
    ]


    logger.info(
        "Creating chunks..."
    )


    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


    chunks = sorted(
        str(x)
        for x in chunk_dir.glob(
            "chunk_*.mp4"
        )
    )


    logger.info(
        f"Created {len(chunks)} chunks"
    )


    return chunks



def transcribe_chunk(
    client,
    chunk
):

    retry = 3


    for attempt in range(retry):

        try:

            logger.info(
                f"Processing {chunk}"
            )


            start = time.perf_counter()


            with open(chunk,"rb") as f:
                data = f.read()



            response = (
                client
                .listen
                .v1
                .media
                .transcribe_file(

                    request=data,

                    model=MODEL,

                    smart_format=True,

                    punctuate=True,

                    diarize=True,

                    detect_language=True,
                )
            )


            latency = (
                time.perf_counter()
                -
                start
            )


            transcript = (
                response
                .results
                .channels[0]
                .alternatives[0]
                .transcript
            )


            confidence = (
                response
                .results
                .channels[0]
                .alternatives[0]
                .confidence
            )


            logger.info(
                f"{Path(chunk).name} "
                f"done "
                f"{latency:.2f}s"
            )


            return {
                "chunk":chunk,
                "latency":latency,
                "confidence":confidence,
                "text":transcript
            }



        except Exception as e:


            logger.warning(
                f"Attempt {attempt+1} failed "
                f"{chunk}: {e}"
            )


            time.sleep(5)



    raise Exception(
        f"Failed processing {chunk}"
    )




# =====================================================
# MAIN
# =====================================================


def main():


    load_dotenv()

    api_key = os.getenv("DEEPGRAM_API_KEY")

    if not api_key:
        raise Exception(
            "Missing DEEPGRAM_API_KEY"
        )


    client = DeepgramClient(api_key=api_key)



    duration = get_duration(
        MEDIA_FILE
    )


    logger.info(
        f"Duration: {duration:.2f}s"
    )



    start_total = time.perf_counter()



    final_results = []



    if duration <= CHUNK_THRESHOLD_SECONDS:


        logger.info(
            "Single file mode"
        )


        final_results.append(
            transcribe_chunk(
                client,
                MEDIA_FILE
            )
        )



    else:


        logger.info(
            "Chunk mode enabled"
        )


        chunks = split_media(
            MEDIA_FILE
        )


        # IMPORTANT
        # sequential processing
        # avoids upload timeout

        for index, chunk in enumerate(chunks):


            logger.info(
                f"Chunk {index+1}/{len(chunks)}"
            )


            result = transcribe_chunk(
                client,
                chunk
            )


            final_results.append(
                result
            )



    total_time = (
        time.perf_counter()
        -
        start_total
    )



    final_results.sort(
        key=lambda x:x["chunk"]
    )



    transcript = "\n".join(
        x["text"]
        for x in final_results
    )



    confidence = (
        sum(
            x["confidence"]
            for x in final_results
        )
        /
        len(final_results)
    )



    rtf = (
        total_time
        /
        duration
    )



    logger.info(
        "="*70
    )

    logger.info(
        f"Total Time: {total_time:.2f}s"
    )

    logger.info(
        f"RTF: {rtf:.3f}"
    )

    logger.info(
        f"Confidence: {confidence:.4f}"
    )



    media_name = Path(MEDIA_FILE).stem
    counter = 1
    output_file = Path(f"{media_name}{counter}.txt")
    while output_file.exists():
        counter += 1
        output_file = Path(f"{media_name}{counter}.txt")

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(transcript)



    logger.info(
        f"Saved {output_file}"
    )



if __name__ == "__main__":
    main()