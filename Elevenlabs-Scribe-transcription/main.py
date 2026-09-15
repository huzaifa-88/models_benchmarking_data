
# ==================== Audio Transcripton ======================

import os
import time
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from mutagen import File as MutagenFile
from pydub import AudioSegment


# ==========================================
# CONFIG
# ==========================================

MODEL_ID = "scribe_v2"

# AUDIO_FILE = (
#     "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/BAL_2023_Ep07__2830__source___1/vocals.wav"
# )
# AUDIO_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/test_video/vocals.wav"
# AUDIO_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/Video__On_Demand___68207434c013a___Chinese__Cantonese__Traditional____source/vocals.wav"
# AUDIO_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/BAL_2023_Ep07__2830__source___1/vocals.wav"
AUDIO_FILE = "/Users/dev/Projects/Test/tmp/deepfilternet_benchmark/FBSNBKFCKNUC00OTA901__source___source/vocals.wav"

# chunk size
CHUNK_MINUTES = 2

CHUNK_SIZE_MS = CHUNK_MINUTES * 60 * 1000


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================
# ENV
# ==========================================

load_dotenv()

API_KEY = os.getenv("ELEVEN_API_KEY")

if not API_KEY:
    raise ValueError(
        "ELEVEN_API_KEY missing"
    )


# ==========================================
# HELPERS
# ==========================================

def get_audio_duration(file_path):

    audio = MutagenFile(file_path)

    return float(
        audio.info.length
    )


def get_file_size_mb(file_path):

    return (
        os.path.getsize(file_path)
        /
        (1024 * 1024)
    )


# ==========================================
# SPLIT AUDIO
# ==========================================

def split_audio(
    audio_path
):

    logger.info(
        "Loading audio..."
    )

    audio = AudioSegment.from_file(
        audio_path
    )


    duration = len(audio)


    chunks = []

    index = 1


    for start in range(
        0,
        duration,
        CHUNK_SIZE_MS
    ):

        end = min(
            start + CHUNK_SIZE_MS,
            duration
        )


        chunk = audio[
            start:end
        ]


        chunk_file = (
            f"chunk_{index}.wav"
        )


        chunk.export(
            chunk_file,
            format="wav"
        )


        chunks.append(
            {
                "index": index,
                "path": chunk_file,
                "start": start / 1000,
                "end": end / 1000,
                "duration": (
                    end-start
                ) / 1000
            }
        )


        index += 1


    return chunks



# ==========================================
# TRANSCRIBE CHUNK
# ==========================================

def transcribe_chunk(
    chunk
):

    path = chunk["path"]


    logger.info("")
    logger.info(
        "=" * 70
    )

    logger.info(
        "Processing Chunk %s",
        chunk["index"]
    )


    logger.info(
        "Start Time : %.2f sec",
        chunk["start"]
    )

    logger.info(
        "End Time   : %.2f sec",
        chunk["end"]
    )

    logger.info(
        "Duration   : %.2f sec",
        chunk["duration"]
    )


    size = get_file_size_mb(
        path
    )

    logger.info(
        "Size       : %.2f MB",
        size
    )


    url = (
        "https://api.elevenlabs.io/"
        "v1/speech-to-text"
    )


    headers = {

        "xi-api-key": API_KEY

    }


    start = time.perf_counter()


    with open(
        path,
        "rb"
    ) as f:


        files = {

            "file": (
                Path(path).name,
                f,
                "audio/wav"
            )

        }


        data = {

            "model_id":
                MODEL_ID

        }


        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=600
        )


    end = time.perf_counter()


    latency = (
        end - start
    )


    if response.status_code != 200:

        logger.error(
            response.text
        )

        response.raise_for_status()


    result = response.json()


    text = result.get(
        "text",
        ""
    )

    confidence = result.get(
        "confidence"
    )


    words = len(
        text.split()
    )

    chars = len(
        text
    )


    rtf = (
        latency
        /
        chunk["duration"]
    )


    logger.info(
        "Latency    : %.3f sec",
        latency
    )


    logger.info(
        "RTF        : %.4f",
        rtf
    )


    if confidence is not None:
        logger.info(
            "Confidence : %.2f",
            confidence
        )
    else:
        logger.info(
            "Confidence : N/A"
        )


    logger.info(
        "Words      : %s",
        words
    )


    logger.info(
        "Characters : %s",
        chars
    )


    logger.info(
        "Transcript:"
    )

    logger.info(
        text
    )


    return {

        "text": text,

        "confidence": confidence,

        "duration":
            chunk["duration"],

        "latency":
            latency,

        "words":
            words,

        "chars":
            chars

    }



# ==========================================
# MAIN BENCHMARK
# ==========================================


def benchmark(
    audio_file
):


    total_start = (
        time.perf_counter()
    )


    duration = get_audio_duration(
        audio_file
    )


    logger.info(
        "Starting ElevenLabs Scribe V2 Benchmark"
    )

    logger.info(
        "Model: %s",
        MODEL_ID
    )

    logger.info(
        "Audio Duration: %.2f sec",
        duration
    )


    chunks = split_audio(
        audio_file
    )


    logger.info(
        "Total Chunks: %s",
        len(chunks)
    )


    transcripts = []


    total_latency = 0

    total_words = 0

    total_chars = 0

    confidences = []



    for chunk in chunks:


        result = transcribe_chunk(
            chunk
        )


        transcripts.append(
            result["text"]
        )

        if result.get("confidence") is not None:
            confidences.append(result.get("confidence"))


        total_latency += (
            result["latency"]
        )


        total_words += (
            result["words"]
        )


        total_chars += (
            result["chars"]
        )



    total_time = (
        time.perf_counter()
        -
        total_start
    )


    logger.info(
        "#" * 70
    )

    # compute average confidence if available
    avg_confidence = None
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)

    if avg_confidence is not None:
        logger.info(
            "Average Confidence: %.4f",
            avg_confidence
        )

    # save transcript to a numbered file to avoid overwrites
    media_name = Path(audio_file).stem
    counter = 1
    output_file = Path(f"{media_name}{counter}.txt")
    while output_file.exists():
        counter += 1
        output_file = Path(f"{media_name}{counter}.txt")

    transcript_text = "\n".join(transcripts)
    output_file.write_text(transcript_text, encoding="utf-8")

    logger.info(
        "Saved transcript: %s",
        output_file
    )


    logger.info(
        "Total Duration : %.2f sec",
        duration
    )


    logger.info(
        "Total Latency  : %.3f sec",
        total_latency
    )


    logger.info(
        "Wall Time      : %.3f sec",
        total_time
    )


    logger.info(
        "RTF            : %.4f",
        total_latency / duration
    )


    logger.info(
        "Words          : %s",
        total_words
    )


    logger.info(
        "Characters     : %s",
        total_chars
    )


    logger.info(
        "#" * 70
    )


    # logger.info(
    #     "FINAL TRANSCRIPT"
    # )

    # logger.info(
    #     "\n".join(transcripts)
    # )



if __name__ == "__main__":

    benchmark(
        AUDIO_FILE
    )



# ==================== Streaming Transcripton ======================


# import os
# import json
# import time
# import base64
# import queue
# import logging
# import threading

# import sounddevice as sd
# from websocket import WebSocketApp
# from dotenv import load_dotenv
# import threading

# shutdown_event = threading.Event()

# # =====================================================
# # CONFIG
# # =====================================================

# MODEL_ID = "scribe_v2_realtime"
# SAMPLE_RATE = 16000
# CHANNELS = 1
# CHUNK_MS = 50

# WS_URL = (
#     "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
#     f"?model_id={MODEL_ID}"
#     "&include_timestamps=true"
# )

# # =====================================================
# # ENV
# # =====================================================

# load_dotenv()

# API_KEY = os.getenv("ELEVEN_API_KEY")

# if not API_KEY:
#     raise RuntimeError("ELEVEN_API_KEY missing")

# # =====================================================
# # LOGGING
# # =====================================================

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
#     datefmt="%H:%M:%S",
# )

# log = logging.getLogger("scribe")

# # =====================================================
# # METRICS
# # =====================================================

# # audio_seconds_sent = 0.0

# # session_start = None
# # first_partial_time = None
# # first_commit_time = None

# # partial_count = 0
# # commit_count = 0
# audio_seconds_sent = 0.0

# session_connected_at = None
# speech_started_at = None

# first_partial_time = None
# first_commit_time = None

# partial_count = 0
# commit_count = 0

# final_transcript = []
# # =====================================================
# # AUDIO QUEUE
# # =====================================================

# audio_queue = queue.Queue()

# # =====================================================
# # AUDIO CALLBACK
# # =====================================================

# def audio_callback(indata, frames, time_info, status):

#     if status:
#         log.warning(status)

#     audio_queue.put(indata.copy())


# # =====================================================
# # WEBSOCKET EVENTS
# # =====================================================

# def on_open(ws):

#     global session_connected_at

#     session_connected_at = time.perf_counter()

#     log.info("Connected to ElevenLabs Realtime")

#     threading.Thread(
#         target=audio_sender,
#         args=(ws,),
#         daemon=True,
#     ).start()


# def on_message(ws, message):

#     global partial_count
#     global commit_count

#     global first_partial_time
#     global first_commit_time

#     data = json.loads(message)

#     event = data.get("message_type")

#     now = time.perf_counter()

#     if event == "partial_transcript":

#         partial_count += 1

#         text = data.get("text", "")

#         if first_partial_time is None:

#             first_partial_time = now

#             latency = (
#                 first_partial_time
#                 - speech_started_at
#             )

#             log.info(
#                 "[FIRST_PARTIAL] %.3f sec",
#                 latency,
#             )

#         log.info("[PARTIAL] %s", text)

#     elif event == "committed_transcript":

#         commit_count += 1

#         text = data.get("text", "")

#         final_transcript.append(text)

#         if first_commit_time is None:

#             first_commit_time = now

#             latency = (
#                 first_commit_time
#                 - speech_started_at
#             )

#             log.info(
#                 "[FIRST_COMMIT] %.3f sec",
#                 latency,
#             )

#         log.info("[COMMIT] %s", text)


# def on_error(ws, error):
#     log.error(error)


# def on_close(ws, status_code, msg):

#     log.info(
#         "Closed status=%s msg=%s",
#         status_code,
#         msg,
#     )

#     shutdown_event.set()

#     print_benchmark()


# # =====================================================
# # AUDIO STREAMER
# # =====================================================

# def audio_sender(ws):

#     global audio_seconds_sent
#     global speech_started_at

#     chunk_size = int(
#         SAMPLE_RATE *
#         CHUNK_MS /
#         1000
#     )

#     with sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         channels=CHANNELS,
#         dtype="int16",
#         blocksize=chunk_size,
#         callback=audio_callback,
#     ):

#         log.info("=" * 60)
#         log.info("Microphone Started")
#         log.info("Speak now...")
#         log.info("=" * 60)

#         # while True:
#         while not shutdown_event.is_set():

#             chunk = audio_queue.get()

#             if speech_started_at is None:
#                 speech_started_at = time.perf_counter()

#             audio_seconds_sent += (
#                 len(chunk) / SAMPLE_RATE
#             )

#             payload = {
#                 "message_type": "input_audio_chunk",
#                 "audio_base_64": base64.b64encode(
#                     chunk.tobytes()
#                 ).decode(),
#             }

#             # ws.send(json.dumps(payload))
#             try:
#                 ws.send(json.dumps(payload))
#             except Exception as e:
#                 log.error(
#                     "Failed to send audio: %s",
#                     e
#                 )

#                 shutdown_event.set()
#                 break
# # =====================================================
# # METRICS
# # =====================================================

# def print_benchmark():

#     transcript = " ".join(final_transcript)

#     words = len(transcript.split())
#     chars = len(transcript)

#     runtime = (
#         time.perf_counter()
#         - session_connected_at
#     )

#     rtf = (
#         runtime / audio_seconds_sent
#         if audio_seconds_sent
#         else 0
#     )

#     throughput = (
#         audio_seconds_sent / runtime
#         if runtime
#         else 0
#     )

#     log.info("")
#     log.info("=" * 70)
#     log.info("BENCHMARK SUMMARY")
#     log.info("=" * 70)

#     log.info(
#         "Audio Duration       : %.2f sec",
#         audio_seconds_sent
#     )

#     log.info(
#         "Runtime              : %.2f sec",
#         runtime
#     )

#     if first_partial_time:
#         log.info(
#             "First Partial Latency: %.3f sec",
#             first_partial_time
#             - speech_started_at
#         )

#     if first_commit_time:
#         log.info(
#             "First Commit Latency : %.3f sec",
#             first_commit_time
#             - speech_started_at
#         )

#     log.info(
#         "RTF                  : %.4f",
#         rtf
#     )

#     log.info(
#         "Throughput           : %.4f",
#         throughput
#     )

#     log.info(
#         "Partial Events       : %d",
#         partial_count
#     )

#     log.info(
#         "Commit Events        : %d",
#         commit_count
#     )

#     log.info(
#         "Words                : %d",
#         words
#     )

#     log.info(
#         "Characters           : %d",
#         chars
#     )

#     log.info("=" * 70)

# def main():
#     print("MAIN CALLED")
#     ws = WebSocketApp(
#         WS_URL,
#         header=[
#             f"xi-api-key: {API_KEY}"
#         ],
#         on_open=on_open,
#         on_message=on_message,
#         on_error=on_error,
#         on_close=on_close,
#     )
#     print("RUNNING WEBSOCKET")

#     try:
#         ws.run_forever()

#     except KeyboardInterrupt:

#         print_benchmark()

#         log.info("Stopped")

# if __name__ == "__main__":
#     main()
