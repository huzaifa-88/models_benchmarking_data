# ================== This script is running on GPU by CoreML ==================
#!/usr/bin/env python3
"""Benchmark DPDFNet speech enhancement on local media files.

Enhanced speech is written to vocals.wav; accompaniment is derived as mix - vocals
(same approach as ClearVoice / DeepFilterNet / ZipEnhancer benchmarks).

Usage:
    ./scripts/run_benchmark_dpdfnet.sh path/to/media.mp4

    # Or manually from the isolated env:
    cd benchmark/dpdfnet && uv sync
    uv run python ../../scripts/benchmark_dpdfnet.py path/to/audio.wav \\
        --model dpdfnet8_48khz_hr

Requires:
    benchmark/dpdfnet venv (dpdfnet; separate from main project)
    ffmpeg on PATH (for video / unsupported audio)
    librosa, pandas, psutil, pyloudnorm, soundfile, onnxruntime
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import dpdfnet
import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import pyloudnorm as pyln
import soundfile as sf


LOG = logging.getLogger(__name__)


def get_execution_provider():
    providers = ort.get_available_providers()

    if "CoreMLExecutionProvider" in providers:
        LOG.info(
            "Using Apple Neural Engine / GPU via CoreMLExecutionProvider"
        )
        return "CoreMLExecutionProvider"

    LOG.info("Using CPUExecutionProvider")
    return "CPUExecutionProvider"


DEFAULT_MODEL = "dpdfnet8_48khz_hr"
# DEFAULT_MODEL = "dpdfnet4"
DEFAULT_CSV_FILE = "benchmark_results.csv"
DEFAULT_OUTPUT_DIR = Path("tmp")

VOCALS_OUTPUT = "vocals.wav"
ACCOMPANIMENT_OUTPUT = "accompaniment.wav"
DEFAULT_CHUNK_SECONDS = 120
CHUNK_OVERLAP_SECONDS = 12

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
    ".amr",
    ".wma",
}


class PeakMemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.peak_mb = 0

    def _monitor(self):
        while self.running:
            rss = self.process.memory_info().rss / 1024 / 1024
            self.peak_mb = max(self.peak_mb, rss)
            time.sleep(0.05)

    def start(self):
        LOG.info("Starting memory monitor")
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        LOG.info("Stopping memory monitor")
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()


def model_sample_rate(model_name: str) -> int:
    if "48khz" in model_name.lower():
        return 48000
    return 16000


def build_output_dir(root_output_dir: str | Path, model_name: str, media_path: str | Path) -> Path:
    output_root = Path(root_output_dir)
    media_stem = Path(media_path).stem
    return output_root / model_name / media_stem


def get_audio_info(audio_path):
    LOG.info("Reading audio info from %s", audio_path)
    y, sr = librosa.load(str(audio_path), sr=None, mono=False)

    if y.ndim == 1:
        duration = len(y) / sr
        channels = 1
    else:
        duration = y.shape[-1] / sr
        channels = y.shape[0]

    info = {
        "duration": duration,
        "sample_rate": sr,
        "channels": channels,
    }
    LOG.info(
        "Audio info: duration=%.3fs, sample_rate=%s, channels=%s",
        info["duration"],
        info["sample_rate"],
        info["channels"],
    )
    return info


def load_audio_channels_first(audio_path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    return audio.astype(np.float32), sample_rate


def align_audio_to_mix(
    mix: np.ndarray,
    other: np.ndarray,
    mix_sr: int,
    other_sr: int,
) -> np.ndarray:
    """Resample and trim/pad `other` to match `mix` shape (channels, samples)."""
    if other.ndim == 1:
        other = other[np.newaxis, :]
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]

    mix_channels, mix_samples = mix.shape
    other_channels = other.shape[0]

    if other_channels != mix_channels:
        if mix_channels == 1:
            other = other.mean(axis=0, keepdims=True)
        elif other_channels == 1:
            other = np.repeat(other, mix_channels, axis=0)
        elif other_channels > mix_channels:
            other = other[:mix_channels, :]
        else:
            repeats = (mix_channels + other_channels - 1) // other_channels
            other = np.tile(other, (repeats, 1))[:mix_channels, :]

    if other_sr != mix_sr:
        other = librosa.resample(other, orig_sr=other_sr, target_sr=mix_sr, axis=-1)

    other_samples = other.shape[-1]
    if other_samples > mix_samples:
        other = other[:, :mix_samples]
    elif other_samples < mix_samples:
        pad_width = mix_samples - other_samples
        other = np.pad(other, ((0, 0), (0, pad_width)), mode="constant")

    return other.astype(np.float32)


def to_soundfile_format(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.T


def loudness_lufs(audio: np.ndarray, sample_rate: int) -> float:
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    meter = pyln.Meter(sample_rate)
    return float(meter.integrated_loudness(audio))


def write_stems(
    input_path: str | Path,
    enhanced_path: str | Path,
    output_dir: Path,
) -> tuple[Path, Path, float, float]:
    """Save enhanced speech as vocals and derive accompaniment as mix - vocals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = output_dir / VOCALS_OUTPUT
    accompaniment_path = output_dir / ACCOMPANIMENT_OUTPUT

    mix, mix_sr = load_audio_channels_first(input_path)
    vocals, vocals_sr = load_audio_channels_first(enhanced_path)
    vocals = align_audio_to_mix(mix, vocals, mix_sr, vocals_sr)
    accompaniment = mix - vocals

    sf.write(vocals_path, to_soundfile_format(vocals), mix_sr)
    sf.write(accompaniment_path, to_soundfile_format(accompaniment), mix_sr)

    input_lufs = loudness_lufs(mix, mix_sr)
    output_lufs = loudness_lufs(vocals, mix_sr)

    LOG.info("Wrote vocals to %s", vocals_path)
    LOG.info("Wrote accompaniment to %s", accompaniment_path)
    return vocals_path, accompaniment_path, input_lufs, output_lufs


def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def create_audio_chunk(
    input_file_path: str,
    output_file_path: str,
    start_time: float,
    end_time: float,
    *,
    sample_rate: int,
) -> str:
    duration = end_time - start_time
    if duration <= 0:
        msg = f"Invalid slice: start={start_time}, end={end_time}"
        raise ValueError(msg)

    _run_subprocess(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_time),
            "-i",
            input_file_path,
            "-t",
            str(duration),
            "-vn",
            "-threads",
            "0",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-fflags",
            "+discardcorrupt",
            "-err_detect",
            "ignore_err",
            "-y",
            output_file_path,
        ],
    )
    return output_file_path


def concatenate_wav_files(list_path: str, output_path: str) -> None:
    _run_subprocess(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            output_path,
        ],
    )
    Path(list_path).unlink(missing_ok=True)


def compute_chunk_boundaries(
    audio_length: float,
    chunk_seconds: float,
) -> list[tuple[float, float]]:
    chunks_number = max(1, int(audio_length // chunk_seconds) + 1)
    chunk_length = audio_length / chunks_number
    boundaries = [
        (idx * chunk_length, (idx + 1) * chunk_length) for idx in range(chunks_number)
    ]

    if (
        len(boundaries) > 1
        and (boundaries[-1][1] - boundaries[-1][0]) < CHUNK_OVERLAP_SECONDS
    ):
        boundaries[-2] = (boundaries[-2][0], boundaries[-1][1])
        boundaries.pop()

    return boundaries


def extract_audio_via_ffmpeg(media_path, output_path, *, sample_rate: int):
    LOG.info("Extracting audio from media via ffmpeg: %s", media_path)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg executable not found in PATH")

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        LOG.error("ffmpeg extraction failed: %s", result.stderr.strip())
        raise RuntimeError(
            f"ffmpeg extraction failed: {result.stderr.strip()}"
        )


def prepare_audio_input(media_path, temp_dir, *, sample_rate: int):
    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(media_path)

    if media_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
        LOG.info(
            "Media has supported audio extension %s, attempting direct load",
            media_path.suffix,
        )
        try:
            info = get_audio_info(media_path)
            if info["sample_rate"] != sample_rate or info["channels"] != 1:
                LOG.info(
                    "Resampling/converting to mono %s Hz via ffmpeg",
                    sample_rate,
                )
                raise ValueError("needs ffmpeg conversion")
            return media_path, None
        except Exception as exc:
            LOG.warning("Direct load failed for supported audio file: %s", exc)

    LOG.info("Media requires extraction to WAV via ffmpeg: %s", media_path)
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
        dir=temp_dir,
    )
    temp_file.close()
    extract_audio_via_ffmpeg(media_path, temp_file.name, sample_rate=sample_rate)
    get_audio_info(temp_file.name)
    return Path(temp_file.name), temp_file.name


def build_enhance_kwargs(
    *,
    sample_rate: int,
    model_name: str,
    attn_limit_db: float | None,
) -> dict:
    kwargs = {
        "sample_rate": sample_rate,
        "model": model_name,
    }
    if attn_limit_db is not None:
        kwargs["attn_limit_db"] = attn_limit_db
    return kwargs


def warmup_model(
    model_name: str,
    sample_rate: int,
    attn_limit_db: float | None,
    *,
    download: bool,
) -> float:
    LOG.info("Preparing DPDFNet model %s", model_name)
    load_start = time.perf_counter()

    if download:
        LOG.info("Downloading model weights for %s", model_name)
        dpdfnet.download(model_name)

    warmup_samples = int(sample_rate * 0.5)
    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    dpdfnet.enhance(
        warmup_audio,
        **build_enhance_kwargs(
            sample_rate=sample_rate,
            model_name=model_name,
            attn_limit_db=attn_limit_db,
        ),
    )

    load_end = time.perf_counter()
    model_load_time = load_end - load_start
    LOG.info("Model ready in %.3fs", model_load_time)
    return model_load_time


def enhance_audio_array(
    audio: np.ndarray,
    *,
    sample_rate: int,
    model_name: str,
    attn_limit_db: float | None,
) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return dpdfnet.enhance(
        audio.astype(np.float32),
        **build_enhance_kwargs(
            sample_rate=sample_rate,
            model_name=model_name,
            attn_limit_db=attn_limit_db,
        ),
    )


def enhance_file(
    input_file: str,
    output_file: str,
    *,
    model_name: str,
    sample_rate: int,
    attn_limit_db: float | None,
    duration_seconds: float,
    chunk_seconds: float,
    work_dir: Path,
) -> tuple[float, bool, int]:
    if duration_seconds <= chunk_seconds:
        LOG.info("Starting inference on %s", input_file)
        infer_start = time.perf_counter()
        audio, sr = sf.read(input_file)
        enhanced = enhance_audio_array(
            audio,
            sample_rate=sr,
            model_name=model_name,
            attn_limit_db=attn_limit_db,
        )
        sf.write(output_file, enhanced, sr)
        infer_end = time.perf_counter()
        LOG.info("Inference completed in %.3fs", infer_end - infer_start)
        return infer_end - infer_start, False, 1

    boundaries = compute_chunk_boundaries(duration_seconds, chunk_seconds)
    LOG.info(
        "Audio exceeds %.0fs; processing in %s chunks",
        chunk_seconds,
        len(boundaries),
    )
    print(
        f"Audio exceeds {chunk_seconds:.0f}s; "
        f"processing in {len(boundaries)} chunks..."
    )

    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    enhanced_list_path = chunks_dir / "enhanced_list.txt"
    enhanced_list_path.write_text("", encoding="utf-8")

    infer_start = time.perf_counter()
    for idx, (start, end) in enumerate(boundaries):
        chunk_path = chunks_dir / f"chunk_{idx}.wav"
        enhanced_chunk_path = chunks_dir / f"enhanced_{idx}.wav"
        create_audio_chunk(
            input_file,
            str(chunk_path),
            start,
            end,
            sample_rate=sample_rate,
        )
        if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
            msg = f"Chunk file was not created: {chunk_path}"
            raise FileNotFoundError(msg)

        LOG.info(
            "Enhancing chunk %s/%s (%.1fs-%.1fs)",
            idx + 1,
            len(boundaries),
            start,
            end,
        )
        audio, sr = sf.read(str(chunk_path))
        enhanced = enhance_audio_array(
            audio,
            sample_rate=sr,
            model_name=model_name,
            attn_limit_db=attn_limit_db,
        )
        sf.write(str(enhanced_chunk_path), enhanced, sr)
        if not enhanced_chunk_path.is_file():
            msg = f"Enhanced chunk missing: {enhanced_chunk_path}"
            raise FileNotFoundError(msg)

        with enhanced_list_path.open("a", encoding="utf-8") as enhanced_list:
            enhanced_list.write(f"file '{enhanced_chunk_path.resolve()}'\n")
        chunk_path.unlink(missing_ok=True)

    concatenate_wav_files(str(enhanced_list_path), output_file)
    infer_end = time.perf_counter()
    LOG.info("Chunked inference completed in %.3fs", infer_end - infer_start)
    return infer_end - infer_start, True, len(boundaries)


def run_model(
    model_name: str,
    input_file: str,
    enhanced_temp_file: str,
    *,
    sample_rate: int,
    attn_limit_db: float | None,
    duration_seconds: float,
    chunk_seconds: float,
    work_dir: Path,
    download: bool,
):
    model_load_time = warmup_model(
        model_name,
        sample_rate,
        attn_limit_db,
        download=download,
    )
    inference_time, chunked, chunk_count = enhance_file(
        input_file,
        enhanced_temp_file,
        model_name=model_name,
        sample_rate=sample_rate,
        attn_limit_db=attn_limit_db,
        duration_seconds=duration_seconds,
        chunk_seconds=chunk_seconds,
        work_dir=work_dir,
    )

    return {
        "model_load_time": model_load_time,
        "inference_time": inference_time,
        "chunked": chunked,
        "chunk_count": chunk_count,
    }


def benchmark(
    media_path,
    model_name,
    output_dir,
    csv_file,
    logs_dir,
    *,
    attn_limit_db=None,
    chunk_seconds=DEFAULT_CHUNK_SECONDS,
    download=False,
):
    LOG.info("Benchmark started")

    provider = get_execution_provider()

    sample_rate = model_sample_rate(model_name)
    audio_path, extracted_temp = prepare_audio_input(
        media_path,
        logs_dir,
        sample_rate=sample_rate,
    )
    audio_info = get_audio_info(audio_path)

    output_dir = build_output_dir(output_dir, model_name, media_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    enhanced_temp = output_dir / "_enhanced_temp.wav"

    monitor = PeakMemoryMonitor()
    monitor.start()

    total_start = time.perf_counter()
    result = run_model(
        model_name,
        str(audio_path),
        str(enhanced_temp),
        sample_rate=sample_rate,
        attn_limit_db=attn_limit_db,
        duration_seconds=audio_info["duration"],
        chunk_seconds=chunk_seconds,
        work_dir=output_dir,
        download=download,
    )
    vocals_path, accompaniment_path, input_lufs, output_lufs = write_stems(
        audio_path,
        enhanced_temp,
        output_dir,
    )
    if enhanced_temp.exists():
        enhanced_temp.unlink()

    total_end = time.perf_counter()
    monitor.stop()

    if extracted_temp and Path(extracted_temp).exists():
        LOG.info("Removing temporary extracted audio: %s", extracted_temp)
        os.unlink(extracted_temp)

    total_time = total_end - total_start
    duration = audio_info["duration"]
    rtf = total_time / duration if duration > 0 else None
    vocals_size_mb = vocals_path.stat().st_size / (1024 * 1024)
    accompaniment_size_mb = accompaniment_path.stat().st_size / (1024 * 1024)

    benchmark_result = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "device": provider,
        "input_file": str(media_path),
        "output_dir": str(output_dir),
        "vocals_output": str(vocals_path),
        "accompaniment_output": str(accompaniment_path),
        "duration_sec": round(duration, 3),
        "sample_rate": audio_info["sample_rate"],
        "model_sample_rate": sample_rate,
        "channels": audio_info["channels"],
        "attn_limit_db": attn_limit_db,
        "chunked": result.get("chunked", False),
        "chunk_count": result.get("chunk_count", 1),
        "chunk_seconds": chunk_seconds,
        "model_load_time_sec": round(result["model_load_time"], 3),
        "inference_time_sec": round(result.get("inference_time", 0), 3),
        "total_time_sec": round(total_time, 3),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "speed_factor": round(1 / rtf, 2) if rtf and rtf > 0 else None,
        "peak_memory_mb": round(monitor.peak_mb, 2),
        "input_lufs": round(input_lufs, 2),
        "output_lufs": round(output_lufs, 2),
        "vocals_size_mb": round(vocals_size_mb, 2),
        "accompaniment_size_mb": round(accompaniment_size_mb, 2),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(benchmark_result, indent=2), encoding="utf-8")

    LOG.info("Model load time: %.3fs", result.get("model_load_time", 0))
    LOG.info("Inference time: %.3fs", result.get("inference_time", 0))
    LOG.info("Total time: %.3fs", total_time)
    if duration > 0:
        LOG.info("RTF (real-time factor): %.4f", rtf)
    LOG.info("Peak memory (MB): %.2f", monitor.peak_mb)
    LOG.info("Input LUFS: %.2f", input_lufs)
    LOG.info("Output LUFS: %.2f", output_lufs)
    LOG.info("Benchmark completed")
    return benchmark_result


def save_result(result, csv_path):
    csv_path = Path(csv_path)
    existing = pd.read_csv(csv_path) if csv_path.exists() else None
    df = pd.DataFrame([result])
    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    LOG.info("Saved benchmark result to %s", csv_path)


def configure_logging(level):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark DPDFNet speech enhancement on local media files.",
    )
    parser.add_argument(
        "media",
        help="Input audio or video file to process",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"DPDFNet model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--attn-limit-db",
        type=float,
        default=None,
        help="Optional attention limit in dB passed to dpdfnet.enhance()",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model weights before benchmarking",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Base directory for outputs; final output is written to "
            "<output-dir>/<model>/<media_name>"
        ),
    )
    parser.add_argument(
        "--csv-file",
        default=DEFAULT_CSV_FILE,
        help="CSV file to append benchmark results",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=(
            "Split longer audio into chunks of this many seconds before inference "
            f"(default: {DEFAULT_CHUNK_SECONDS}). Helps avoid OOM on long files."
        ),
    )
    parser.add_argument(
        "--temp-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for temporary audio extraction files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(args.log_level)

    LOG.info("Starting DPDFNet benchmark script")
    LOG.info("media=%s model=%s", args.media, args.model)

    try:
        result = benchmark(
            args.media,
            args.model,
            args.output_dir,
            args.csv_file,
            args.temp_dir,
            attn_limit_db=args.attn_limit_db,
            chunk_seconds=args.chunk_seconds,
            download=args.download,
        )

        save_result(result, args.csv_file)

        print("\nBenchmark Result\n")
        for key, value in result.items():
            print(f"{key}: {value}")
        print(f"\nCSV updated: {args.csv_file}")

    except Exception as exc:
        LOG.error("Benchmark failed: %s", exc)
        print(f"Benchmark failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()











# #!/usr/bin/env python3
# """Benchmark DPDFNet speech enhancement on local media files.

# Enhanced speech is written to vocals.wav; accompaniment is derived as mix - vocals
# (same approach as ClearVoice / DeepFilterNet / ZipEnhancer benchmarks).

# Usage:
#     ./scripts/run_benchmark_dpdfnet.sh path/to/media.mp4

#     # Or manually from the isolated env:
#     cd benchmark/dpdfnet && uv sync
#     uv run python ../../scripts/benchmark_dpdfnet.py path/to/audio.wav \\
#         --model dpdfnet8_48khz_hr

# Requires:
#     benchmark/dpdfnet venv (dpdfnet; separate from main project)
#     ffmpeg on PATH (for video / unsupported audio)
#     librosa, pandas, psutil, pyloudnorm, soundfile, torch, onnxruntime

# On Mac, auto device selection prefers MPS and maps to ONNX CoreMLExecutionProvider.
# Use --cpu to force CPUExecutionProvider.
# """

# from __future__ import annotations

# import argparse
# import inspect
# import json
# import logging
# import os
# import shutil
# import subprocess
# import sys
# import tempfile
# import threading
# import time
# from datetime import datetime
# from pathlib import Path

# import dpdfnet
# import librosa
# import numpy as np
# import onnxruntime as ort
# import pandas as pd
# import psutil
# import pyloudnorm as pyln
# import soundfile as sf
# import torch


# LOG = logging.getLogger(__name__)

# DEFAULT_MODEL = "dpdfnet4"
# DEFAULT_CSV_FILE = "benchmark_results.csv"
# DEFAULT_OUTPUT_DIR = Path("tmp/dpdfnet_benchmark")
# DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VOCALS_OUTPUT = "vocals.wav"
# ACCOMPANIMENT_OUTPUT = "accompaniment.wav"
# DEFAULT_CHUNK_SECONDS = 120
# CHUNK_OVERLAP_SECONDS = 12

# SUPPORTED_AUDIO_EXTENSIONS = {
#     ".wav",
#     ".flac",
#     ".mp3",
#     ".m4a",
#     ".aac",
#     ".ogg",
#     ".opus",
#     ".aiff",
#     ".aif",
#     ".amr",
#     ".wma",
# }


# class PeakMemoryMonitor:
#     def __init__(self):
#         self.process = psutil.Process(os.getpid())
#         self.running = False
#         self.peak_mb = 0

#     def _monitor(self):
#         while self.running:
#             rss = self.process.memory_info().rss / 1024 / 1024
#             self.peak_mb = max(self.peak_mb, rss)
#             time.sleep(0.05)

#     def start(self):
#         LOG.info("Starting memory monitor")
#         self.running = True
#         self.thread = threading.Thread(target=self._monitor)
#         self.thread.daemon = True
#         self.thread.start()

#     def stop(self):
#         LOG.info("Stopping memory monitor")
#         self.running = False
#         if hasattr(self, "thread"):
#             self.thread.join()


# def model_sample_rate(model_name: str) -> int:
#     if "48khz" in model_name.lower():
#         return 48000
#     return 16000


# def resolve_device(*, force_cpu: bool = False) -> str:
#     if force_cpu:
#         LOG.info("Device detected: CPU (--cpu requested)")
#         return "cpu"

#     if torch.backends.mps.is_available():
#         LOG.info("Device detected: MPS")
#         return "mps"

#     if torch.cuda.is_available():
#         LOG.info("Device detected: CUDA")
#         return "cuda"

#     LOG.info("Device detected: CPU")
#     return "cpu"


# def format_device_label(device: str) -> str:
#     if device == "cuda":
#         return "CUDA"
#     if device == "mps":
#         return "MPS"
#     return "CPU"


# def resolve_onnx_providers(device: str) -> list[str | tuple[str, dict]]:
#     """Map PyTorch-style device labels to ONNX Runtime execution providers."""
#     available = set(ort.get_available_providers())
#     LOG.info("ONNX Runtime providers available: %s", sorted(available))

#     if device == "cpu":
#         return ["CPUExecutionProvider"]

#     if device == "mps":
#         if "CoreMLExecutionProvider" in available:
#             return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
#         LOG.warning(
#             "MPS requested but CoreMLExecutionProvider is unavailable; using CPU",
#         )
#         return ["CPUExecutionProvider"]

#     if device == "cuda":
#         if "CUDAExecutionProvider" in available:
#             return ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         LOG.warning(
#             "CUDA requested but CUDAExecutionProvider is unavailable; using CPU",
#         )
#         return ["CPUExecutionProvider"]

#     return ["CPUExecutionProvider"]


# def active_onnx_provider(providers: list[str | tuple[str, dict]]) -> str:
#     if not providers:
#         return "CPUExecutionProvider"
#     first = providers[0]
#     if isinstance(first, tuple):
#         return first[0]
#     return first


# def filter_enhance_kwargs(kwargs: dict) -> dict:
#     try:
#         params = inspect.signature(dpdfnet.enhance).parameters
#     except (TypeError, ValueError):
#         return kwargs

#     filtered = {key: value for key, value in kwargs.items() if key in params}
#     dropped = set(kwargs) - set(filtered)
#     if dropped:
#         provider_keys = {"providers", "execution_providers"}
#         if dropped & provider_keys and not (set(filtered) & provider_keys):
#             LOG.warning(
#                 "dpdfnet.enhance() does not accept execution providers; "
#                 "inference may remain on CPU",
#             )
#         else:
#             LOG.debug("Dropped unsupported dpdfnet.enhance kwargs: %s", sorted(dropped))
#     return filtered


# def get_audio_info(audio_path):
#     LOG.info("Reading audio info from %s", audio_path)
#     y, sr = librosa.load(str(audio_path), sr=None, mono=False)

#     if y.ndim == 1:
#         duration = len(y) / sr
#         channels = 1
#     else:
#         duration = y.shape[-1] / sr
#         channels = y.shape[0]

#     info = {
#         "duration": duration,
#         "sample_rate": sr,
#         "channels": channels,
#     }
#     LOG.info(
#         "Audio info: duration=%.3fs, sample_rate=%s, channels=%s",
#         info["duration"],
#         info["sample_rate"],
#         info["channels"],
#     )
#     return info


# def load_audio_channels_first(audio_path: str | Path) -> tuple[np.ndarray, int]:
#     audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
#     if audio.ndim == 1:
#         audio = audio[np.newaxis, :]
#     return audio.astype(np.float32), sample_rate


# def align_audio_to_mix(
#     mix: np.ndarray,
#     other: np.ndarray,
#     mix_sr: int,
#     other_sr: int,
# ) -> np.ndarray:
#     """Resample and trim/pad `other` to match `mix` shape (channels, samples)."""
#     if other.ndim == 1:
#         other = other[np.newaxis, :]
#     if mix.ndim == 1:
#         mix = mix[np.newaxis, :]

#     mix_channels, mix_samples = mix.shape
#     other_channels = other.shape[0]

#     if other_channels != mix_channels:
#         if mix_channels == 1:
#             other = other.mean(axis=0, keepdims=True)
#         elif other_channels == 1:
#             other = np.repeat(other, mix_channels, axis=0)
#         elif other_channels > mix_channels:
#             other = other[:mix_channels, :]
#         else:
#             repeats = (mix_channels + other_channels - 1) // other_channels
#             other = np.tile(other, (repeats, 1))[:mix_channels, :]

#     if other_sr != mix_sr:
#         other = librosa.resample(other, orig_sr=other_sr, target_sr=mix_sr, axis=-1)

#     other_samples = other.shape[-1]
#     if other_samples > mix_samples:
#         other = other[:, :mix_samples]
#     elif other_samples < mix_samples:
#         pad_width = mix_samples - other_samples
#         other = np.pad(other, ((0, 0), (0, pad_width)), mode="constant")

#     return other.astype(np.float32)


# def to_soundfile_format(audio: np.ndarray) -> np.ndarray:
#     if audio.ndim == 1:
#         return audio
#     return audio.T


# def loudness_lufs(audio: np.ndarray, sample_rate: int) -> float:
#     if audio.ndim > 1:
#         audio = audio.mean(axis=0)
#     meter = pyln.Meter(sample_rate)
#     return float(meter.integrated_loudness(audio))


# def write_stems(
#     input_path: str | Path,
#     enhanced_path: str | Path,
#     output_dir: Path,
# ) -> tuple[Path, Path, float, float]:
#     """Save enhanced speech as vocals and derive accompaniment as mix - vocals."""
#     output_dir.mkdir(parents=True, exist_ok=True)
#     vocals_path = output_dir / VOCALS_OUTPUT
#     accompaniment_path = output_dir / ACCOMPANIMENT_OUTPUT

#     mix, mix_sr = load_audio_channels_first(input_path)
#     vocals, vocals_sr = load_audio_channels_first(enhanced_path)
#     vocals = align_audio_to_mix(mix, vocals, mix_sr, vocals_sr)
#     accompaniment = mix - vocals

#     sf.write(vocals_path, to_soundfile_format(vocals), mix_sr)
#     sf.write(accompaniment_path, to_soundfile_format(accompaniment), mix_sr)

#     input_lufs = loudness_lufs(mix, mix_sr)
#     output_lufs = loudness_lufs(vocals, mix_sr)

#     LOG.info("Wrote vocals to %s", vocals_path)
#     LOG.info("Wrote accompaniment to %s", accompaniment_path)
#     return vocals_path, accompaniment_path, input_lufs, output_lufs


# def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
#     return subprocess.run(cmd, capture_output=True, text=True, check=True)


# def create_audio_chunk(
#     input_file_path: str,
#     output_file_path: str,
#     start_time: float,
#     end_time: float,
#     *,
#     sample_rate: int,
# ) -> str:
#     duration = end_time - start_time
#     if duration <= 0:
#         msg = f"Invalid slice: start={start_time}, end={end_time}"
#         raise ValueError(msg)

#     _run_subprocess(
#         [
#             "ffmpeg",
#             "-hide_banner",
#             "-loglevel",
#             "error",
#             "-ss",
#             str(start_time),
#             "-i",
#             input_file_path,
#             "-t",
#             str(duration),
#             "-vn",
#             "-threads",
#             "0",
#             "-acodec",
#             "pcm_s16le",
#             "-ar",
#             str(sample_rate),
#             "-ac",
#             "1",
#             "-fflags",
#             "+discardcorrupt",
#             "-err_detect",
#             "ignore_err",
#             "-y",
#             output_file_path,
#         ],
#     )
#     return output_file_path


# def concatenate_wav_files(list_path: str, output_path: str) -> None:
#     _run_subprocess(
#         [
#             "ffmpeg",
#             "-loglevel",
#             "error",
#             "-y",
#             "-f",
#             "concat",
#             "-safe",
#             "0",
#             "-i",
#             list_path,
#             "-c",
#             "copy",
#             output_path,
#         ],
#     )
#     Path(list_path).unlink(missing_ok=True)


# def compute_chunk_boundaries(
#     audio_length: float,
#     chunk_seconds: float,
# ) -> list[tuple[float, float]]:
#     chunks_number = max(1, int(audio_length // chunk_seconds) + 1)
#     chunk_length = audio_length / chunks_number
#     boundaries = [
#         (idx * chunk_length, (idx + 1) * chunk_length) for idx in range(chunks_number)
#     ]

#     if (
#         len(boundaries) > 1
#         and (boundaries[-1][1] - boundaries[-1][0]) < CHUNK_OVERLAP_SECONDS
#     ):
#         boundaries[-2] = (boundaries[-2][0], boundaries[-1][1])
#         boundaries.pop()

#     return boundaries


# def extract_audio_via_ffmpeg(media_path, output_path, *, sample_rate: int):
#     LOG.info("Extracting audio from media via ffmpeg: %s", media_path)
#     ffmpeg_path = shutil.which("ffmpeg")
#     if ffmpeg_path is None:
#         raise RuntimeError("ffmpeg executable not found in PATH")

#     command = [
#         ffmpeg_path,
#         "-y",
#         "-hide_banner",
#         "-loglevel",
#         "error",
#         "-i",
#         str(media_path),
#         "-vn",
#         "-acodec",
#         "pcm_s16le",
#         "-ar",
#         str(sample_rate),
#         "-ac",
#         "1",
#         str(output_path),
#     ]
#     result = subprocess.run(
#         command,
#         capture_output=True,
#         text=True,
#     )
#     if result.returncode != 0:
#         LOG.error("ffmpeg extraction failed: %s", result.stderr.strip())
#         raise RuntimeError(
#             f"ffmpeg extraction failed: {result.stderr.strip()}"
#         )


# def prepare_audio_input(media_path, temp_dir, *, sample_rate: int):
#     media_path = Path(media_path)
#     if not media_path.exists():
#         raise FileNotFoundError(media_path)

#     if media_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
#         LOG.info(
#             "Media has supported audio extension %s, attempting direct load",
#             media_path.suffix,
#         )
#         try:
#             info = get_audio_info(media_path)
#             if info["sample_rate"] != sample_rate or info["channels"] != 1:
#                 LOG.info(
#                     "Resampling/converting to mono %s Hz via ffmpeg",
#                     sample_rate,
#                 )
#                 raise ValueError("needs ffmpeg conversion")
#             return media_path, None
#         except Exception as exc:
#             LOG.warning("Direct load failed for supported audio file: %s", exc)

#     LOG.info("Media requires extraction to WAV via ffmpeg: %s", media_path)
#     temp_file = tempfile.NamedTemporaryFile(
#         suffix=".wav",
#         delete=False,
#         dir=temp_dir,
#     )
#     temp_file.close()
#     extract_audio_via_ffmpeg(media_path, temp_file.name, sample_rate=sample_rate)
#     get_audio_info(temp_file.name)
#     return Path(temp_file.name), temp_file.name


# def build_enhance_kwargs(
#     *,
#     sample_rate: int,
#     model_name: str,
#     attn_limit_db: float | None,
#     providers: list[str | tuple[str, dict]],
# ) -> dict:
#     kwargs = {
#         "sample_rate": sample_rate,
#         "model": model_name,
#         "providers": providers,
#         "execution_providers": providers,
#     }
#     if attn_limit_db is not None:
#         kwargs["attn_limit_db"] = attn_limit_db
#     return filter_enhance_kwargs(kwargs)


# def warmup_model(
#     model_name: str,
#     sample_rate: int,
#     attn_limit_db: float | None,
#     providers: list[str | tuple[str, dict]],
#     *,
#     download: bool,
# ) -> float:
#     LOG.info("Preparing DPDFNet model %s", model_name)
#     load_start = time.perf_counter()

#     if download:
#         LOG.info("Downloading model weights for %s", model_name)
#         dpdfnet.download(model_name)

#     warmup_samples = int(sample_rate * 0.5)
#     warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
#     dpdfnet.enhance(
#         warmup_audio,
#         **build_enhance_kwargs(
#             sample_rate=sample_rate,
#             model_name=model_name,
#             attn_limit_db=attn_limit_db,
#             providers=providers,
#         ),
#     )

#     load_end = time.perf_counter()
#     model_load_time = load_end - load_start
#     LOG.info("Model ready in %.3fs", model_load_time)
#     return model_load_time


# def enhance_audio_array(
#     audio: np.ndarray,
#     *,
#     sample_rate: int,
#     model_name: str,
#     attn_limit_db: float | None,
#     providers: list[str | tuple[str, dict]],
# ) -> np.ndarray:
#     if audio.ndim > 1:
#         audio = audio.mean(axis=-1)
#     return dpdfnet.enhance(
#         audio.astype(np.float32),
#         **build_enhance_kwargs(
#             sample_rate=sample_rate,
#             model_name=model_name,
#             attn_limit_db=attn_limit_db,
#             providers=providers,
#         ),
#     )


# def enhance_file(
#     input_file: str,
#     output_file: str,
#     *,
#     model_name: str,
#     sample_rate: int,
#     attn_limit_db: float | None,
#     providers: list[str | tuple[str, dict]],
#     duration_seconds: float,
#     chunk_seconds: float,
#     work_dir: Path,
# ) -> tuple[float, bool, int]:
#     if duration_seconds <= chunk_seconds:
#         LOG.info("Starting inference on %s", input_file)
#         infer_start = time.perf_counter()
#         audio, sr = sf.read(input_file)
#         enhanced = enhance_audio_array(
#             audio,
#             sample_rate=sr,
#             model_name=model_name,
#             attn_limit_db=attn_limit_db,
#             providers=providers,
#         )
#         sf.write(output_file, enhanced, sr)
#         infer_end = time.perf_counter()
#         LOG.info("Inference completed in %.3fs", infer_end - infer_start)
#         return infer_end - infer_start, False, 1

#     boundaries = compute_chunk_boundaries(duration_seconds, chunk_seconds)
#     LOG.info(
#         "Audio exceeds %.0fs; processing in %s chunks",
#         chunk_seconds,
#         len(boundaries),
#     )
#     print(
#         f"Audio exceeds {chunk_seconds:.0f}s; "
#         f"processing in {len(boundaries)} chunks..."
#     )

#     chunks_dir = work_dir / "chunks"
#     chunks_dir.mkdir(parents=True, exist_ok=True)
#     enhanced_list_path = chunks_dir / "enhanced_list.txt"
#     enhanced_list_path.write_text("", encoding="utf-8")

#     infer_start = time.perf_counter()
#     for idx, (start, end) in enumerate(boundaries):
#         chunk_path = chunks_dir / f"chunk_{idx}.wav"
#         enhanced_chunk_path = chunks_dir / f"enhanced_{idx}.wav"
#         create_audio_chunk(
#             input_file,
#             str(chunk_path),
#             start,
#             end,
#             sample_rate=sample_rate,
#         )
#         if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
#             msg = f"Chunk file was not created: {chunk_path}"
#             raise FileNotFoundError(msg)

#         LOG.info(
#             "Enhancing chunk %s/%s (%.1fs-%.1fs)",
#             idx + 1,
#             len(boundaries),
#             start,
#             end,
#         )
#         audio, sr = sf.read(str(chunk_path))
#         enhanced = enhance_audio_array(
#             audio,
#             sample_rate=sr,
#             model_name=model_name,
#             attn_limit_db=attn_limit_db,
#             providers=providers,
#         )
#         sf.write(str(enhanced_chunk_path), enhanced, sr)
#         if not enhanced_chunk_path.is_file():
#             msg = f"Enhanced chunk missing: {enhanced_chunk_path}"
#             raise FileNotFoundError(msg)

#         with enhanced_list_path.open("a", encoding="utf-8") as enhanced_list:
#             enhanced_list.write(f"file '{enhanced_chunk_path.resolve()}'\n")
#         chunk_path.unlink(missing_ok=True)

#     concatenate_wav_files(str(enhanced_list_path), output_file)
#     infer_end = time.perf_counter()
#     LOG.info("Chunked inference completed in %.3fs", infer_end - infer_start)
#     return infer_end - infer_start, True, len(boundaries)


# def run_model(
#     model_name: str,
#     input_file: str,
#     enhanced_temp_file: str,
#     *,
#     sample_rate: int,
#     attn_limit_db: float | None,
#     providers: list[str | tuple[str, dict]],
#     duration_seconds: float,
#     chunk_seconds: float,
#     work_dir: Path,
#     download: bool,
# ):
#     model_load_time = warmup_model(
#         model_name,
#         sample_rate,
#         attn_limit_db,
#         providers,
#         download=download,
#     )
#     inference_time, chunked, chunk_count = enhance_file(
#         input_file,
#         enhanced_temp_file,
#         model_name=model_name,
#         sample_rate=sample_rate,
#         attn_limit_db=attn_limit_db,
#         providers=providers,
#         duration_seconds=duration_seconds,
#         chunk_seconds=chunk_seconds,
#         work_dir=work_dir,
#     )

#     return {
#         "model_load_time": model_load_time,
#         "inference_time": inference_time,
#         "chunked": chunked,
#         "chunk_count": chunk_count,
#     }


# def benchmark(
#     media_path,
#     model_name,
#     output_dir,
#     csv_file,
#     logs_dir,
#     *,
#     attn_limit_db=None,
#     chunk_seconds=DEFAULT_CHUNK_SECONDS,
#     download=False,
#     force_cpu=False,
# ):
#     LOG.info("Benchmark started")

#     device = resolve_device(force_cpu=force_cpu)
#     providers = resolve_onnx_providers(device)
#     onnx_provider = active_onnx_provider(providers)
#     LOG.info(
#         "Using device=%s onnx_provider=%s",
#         format_device_label(device),
#         onnx_provider,
#     )

#     sample_rate = model_sample_rate(model_name)
#     audio_path, extracted_temp = prepare_audio_input(
#         media_path,
#         logs_dir,
#         sample_rate=sample_rate,
#     )
#     audio_info = get_audio_info(audio_path)

#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     enhanced_temp = output_dir / "_enhanced_temp.wav"

#     monitor = PeakMemoryMonitor()
#     monitor.start()

#     total_start = time.perf_counter()
#     result = run_model(
#         model_name,
#         str(audio_path),
#         str(enhanced_temp),
#         sample_rate=sample_rate,
#         attn_limit_db=attn_limit_db,
#         providers=providers,
#         duration_seconds=audio_info["duration"],
#         chunk_seconds=chunk_seconds,
#         work_dir=output_dir,
#         download=download,
#     )
#     vocals_path, accompaniment_path, input_lufs, output_lufs = write_stems(
#         audio_path,
#         enhanced_temp,
#         output_dir,
#     )
#     if enhanced_temp.exists():
#         enhanced_temp.unlink()

#     total_end = time.perf_counter()
#     monitor.stop()

#     if extracted_temp and Path(extracted_temp).exists():
#         LOG.info("Removing temporary extracted audio: %s", extracted_temp)
#         os.unlink(extracted_temp)

#     total_time = total_end - total_start
#     duration = audio_info["duration"]
#     rtf = total_time / duration if duration > 0 else None
#     vocals_size_mb = vocals_path.stat().st_size / (1024 * 1024)
#     accompaniment_size_mb = accompaniment_path.stat().st_size / (1024 * 1024)

#     benchmark_result = {
#         "timestamp": datetime.now().isoformat(),
#         "model": model_name,
#         "device": format_device_label(device),
#         "onnx_provider": onnx_provider,
#         "input_file": str(media_path),
#         "output_dir": str(output_dir),
#         "vocals_output": str(vocals_path),
#         "accompaniment_output": str(accompaniment_path),
#         "duration_sec": round(duration, 3),
#         "sample_rate": audio_info["sample_rate"],
#         "model_sample_rate": sample_rate,
#         "channels": audio_info["channels"],
#         "attn_limit_db": attn_limit_db,
#         "chunked": result.get("chunked", False),
#         "chunk_count": result.get("chunk_count", 1),
#         "chunk_seconds": chunk_seconds,
#         "model_load_time_sec": round(result["model_load_time"], 3),
#         "inference_time_sec": round(result.get("inference_time", 0), 3),
#         "total_time_sec": round(total_time, 3),
#         "rtf": round(rtf, 4) if rtf is not None else None,
#         "speed_factor": round(1 / rtf, 2) if rtf and rtf > 0 else None,
#         "peak_memory_mb": round(monitor.peak_mb, 2),
#         "input_lufs": round(input_lufs, 2),
#         "output_lufs": round(output_lufs, 2),
#         "vocals_size_mb": round(vocals_size_mb, 2),
#         "accompaniment_size_mb": round(accompaniment_size_mb, 2),
#     }

#     metrics_path = output_dir / "metrics.json"
#     metrics_path.write_text(json.dumps(benchmark_result, indent=2), encoding="utf-8")

#     LOG.info("Model load time: %.3fs", result.get("model_load_time", 0))
#     LOG.info("Inference time: %.3fs", result.get("inference_time", 0))
#     LOG.info("Total time: %.3fs", total_time)
#     if duration > 0:
#         LOG.info("RTF (real-time factor): %.4f", rtf)
#     LOG.info("Peak memory (MB): %.2f", monitor.peak_mb)
#     LOG.info("Input LUFS: %.2f", input_lufs)
#     LOG.info("Output LUFS: %.2f", output_lufs)
#     LOG.info("Benchmark completed")
#     return benchmark_result


# def save_result(result, csv_path):
#     csv_path = Path(csv_path)
#     existing = pd.read_csv(csv_path) if csv_path.exists() else None
#     df = pd.DataFrame([result])
#     if existing is not None:
#         df = pd.concat([existing, df], ignore_index=True)
#     df.to_csv(csv_path, index=False)
#     LOG.info("Saved benchmark result to %s", csv_path)


# def configure_logging(level):
#     logging.basicConfig(
#         level=level,
#         format="%(asctime)s %(levelname)s: %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Benchmark DPDFNet speech enhancement on local media files.",
#     )
#     parser.add_argument(
#         "media",
#         help="Input audio or video file to process",
#     )
#     parser.add_argument(
#         "--model",
#         default=DEFAULT_MODEL,
#         help=f"DPDFNet model name (default: {DEFAULT_MODEL})",
#     )
#     parser.add_argument(
#         "--attn-limit-db",
#         type=float,
#         default=None,
#         help="Optional attention limit in dB passed to dpdfnet.enhance()",
#     )
#     parser.add_argument(
#         "--download",
#         action="store_true",
#         help="Download model weights before benchmarking",
#     )
#     parser.add_argument(
#         "--cpu",
#         action="store_true",
#         help="Force CPU inference (default: auto — MPS on Mac, else CUDA, else CPU)",
#     )
#     parser.add_argument(
#         "--output-dir",
#         default=str(DEFAULT_OUTPUT_DIR),
#         help="Directory to write vocals.wav and accompaniment.wav",
#     )
#     parser.add_argument(
#         "--csv-file",
#         default=DEFAULT_CSV_FILE,
#         help="CSV file to append benchmark results",
#     )
#     parser.add_argument(
#         "--log-level",
#         default="INFO",
#         choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
#         help="Logging verbosity level",
#     )
#     parser.add_argument(
#         "--chunk-seconds",
#         type=float,
#         default=DEFAULT_CHUNK_SECONDS,
#         help=(
#             "Split longer audio into chunks of this many seconds before inference "
#             f"(default: {DEFAULT_CHUNK_SECONDS}). Helps avoid OOM on long files."
#         ),
#     )
#     parser.add_argument(
#         "--temp-dir",
#         default=str(DEFAULT_OUTPUT_DIR),
#         help="Directory for temporary audio extraction files",
#     )
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     configure_logging(args.log_level)

#     LOG.info("Starting DPDFNet benchmark script")
#     LOG.info("media=%s model=%s", args.media, args.model)
#     LOG.info(
#         "PyTorch backends: mps=%s, cuda=%s",
#         torch.backends.mps.is_available(),
#         torch.cuda.is_available(),
#     )

#     try:
#         result = benchmark(
#             args.media,
#             args.model,
#             args.output_dir,
#             args.csv_file,
#             args.temp_dir,
#             attn_limit_db=args.attn_limit_db,
#             chunk_seconds=args.chunk_seconds,
#             download=args.download,
#             force_cpu=args.cpu,
#         )

#         save_result(result, args.csv_file)

#         print("\nBenchmark Result\n")
#         for key, value in result.items():
#             print(f"{key}: {value}")
#         print(f"\nCSV updated: {args.csv_file}")

#     except Exception as exc:
#         LOG.error("Benchmark failed: %s", exc)
#         print(f"Benchmark failed: {exc}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
