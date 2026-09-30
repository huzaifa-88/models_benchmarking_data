"""Benchmark EasyOCR and PaddleOCR on a PDF or image.

From the ``ocr`` project directory, install the dependencies and run:

    uv sync
    uv run python benchmark_ocr.py /path/to/document.pdf --output my_results.json

Run a specific engine or an image input:

    uv run python benchmark_ocr.py /path/to/document.pdf --engine easy
    uv run python benchmark_ocr.py /path/to/image.png --engine paddle

Available engines are ``easy``, ``paddle``, and ``both`` (the default). Other
options include ``--dpi`` for PDF rendering, ``--gpu`` for EasyOCR, ``--languages``
for EasyOCR language codes, and ``--output`` to choose the results JSON path. By
default, results are saved as ``benchmark_results.json`` in the current directory,
overwriting that file if it already exists. Use ``uv run python benchmark_ocr.py
--help`` to see all options.
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
import numpy as np

import fitz  # PyMuPDF
from PIL import Image
import tempfile
from pathlib import Path


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

logger = logging.getLogger("ocr-benchmark")


# ============================================================
# Utilities
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


def ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def get_input_type(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "pdf"

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return "image"

    raise ValueError(f"Unsupported file type: {path.suffix}")


# ============================================================
# PDF -> Images
# ============================================================

def pdf_to_images(pdf_path: Path, dpi: int = 150):
    """
    Render PDF pages to temporary PNG files.

    Returns:
        image_paths,
        rendering_time
    """

    logger.info("Opening PDF: %s", pdf_path)

    start = time.perf_counter()

    document = fitz.open(pdf_path)

    image_paths = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_benchmark_"))

    for page_number, page in enumerate(document, start=1):

        page_start = time.perf_counter()

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        output_path = temp_dir / f"page_{page_number}.png"

        pixmap.save(str(output_path))

        image_paths.append(output_path)

        logger.info(
            "Rendered page %d | %.2f ms | %dx%d | %s",
            page_number,
            ms(time.perf_counter() - page_start),
            pixmap.width,
            pixmap.height,
            output_path,
        )

    document.close()

    total = time.perf_counter() - start

    logger.info(
        "PDF rendering completed | pages=%d | %.2f ms",
        len(image_paths),
        ms(total),
    )

    return image_paths, total


# def load_input(path: Path, dpi: int = 150):
    """
    Returns:
        images,
        preprocessing_time
    """

    input_type = get_input_type(path)

    if input_type == "image":
        start = time.perf_counter()

        image = Image.open(path).convert("RGB")

        elapsed = time.perf_counter() - start

        logger.info(
            "Loaded image | %.2f ms | %dx%d",
            ms(elapsed),
            image.width,
            image.height,
        )

        return [image], elapsed

    return pdf_to_images(path, dpi=dpi)
def load_input(path: Path, dpi: int = 150):

    input_type = get_input_type(path)

    if input_type == "image":

        logger.info("Using image directly: %s", path)

        return [path], 0

    return pdf_to_images(
        path,
        dpi=dpi,
    )

# ============================================================
# EasyOCR
# ============================================================

def run_easyocr(images, languages, gpu=False):
    logger.info("=" * 70)
    logger.info("INITIALIZING EASYOCR")
    logger.info("=" * 70)

    initialization_start = time.perf_counter()

    import easyocr

    reader = easyocr.Reader(
        languages,
        gpu=gpu,
    )

    initialization_time = time.perf_counter() - initialization_start

    logger.info(
        "EasyOCR model initialization: %.2f ms",
        ms(initialization_time),
    )

    results = []
    inference_times = []

    logger.info("=" * 70)
    logger.info("RUNNING EASYOCR")
    logger.info("=" * 70)

    for page_number, image in enumerate(images, start=1):

        start = time.perf_counter()

        result = reader.readtext(
            np.array(image),
            detail=1,
        )

        elapsed = time.perf_counter() - start

        inference_times.append(elapsed)

        page_text = []

        for detection in result:
            if len(detection) >= 2:
                text = detection[1]
                confidence = detection[2] if len(detection) >= 3 else None

                page_text.append(
                    {
                        "text": text,
                        "confidence": confidence,
                    }
                )

        results.append(page_text)

        logger.info(
            "EasyOCR | page=%d | %.2f ms | detections=%d",
            page_number,
            ms(elapsed),
            len(page_text),
        )

    total_inference = sum(inference_times)

    logger.info(
        "EasyOCR inference completed | %.2f ms",
        ms(total_inference),
    )

    return {
        "engine": "easyocr",
        "initialization_time_ms": ms(initialization_time),
        "inference_time_ms": ms(total_inference),
        "average_page_time_ms": ms(
            statistics.mean(inference_times)
        ) if inference_times else 0,
        "pages": results,
    }


# ============================================================
# PaddleOCR
# ============================================================

# def run_paddleocr(image_paths):
    logger.info("=" * 70)
    logger.info("INITIALIZING PADDLEOCR")
    logger.info("=" * 70)

    initialization_start = time.perf_counter()

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
    )

    initialization_time = time.perf_counter() - initialization_start

    logger.info(
        "PaddleOCR model initialization: %.2f ms",
        ms(initialization_time),
    )

    results = []
    inference_times = []

    logger.info("=" * 70)
    logger.info("RUNNING PADDLEOCR")
    logger.info("=" * 70)

    for page_number, image_path in enumerate(image_paths, start=1):

        start = time.perf_counter()

        result = ocr.predict(str(image_path))

        elapsed = time.perf_counter() - start

        inference_times.append(elapsed)

        page_results = []

        for res in result:
            try:
                data = res.json

                if callable(data):
                    data = data()

                if isinstance(data, str):
                    data = json.loads(data)

                page_results.append(data)

                # Print recognized text immediately
                if isinstance(data, dict):
                    texts = data.get("rec_texts", [])

                    if texts:
                        logger.info(
                            "PaddleOCR | page=%d | detected_lines=%d",
                            page_number,
                            len(texts),
                        )

                        for text in texts:
                            logger.info("TEXT: %s", text)

            except Exception as exc:
                logger.warning(
                    "Could not parse PaddleOCR result: %s",
                    exc,
                )

                page_results.append(str(res))

        results.append(page_results)

        logger.info(
            "PaddleOCR | page=%d | %.2f ms",
            page_number,
            ms(elapsed),
        )

    total_inference = sum(inference_times)

    logger.info(
        "PaddleOCR inference completed | %.2f ms",
        ms(total_inference),
    )

    return {
        "engine": "paddleocr",
        "initialization_time_ms": ms(initialization_time),
        "inference_time_ms": ms(total_inference),
        "average_page_time_ms": ms(
            statistics.mean(inference_times)
        ) if inference_times else 0,
        "pages": results,
    }
def run_paddleocr(image_paths):
    logger.info("=" * 70)
    logger.info("INITIALIZING PADDLEOCR")
    logger.info("=" * 70)

    initialization_start = time.perf_counter()

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
    )

    initialization_time = time.perf_counter() - initialization_start

    logger.info(
        "PaddleOCR model initialization: %.2f ms",
        ms(initialization_time),
    )

    results = []
    inference_times = []

    logger.info("=" * 70)
    logger.info("RUNNING PADDLEOCR")
    logger.info("=" * 70)

    for page_number, image_path in enumerate(image_paths, start=1):

        logger.info(
            "Processing page %d: %s",
            page_number,
            image_path,
        )

        start = time.perf_counter()

        result = ocr.predict(str(image_path))

        elapsed = time.perf_counter() - start

        inference_times.append(elapsed)

        page_results = []

        for res in result:

            try:
                data = res.json

                if callable(data):
                    data = data()

                if isinstance(data, str):
                    data = json.loads(data)

                page_results.append(data)

                # Print recognized text
                if isinstance(data, dict):

                    texts = data.get("rec_texts", [])
                    scores = data.get("rec_scores", [])

                    logger.info(
                        "PaddleOCR | page=%d | detected_lines=%d",
                        page_number,
                        len(texts),
                    )

                    for index, text in enumerate(texts):

                        confidence = (
                            scores[index]
                            if index < len(scores)
                            else None
                        )

                        logger.info(
                            "TEXT: %s | confidence=%s",
                            text,
                            confidence,
                        )

            except Exception as exc:

                logger.warning(
                    "Could not parse PaddleOCR result: %s",
                    exc,
                )

                page_results.append(str(res))

        results.append(page_results)

        logger.info(
            "PaddleOCR | page=%d | %.2f ms",
            page_number,
            ms(elapsed),
        )

    total_inference = sum(inference_times)

    logger.info(
        "PaddleOCR inference completed | %.2f ms",
        ms(total_inference),
    )

    return {
        "engine": "paddleocr",
        "initialization_time_ms": ms(initialization_time),
        "inference_time_ms": ms(total_inference),
        "average_page_time_ms": ms(
            statistics.mean(inference_times)
        ) if inference_times else 0,
        "pages": results,
    }

# ============================================================
# Text extraction
# ============================================================

def extract_easyocr_text(result):
    lines = []

    for page in result["pages"]:
        for item in page:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    lines.append(text)

    return "\n".join(lines)


def extract_paddle_text(result):
    """
    Attempts to extract recognized text from PaddleOCR's
    structured result.
    """

    texts = []

    def recursive_extract(value):

        if isinstance(value, dict):

            # Common PaddleOCR result key
            if "rec_texts" in value:
                rec_texts = value["rec_texts"]

                if isinstance(rec_texts, list):
                    for text in rec_texts:
                        if text:
                            texts.append(str(text))

            for child in value.values():
                recursive_extract(child)

        elif isinstance(value, list):

            for child in value:
                recursive_extract(child)

    recursive_extract(result["pages"])

    return "\n".join(texts)


# ============================================================
# Main benchmark
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PaddleOCR or EasyOCR on an image/PDF."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to PDF or image",
    )

    parser.add_argument(
        "--engine",
        choices=["paddle", "easy", "both"],
        default="both",
        help="OCR engine to benchmark",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF rendering DPI",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for EasyOCR",
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="EasyOCR languages, e.g. en fr",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output JSON file",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    logger.info("=" * 70)
    logger.info("OCR BENCHMARK")
    logger.info("=" * 70)

    logger.info("Input: %s", input_path)
    logger.info("Engine: %s", args.engine)
    logger.info("DPI: %d", args.dpi)

    # --------------------------------------------------------
    # Load / render input
    # --------------------------------------------------------

    if get_input_type(input_path) == "image":
        images = [input_path]
        preprocessing_time = 0

    else:
        images, preprocessing_time = pdf_to_images(
            input_path,
            dpi=args.dpi,
        )

    logger.info(
        "Input preparation: %.2f ms",
        ms(preprocessing_time),
    )

    logger.info("Pages/images: %d", len(images))

    benchmark_results = {
        "input": str(input_path),
        "input_type": get_input_type(input_path),
        "dpi": args.dpi,
        "page_count": len(images),
        "preprocessing_time_ms": ms(preprocessing_time),
        "engines": {},
    }

    # --------------------------------------------------------
    # EasyOCR
    # --------------------------------------------------------

    if args.engine in ("easy", "both"):

        easy_result = run_easyocr(
            images,
            languages=args.languages,
            gpu=args.gpu,
        )

        easy_text = extract_easyocr_text(easy_result)

        easy_result["text"] = easy_text

        benchmark_results["engines"]["easyocr"] = easy_result

    # --------------------------------------------------------
    # PaddleOCR
    # --------------------------------------------------------

    if args.engine in ("paddle", "both"):

        paddle_result = run_paddleocr(images)

        paddle_text = extract_paddle_text(paddle_result)

        paddle_result["text"] = paddle_text

        benchmark_results["engines"]["paddleocr"] = paddle_result

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)

    for engine_name, result in benchmark_results["engines"].items():

        inference_time = result["inference_time_ms"]
        pages = benchmark_results["page_count"]

        logger.info(
            "%-12s | inference=%10.2f ms | avg/page=%10.2f ms",
            engine_name,
            inference_time,
            result["average_page_time_ms"],
        )

        logger.info(
            "%-12s | initialization=%10.2f ms",
            engine_name,
            result["initialization_time_ms"],
        )

        logger.info(
            "%-12s | OCR total incl. init=%10.2f ms",
            engine_name,
            result["initialization_time_ms"]
            + inference_time,
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(
            benchmark_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("")
    logger.info(
        "Results written to: %s",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()