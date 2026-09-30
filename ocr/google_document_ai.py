import os
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai


# ============================================================
# Configuration
# ============================================================

GOOGLE_CREDENTIALS = Path(__file__).parent / "google.json"

# Put your test PDF/image here
INPUT_FILE = Path(__file__).parent / "test.pdf"

# IMPORTANT:
# Replace this after creating/finding your Document AI processor.
PROCESSOR_ID = "YOUR_PROCESSOR_ID"

# Example locations:
# "us"
# "eu"
PROCESSOR_LOCATION = "us"


# ============================================================
# Authentication
# ============================================================

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(GOOGLE_CREDENTIALS)


def get_client():
    """
    Create a Document AI client using the existing google.json.
    """

    client_options = ClientOptions(
        api_endpoint=f"{PROCESSOR_LOCATION}-documentai.googleapis.com"
    )

    client = documentai.DocumentProcessorServiceClient(
        client_options=client_options
    )

    return client


# ============================================================
# Process Document
# ============================================================

def process_document():
    print("=" * 70)
    print("Google Document AI - Enterprise Document OCR Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not GOOGLE_CREDENTIALS.exists():
        print(f"❌ google.json not found:")
        print(f"   {GOOGLE_CREDENTIALS}")
        return

    if not INPUT_FILE.exists():
        print(f"❌ Input file not found:")
        print(f"   {INPUT_FILE}")
        return

    if PROCESSOR_ID == "YOUR_PROCESSOR_ID":
        print("❌ PROCESSOR_ID has not been configured.")
        print()
        print("Create an Enterprise Document OCR processor in")
        print("Google Cloud Document AI and put its processor ID here.")
        return

    # --------------------------------------------------------
    # Load project ID from credentials
    # --------------------------------------------------------

    import json

    with open(GOOGLE_CREDENTIALS, "r") as f:
        credentials = json.load(f)

    project_id = credentials.get("project_id")

    if not project_id:
        print("❌ project_id was not found in google.json")
        return

    print(f"Project ID       : {project_id}")
    print(f"Processor ID     : {PROCESSOR_ID}")
    print(f"Processor region : {PROCESSOR_LOCATION}")
    print(f"Input file       : {INPUT_FILE.name}")
    print()

    # --------------------------------------------------------
    # Create client
    # --------------------------------------------------------

    try:
        client = get_client()

        print("✅ Google credentials loaded")
        print("✅ Document AI client created")

    except Exception as e:
        print("❌ Failed to create Document AI client")
        print()
        print(type(e).__name__)
        print(e)
        return

    # --------------------------------------------------------
    # Processor name
    # --------------------------------------------------------

    processor_name = client.processor_path(
        project_id,
        PROCESSOR_LOCATION,
        PROCESSOR_ID,
    )

    print()
    print(f"Processor:")
    print(processor_name)

    # --------------------------------------------------------
    # Read document
    # --------------------------------------------------------

    try:
        document_bytes = INPUT_FILE.read_bytes()

    except Exception as e:
        print("❌ Failed to read input file")
        print(e)
        return

    # Determine MIME type
    suffix = INPUT_FILE.suffix.lower()

    mime_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }

    mime_type = mime_types.get(suffix)

    if not mime_type:
        print(f"❌ Unsupported file type: {suffix}")
        return

    print(f"MIME type        : {mime_type}")
    print(f"File size        : {len(document_bytes):,} bytes")
    print()

    # --------------------------------------------------------
    # Build request
    # --------------------------------------------------------

    raw_document = documentai.RawDocument(
        content=document_bytes,
        mime_type=mime_type,
    )

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
    )

    # --------------------------------------------------------
    # Send document to Document AI
    # --------------------------------------------------------

    print("Sending document to Google Document AI...")
    print()

    try:
        result = client.process_document(request=request)

    except Exception as e:
        print("❌ Document processing failed")
        print()
        print("Error type:")
        print(type(e).__name__)
        print()
        print("Error:")
        print(e)
        print()

        print("Possible causes:")
        print("  - Document AI API is not enabled")
        print("  - Processor ID is incorrect")
        print("  - Processor location is incorrect")
        print("  - Service account does not have permission")
        print("  - Processor does not exist")
        print("  - Unsupported document format")
        return

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    document = result.document

    print("=" * 70)
    print("✅ DOCUMENT AI PROCESSING SUCCESSFUL")
    print("=" * 70)

    print()
    print(f"Pages detected: {len(document.pages)}")

    print()
    print("-" * 70)
    print("EXTRACTED TEXT")
    print("-" * 70)

    if document.text:
        print(document.text)
    else:
        print("⚠️ Document AI returned no text.")

    print()
    print("-" * 70)
    print("PAGE INFORMATION")
    print("-" * 70)

    for index, page in enumerate(document.pages, start=1):
        print()
        print(f"Page {index}")

        if page.detected_languages:
            languages = []

            for language in page.detected_languages:
                languages.append(
                    f"{language.language_code} "
                    f"({language.confidence:.2%})"
                )

            print("Languages:", ", ".join(languages))

        if page.dimension:
            print(
                f"Dimensions: "
                f"{page.dimension.width:.0f} x "
                f"{page.dimension.height:.0f}"
            )

        print(f"Blocks     : {len(page.blocks)}")
        print(f"Paragraphs : {len(page.paragraphs)}")
        print(f"Lines      : {len(page.lines)}")
        print(f"Tokens     : {len(page.tokens)}")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    process_document()