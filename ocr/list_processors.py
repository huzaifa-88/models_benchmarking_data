import os
import json
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai


GOOGLE_CREDENTIALS = Path(__file__).parent / "google.json"

LOCATION = "us"


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    GOOGLE_CREDENTIALS
)


with open(GOOGLE_CREDENTIALS) as f:
    credentials = json.load(f)

project_id = credentials["project_id"]

print(f"Project: {project_id}")
print(f"Location: {LOCATION}")
print()


client_options = ClientOptions(
    api_endpoint=f"{LOCATION}-documentai.googleapis.com"
)

client = documentai.DocumentProcessorServiceClient(
    client_options=client_options
)

parent = f"projects/{project_id}/locations/{LOCATION}"

print(f"Listing processors:")
print(parent)
print()

try:
    processors = client.list_processors(parent=parent)

    found = False

    for processor in processors:
        found = True

        print("=" * 70)
        print(f"Name       : {processor.name}")
        print(f"Display    : {processor.display_name}")
        print(f"Type       : {processor.type_}")
        print(f"State      : {processor.state.name}")
        print("=" * 70)

    if not found:
        print("No processors found in this location.")

except Exception as e:
    print("❌ Could not list processors")
    print()
    print(type(e).__name__)
    print(e)