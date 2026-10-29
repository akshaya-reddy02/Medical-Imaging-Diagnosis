
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

PROJECT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT / "data" / "sample" / "images"
DOWNLOAD_DIR = PROJECT / "data" / "sample"

DATASET = "tasnimakters/nih-chexpert-hybrid-chest-x-ray-dataset"

FILES = [
    "CX_patient40289_study3_view1_frontal.jpg",
    "CX_patient40337_study9_view1_frontal.jpg",
    "CX_patient40423_study1_view1_frontal.jpg",
    "CX_patient40457_study2_view1_frontal.jpg",
]


def is_valid(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        destination = IMAGE_DIR / filename

        if destination.exists() and is_valid(destination):
            print(f"Already valid: {filename}")
            continue

        success = False

        for attempt in range(1, 4):
            print(f"Downloading {filename} (attempt {attempt}/3)...")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    DATASET,
                    "-f",
                    f"images/{filename}",
                    "-p",
                    str(DOWNLOAD_DIR),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode == 0 and destination.exists():
                if is_valid(destination):
                    print(f"PASS: {filename}")
                    success = True
                    break

            print((result.stderr or result.stdout or "Download failed")[-500:])
            time.sleep(attempt * 2)

        if not success:
            print(f"FAILED: {filename}")

    print("\nRetry script finished. Verify all selected images before training.")


if __name__ == "__main__":
    main()
