
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "data" / "sample" / "images"
MANIFEST_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"

DATASET = "tasnimakters/nih-chexpert-hybrid-chest-x-ray-dataset"
MANIFEST_FILES = [
    MANIFEST_DIR / "train_manifest.csv",
    MANIFEST_DIR / "val_manifest.csv",
]

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def valid_image(path):
    """Return True only if the image can be opened and verified."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_image(filename):
    """Download one selected image into the expected images folder."""
    if Path(filename).name != filename or not filename:
        return "FAILED: unexpected filename"

    destination = IMAGE_DIR / filename

    if destination.exists() and valid_image(destination):
        return "EXISTING"

    for attempt in range(1, 4):
        # Kaggle creates a nested images/ folder relative to this parent.
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
                str(IMAGE_DIR.parent),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if destination.exists() and valid_image(destination):
            return "DOWNLOADED"

        # Some CLI versions may create images/images; recover the file
        # only when the expected destination is missing.
        nested = IMAGE_DIR / "images" / filename
        if nested.exists() and valid_image(nested):
            destination.parent.mkdir(parents=True, exist_ok=True)
            nested.replace(destination)
            return "DOWNLOADED"

        message = (result.stderr or result.stdout or "").strip()

        if attempt < 3:
            print(f"Retry {attempt}/3 for {filename}: {message[-200:]}")
            time.sleep(attempt * 2)
        else:
            return (
                f"FAILED: exit_code={result.returncode}; "
                f"{message[-500:] or 'image missing or invalid after download'}"
            )

    return "FAILED: unknown error"


def main():
    selected = []

    for manifest_path in MANIFEST_FILES:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        manifest = pd.read_csv(manifest_path)

        required_columns = {
            "processed_image",
            "split",
            "patient_id",
            "Cardiomegaly",
        }
        missing = required_columns - set(manifest.columns)

        if missing:
            raise ValueError(
                f"{manifest_path.name} is missing columns: {sorted(missing)}"
            )

        expected_split = (
            "train" if "train_manifest" in manifest_path.name else "val"
        )
        if not manifest["split"].eq(expected_split).all():
            raise ValueError(
                f"Unexpected split values in {manifest_path.name}"
            )

        selected.append(manifest)

    # Verify patient separation before downloading.
    train_patients = set(selected[0]["patient_id"].dropna())
    val_patients = set(selected[1]["patient_id"].dropna())
    overlap = train_patients & val_patients

    if overlap:
        raise ValueError(
            f"Patient overlap detected: {len(overlap)} patients"
        )

    filenames = sorted(
        set(
            str(name)
            for manifest in selected
            for name in manifest["processed_image"].dropna()
        )
    )

    print(f"Unique selected images: {len(filenames)}")
    print(f"Image destination: {IMAGE_DIR}")
    print("Patient separation check: PASS")
    print("Starting download and image verification...")

    failures = []
    downloaded = 0
    existing = 0

    for index, filename in enumerate(filenames, start=1):
        try:
            status = download_image(filename)
        except subprocess.TimeoutExpired:
            status = "FAILED: download timed out"
        except Exception as exc:
            status = f"FAILED: {type(exc).__name__}: {exc}"

        if status == "DOWNLOADED":
            downloaded += 1
        elif status == "EXISTING":
            existing += 1
        else:
            failures.append(
                {"filename": filename, "error": status}
            )

        if index % 10 == 0 or index == len(filenames):
            print(
                f"Progress {index}/{len(filenames)} | "
                f"downloaded={downloaded} | "
                f"existing={existing} | "
                f"failed={len(failures)}"
            )

    failure_report = REPORT_DIR / "image_download_failures.csv"
    pd.DataFrame(
        failures, columns=["filename", "error"]
    ).to_csv(failure_report, index=False)

    print("\n========== DOWNLOAD SUMMARY ==========")
    print("Selected:", len(filenames))
    print("Downloaded:", downloaded)
    print("Already valid:", existing)
    print("Failed:", len(failures))
    print("Failure report:", failure_report)

    if not failures:
        print("Image integrity checks: PASS")
        print("All selected images are available.")
    else:
        print("Some images failed. Review the failure report before training.")


if __name__ == "__main__":
    main()
