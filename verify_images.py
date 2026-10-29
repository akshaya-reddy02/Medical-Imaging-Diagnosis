
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "sample" / "images"
MANIFESTS = [
    ROOT / "data" / "processed" / "train_manifest.csv",
    ROOT / "data" / "processed" / "val_manifest.csv",
]
REPORT = ROOT / "reports" / "image_validation_report.csv"

records = []

for manifest_path in MANIFESTS:
    df = pd.read_csv(manifest_path)

    for filename in df["processed_image"].dropna().unique():
        path = IMAGE_DIR / filename
        status = "PASS"
        details = ""

        try:
            if not path.is_file():
                raise FileNotFoundError("Image file missing")

            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                if image.width < 1 or image.height < 1:
                    raise ValueError("Invalid dimensions")

        except Exception as exc:
            status = "FAIL"
            details = f"{type(exc).__name__}: {exc}"

        records.append({
            "filename": filename,
            "status": status,
            "details": details,
        })

report = pd.DataFrame(records)
REPORT.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(REPORT, index=False)

print("\n========== IMAGE VALIDATION ==========")
print(f"Unique images checked: {len(report)}")
print(f"Passed: {(report['status'] == 'PASS').sum()}")
print(f"Failed: {(report['status'] == 'FAIL').sum()}")
print(f"Report saved: {REPORT}")

if (report["status"] == "FAIL").any():
    print("\nFailed images:")
    print(report.loc[report["status"] == "FAIL"].to_string(index=False))
else:
    print("\nAll manifest images passed integrity validation.")
