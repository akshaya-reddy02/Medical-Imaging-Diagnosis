
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "data" / "sample" / "hybrid_master.csv"
INFO_PATH = BASE / "data" / "sample" / "dataset_info.json"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, low_memory=False)

    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== FIRST 5 ROWS (selected columns) ===")
    selected = [
        c for c in [
            "image_id", "processed_image", "source",
            "patient_id", "age", "sex", "view", "split"
        ] if c in df.columns
    ]
    print(df[selected].head().to_string(index=False))

    print("\n=== SOURCE COUNTS ===")
    if "source" in df.columns:
        print(df["source"].value_counts(dropna=False).to_string())

    print("\n=== SPLIT COUNTS ===")
    if "split" in df.columns:
        print(df["split"].value_counts(dropna=False).to_string())

    print("\n=== MISSING VALUES (top 15 columns) ===")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing.head(15).to_string())

    label_columns = [
        c for c in df.columns
        if c not in {
            "image_id", "processed_image", "source", "patient_id",
            "age", "sex", "view", "follow_up_num", "has_bbox", "split"
        }
        and not c.startswith("uncertain_")
    ]

    print("\n=== LABEL VALUE COUNTS ===")
    for col in label_columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_dict())

    print("\n=== PATIENT SPLIT CHECK ===")
    if {"patient_id", "split"}.issubset(df.columns):
        patient_splits = df.groupby("patient_id")["split"].nunique()
        overlap_count = int((patient_splits > 1).sum())
        print(f"Unique patients: {df['patient_id'].nunique():,}")
        print(f"Patients appearing in multiple splits: {overlap_count:,}")

    print("\n=== DOCUMENTATION CROSS-CHECK ===")
    if INFO_PATH.exists():
        info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
        print(f"Documented total images: {info.get('total_images')}")
        print(f"Actual CSV rows: {len(df)}")
        print(f"Documented labels: {len(info.get('labels', []))}")
        print(f"Labels found in CSV: {len(label_columns)}")
        if len(df) != info.get("total_images"):
            print("WARNING: CSV row count differs from documentation.")
        if len(label_columns) != len(info.get("labels", [])):
            print("WARNING: Label column count differs from documentation.")

    print("\nInspection complete.")

if __name__ == "__main__":
    main()
