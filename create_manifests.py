
from pathlib import Path
import pandas as pd

DATASET = Path("data/sample/hybrid_master.csv")
OUTPUT = Path("data/processed")
RANDOM_STATE = 42

OUTPUT.mkdir(parents=True, exist_ok=True)

print("Reading dataset metadata...")
df = pd.read_csv(DATASET)

df = df[
    (df["source"] == "CheXpert")
    & df["processed_image"].str.contains("frontal", case=False, na=False)
    & df["split"].isin(["train", "val"])
].copy()

df = df.drop_duplicates("processed_image")

selected = {}

for split, sample_size in [("train", 500), ("val", 100)]:
    subset = df[df["split"] == split]
    counts = subset["Cardiomegaly"].value_counts()

    per_class = sample_size // 2

    if counts.get(0, 0) < per_class or counts.get(1, 0) < per_class:
        raise ValueError(f"Insufficient examples in {split} split.")

    negatives = subset[subset["Cardiomegaly"] == 0].sample(
        n=per_class, random_state=RANDOM_STATE
    )
    positives = subset[subset["Cardiomegaly"] == 1].sample(
        n=sample_size - per_class, random_state=RANDOM_STATE
    )

    manifest = pd.concat([negatives, positives])
    manifest = manifest.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    manifest.to_csv(OUTPUT / f"{split}_manifest.csv", index=False)
    selected[split] = manifest

    print(f"{split}: {len(manifest)} rows")
    print("Label counts:", manifest["Cardiomegaly"].value_counts().sort_index().to_dict())
    print("Unique patients:", manifest["patient_id"].nunique())

overlap = set(selected["train"]["patient_id"]) & set(selected["val"]["patient_id"])
print("Patient overlap between selected splits:", len(overlap))
print("Manifests saved in:", OUTPUT.resolve())

if overlap:
    raise ValueError("Patient overlap detected; do not train until resolved.")

print("Manifest creation and patient separation checks passed.")
