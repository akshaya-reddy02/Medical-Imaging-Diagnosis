
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -------------------- Configuration --------------------

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "sample" / "images"
VAL_CSV = ROOT / "data" / "processed" / "val_manifest.csv"
CHECKPOINT = ROOT / "models" / "cardiomegaly_resnet18.pth"
REPORT_DIR = ROOT / "reports"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224
THRESHOLD = 0.5


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not VAL_CSV.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {VAL_CSV}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")

    data = pd.read_csv(VAL_CSV)

    required = {"processed_image", "Cardiomegaly", "patient_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Validation manifest is missing columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Validation manifest contains no images.")

    # Select the first validation image deterministically.
    row = data.iloc[0]
    image_name = str(row["processed_image"])
    image_path = IMAGE_DIR / image_name
    true_label = int(row["Cardiomegaly"])

    if not image_path.is_file():
        raise FileNotFoundError(f"Validation image not found: {image_path}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("The saved checkpoint is not a ResNet18 model.")

    if checkpoint.get("target") != "Cardiomegaly":
        raise ValueError("The checkpoint target is not Cardiomegaly.")

    # Re-create the same architecture and load the saved weights.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["normalization_mean"],
            std=checkpoint["normalization_std"],
        ),
    ])

    with Image.open(image_path) as image:
        original = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        original_array = np.asarray(original, dtype=np.float32) / 255.0
        input_tensor = transform(original).unsqueeze(0).to(DEVICE)

    # ResNet18's final convolutional block is used for Grad-CAM.
    target_layers = [model.layer4[-1]]

    # The model has one output logit, so output index 0 is the target.
    # Grad-CAM needs gradients with respect to the input activations.
    with GradCAM(
        model=model,
        target_layers=target_layers,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(0)],
        )[0]

    overlay = show_cam_on_image(
        original_array,
        grayscale_cam,
        use_rgb=True,
    )

    # Obtain the model's score without updating model parameters.
    with torch.no_grad():
        logit = model(input_tensor).squeeze().item()
        probability = float(torch.sigmoid(torch.tensor(logit)).item())

    threshold = float(checkpoint.get("decision_threshold", THRESHOLD))
    predicted_label = int(probability >= threshold)

    original_path = REPORT_DIR / "cardiomegaly_gradcam_original.png"
    overlay_path = REPORT_DIR / "cardiomegaly_gradcam_overlay.png"
    metadata_path = REPORT_DIR / "cardiomegaly_gradcam_metadata.json"

    plt.imsave(original_path, original_array)
    plt.imsave(overlay_path, overlay)

    metadata = {
        "target": "Cardiomegaly",
        "image": image_name,
        "patient_id": str(row["patient_id"]),
        "split": "validation",
        "true_label": true_label,
        "predicted_label_at_threshold": predicted_label,
        "predicted_probability": probability,
        "decision_threshold": threshold,
        "gradcam_target_layer": "layer4[-1]",
        "clinical_use": "Not intended for clinical use or diagnosis.",
        "interpretation_warning": (
            "Grad-CAM highlights image regions influencing this model's "
            "output. It does not prove disease presence or clinical validity."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n========== GRAD-CAM COMPLETE ==========")
    print(f"Image: {image_name}")
    print(f"Actual label: {true_label}")
    print(f"Predicted probability: {probability:.4f}")
    print(f"Predicted label at threshold: {predicted_label}")
    print(f"Original image saved: {original_path}")
    print(f"Grad-CAM overlay saved: {overlay_path}")
    print(f"Metadata saved: {metadata_path}")
    print("\nEducational experiment only — not for clinical diagnosis.")


if __name__ == "__main__":
    main()
