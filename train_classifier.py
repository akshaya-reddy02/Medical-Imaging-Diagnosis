
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# -------------------- Configuration --------------------

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "sample" / "images"
TRAIN_CSV = ROOT / "data" / "processed" / "train_manifest.csv"
VAL_CSV = ROOT / "data" / "processed" / "val_manifest.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Cardiomegaly"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 3e-4
SEED = 42
THRESHOLD = 0.5

DEVICE = torch.device("cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        required = {"processed_image", TARGET, "patient_id"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(
                f"{csv_path.name} is missing columns: {sorted(missing)}"
            )

        self.data = self.data.dropna(
            subset=["processed_image", TARGET, "patient_id"]
        ).reset_index(drop=True)

        self.data[TARGET] = pd.to_numeric(
            self.data[TARGET], errors="raise"
        )

        if not self.data[TARGET].isin([0, 1]).all():
            raise ValueError(f"{TARGET} must contain only 0 or 1.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        path = IMAGE_DIR / str(row["processed_image"])

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Could not load image: {path}") from exc

        label = torch.tensor(float(row[TARGET]), dtype=torch.float32)
        return image, label


def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= THRESHOLD).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(
            average_precision_score(y_true, y_prob)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
    }


def evaluate(model, loader, criterion):
    model.eval()
    labels_all = []
    probabilities_all = []
    loss_total = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels_for_loss = labels.to(DEVICE).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels_for_loss)

            probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            loss_total += loss.item() * len(labels)
            total += len(labels)
            labels_all.extend(labels.numpy().tolist())
            probabilities_all.extend(probabilities.tolist())

    y_true = np.asarray(labels_all, dtype=int)
    y_prob = np.asarray(probabilities_all)

    return (
        loss_total / max(total, 1),
        y_true,
        y_prob,
        calculate_metrics(y_true, y_prob),
    )


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.is_file() or not VAL_CSV.is_file():
        raise FileNotFoundError("Training or validation manifest is missing.")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Mild augmentation for training images only.
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_data = ChestXrayDataset(TRAIN_CSV, train_transform)
    val_data = ChestXrayDataset(VAL_CSV, val_transform)

    train_patients = set(train_data.data["patient_id"].astype(str))
    val_patients = set(val_data.data["patient_id"].astype(str))
    overlap = train_patients & val_patients

    if overlap:
        raise ValueError(
            f"Patient leakage detected: {len(overlap)} overlapping IDs."
        )

    for name, dataset in [
        ("training", train_data),
        ("validation", val_data),
    ]:
        if dataset.data[TARGET].nunique() != 2:
            raise ValueError(f"{name} data must contain both classes.")

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"Device: {DEVICE}")
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Patient overlap: {len(overlap)}")
    print(f"Epochs: {EPOCHS}")
    print("Loading pretrained ResNet18...")

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze all pretrained feature-extraction parameters.
    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(), lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 3

    model_path = MODEL_DIR / "cardiomegaly_resnet18.pth"
    report_path = REPORT_DIR / "cardiomegaly_validation_metrics.json"

    for epoch in range(EPOCHS):
        # Keep the frozen backbone, including BatchNorm layers, in eval mode.
        model.eval()
        model.fc.train()

        running_loss = 0.0
        seen = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            seen += len(labels)

        train_loss = running_loss / max(seen, 1)
        val_loss, y_true, y_prob, metrics = evaluate(
            model, val_loader, criterion
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch + 1,
            "training_loss": float(train_loss),
            "validation_loss": float(val_loss),
            "learning_rate": float(current_lr),
            "validation_accuracy": metrics["accuracy"],
            "validation_f1": metrics["f1"],
            "validation_average_precision": metrics["average_precision"],
            "validation_roc_auc": metrics["roc_auc"],
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | "
            f"validation loss={val_loss:.4f} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"AP={metrics['average_precision']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": best_state,
                "target": TARGET,
                "image_size": IMAGE_SIZE,
                "architecture": "resnet18",
                "pretrained_weights": "ResNet18_Weights.DEFAULT",
                "decision_threshold": THRESHOLD,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
            }, model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping: validation loss did not improve.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)
    _, y_true, y_prob, final_metrics = evaluate(
        model, val_loader, criterion
    )

    report = {
        "target": TARGET,
        "model": "Pretrained ResNet18; frozen backbone; trained classifier head",
        "experiment_type": "small educational pipeline experiment",
        "device": str(DEVICE),
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history),
        "best_epoch_by_validation_loss": best_epoch,
        "best_validation_loss": float(best_val_loss),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "train_images": len(train_data),
        "validation_images": len(val_data),
        "patient_overlap": len(overlap),
        "decision_threshold": THRESHOLD,
        **final_metrics,
        "training_history": history,
        "seed": SEED,
        "clinical_use": "Not intended for clinical use or diagnosis.",
        "limitations": [
            "Only 500 training and 100 validation images were used.",
            "Validation metrics are uncertain because the sample is small.",
            "The selected validation checkpoint must not be treated as a clinical model.",
            "The held-out test set must remain untouched until final evaluation.",
        ],
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n========== BEST CHECKPOINT VALIDATION RESULTS ==========")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    for name in (
        "accuracy", "precision", "recall", "f1",
        "average_precision", "roc_auc"
    ):
        print(f"{name}: {final_metrics[name]:.4f}")

    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {report_path}")
    print("\nEducational experiment only — not for clinical diagnosis.")


if __name__ == "__main__":
    main()
