
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    EnsureTyped,
    DivisiblePadd,
)
from monai.networks.nets import UNet


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "segmentation"
    / "Task04_Hippocampus"
    / "Task04_Hippocampus"
)
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "hippocampus_3d_unet.pth"
REPORT_PATH = REPORT_DIR / "hippocampus_segmentation_metrics.json"

SEED = 42
TRAIN_VOLUMES = 12
VALIDATION_VOLUMES = 2
EPOCHS = 12
LEARNING_RATE = 0.0005

# Give the smaller foreground classes more influence in cross-entropy.
CLASS_WEIGHTS = torch.tensor([0.15, 1.0, 1.0], dtype=torch.float32)


def calculate_dice(logits, labels):
    predictions = logits.argmax(dim=1)
    scores = []

    for class_id in (1, 2):
        pred_mask = predictions == class_id
        true_mask = labels[:, 0] == class_id

        intersection = (pred_mask & true_mask).sum().float()
        denominator = pred_mask.sum().float() + true_mask.sum().float()

        if denominator.item() == 0:
            scores.append(float("nan"))
        else:
            scores.append((2.0 * intersection / denominator).item())

    valid_scores = [score for score in scores if np.isfinite(score)]
    return float(np.mean(valid_scores)) if valid_scores else 0.0, scores


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(2)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    image_dir = DATA_DIR / "imagesTr"
    label_dir = DATA_DIR / "labelsTr"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Dataset directories not found: {DATA_DIR}")

    image_files = {p.name: p for p in image_dir.glob("*.nii.gz")}
    label_files = {p.name: p for p in label_dir.glob("*.nii.gz")}
    names = sorted(set(image_files) & set(label_files))

    if len(names) < TRAIN_VOLUMES + VALIDATION_VOLUMES:
        raise RuntimeError("Insufficient matching image-label pairs.")

    # Keep the same deterministic train/validation split as the
    # original experiment so that results remain comparable.
    random.shuffle(names)
    train_names = names[:TRAIN_VOLUMES]
    val_names = names[
        TRAIN_VOLUMES:TRAIN_VOLUMES + VALIDATION_VOLUMES
    ]

    def make_items(selected_names):
        return [
            {
                "image": str(image_files[name]),
                "label": str(label_files[name]),
            }
            for name in selected_names
        ]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(
            keys=["image", "label"],
            channel_dim="no_channel",
        ),
        ScaleIntensityd(keys=["image"]),
        DivisiblePadd(
            keys=["image", "label"],
            k=4,
            mode="constant",
        ),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.int64),
    ])

    train_loader = DataLoader(
        Dataset(make_items(train_names), transform=transforms),
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Dataset(make_items(val_names), transform=transforms),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(8, 16),
        strides=(2,),
        num_res_units=1,
    )

    class_weights = CLASS_WEIGHTS.clone()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []
    best_val_loss = float("inf")
    best_val_dice = -1.0
    start_time = time.time()

    print("=" * 60)
    print("Hippocampus 3D U-Net — weighted-loss experiment")
    print("=" * 60)
    print(f"Device: CPU")
    print(f"Training volumes: {len(train_names)}")
    print(f"Validation volumes: {len(val_names)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Class weights: {class_weights.tolist()}")
    print("This small experiment is not clinical validation.")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0

        for batch in train_loader:
            images = batch["image"]
            labels = batch["label"].long()

            if labels.min().item() < 0 or labels.max().item() > 2:
                raise ValueError("Unexpected segmentation label value.")

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)

            ce_loss = F.cross_entropy(
                logits,
                labels[:, 0],
                weight=class_weights,
            )

            probabilities = torch.softmax(logits, dim=1)
            one_hot_labels = F.one_hot(
                labels[:, 0],
                num_classes=3,
            ).movedim(-1, 1).float()

            # Foreground Dice loss; background is excluded.
            dice_losses = []
            for class_id in (1, 2):
                pred_class = probabilities[:, class_id]
                true_class = one_hot_labels[:, class_id]

                intersection = (pred_class * true_class).sum()
                denominator = pred_class.sum() + true_class.sum()

                dice = (2.0 * intersection + 1e-5) / (
                    denominator + 1e-5
                )
                dice_losses.append(1.0 - dice)

            dice_loss = torch.stack(dice_losses).mean()
            loss = 0.5 * ce_loss + 0.5 * dice_loss

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")

            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        average_train_loss = train_loss_total / len(train_loader)

        model.eval()
        val_loss_total = 0.0
        val_dice_scores = []
        val_class_dice = [[], []]

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"]
                labels = batch["label"].long()
                logits = model(images)

                ce_loss = F.cross_entropy(
                    logits,
                    labels[:, 0],
                    weight=class_weights,
                )

                probabilities = torch.softmax(logits, dim=1)
                one_hot_labels = F.one_hot(
                    labels[:, 0],
                    num_classes=3,
                ).movedim(-1, 1).float()

                dice_losses = []
                for class_id in (1, 2):
                    pred_class = probabilities[:, class_id]
                    true_class = one_hot_labels[:, class_id]
                    intersection = (pred_class * true_class).sum()
                    denominator = pred_class.sum() + true_class.sum()
                    dice = (2.0 * intersection + 1e-5) / (
                        denominator + 1e-5
                    )
                    dice_losses.append(1.0 - dice)

                dice_loss = torch.stack(dice_losses).mean()
                val_loss_total += (0.5 * ce_loss + 0.5 * dice_loss).item()

                mean_dice, class_dice = calculate_dice(logits, labels)
                val_dice_scores.append(mean_dice)

                for index, score in enumerate(class_dice):
                    if np.isfinite(score):
                        val_class_dice[index].append(score)

        average_val_loss = val_loss_total / len(val_loader)
        average_val_dice = float(np.mean(val_dice_scores))
        anterior_dice = (
            float(np.mean(val_class_dice[0]))
            if val_class_dice[0] else 0.0
        )
        posterior_dice = (
            float(np.mean(val_class_dice[1]))
            if val_class_dice[1] else 0.0
        )

        record = {
            "epoch": epoch,
            "train_loss": round(average_train_loss, 6),
            "validation_loss": round(average_val_loss, 6),
            "validation_dice_foreground": round(average_val_dice, 6),
            "validation_dice_anterior": round(anterior_dice, 6),
            "validation_dice_posterior": round(posterior_dice, 6),
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {average_train_loss:.4f} | "
            f"Val loss: {average_val_loss:.4f} | "
            f"Foreground Dice: {average_val_dice:.4f} | "
            f"Anterior: {anterior_dice:.4f} | "
            f"Posterior: {posterior_dice:.4f}"
        )

        # Keep the checkpoint with the best foreground Dice,
        # rather than selecting it by loss alone.
        if average_val_dice > best_val_dice:
            best_val_dice = average_val_dice
            best_val_loss = average_val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "spatial_dims": 3,
                        "in_channels": 1,
                        "out_channels": 3,
                        "channels": [8, 16],
                        "strides": [2],
                    },
                    "seed": SEED,
                    "epoch": epoch,
                    "validation_loss": average_val_loss,
                    "validation_dice": average_val_dice,
                    "class_names": [
                        "background",
                        "anterior",
                        "posterior",
                    ],
                },
                MODEL_PATH,
            )

    report = {
        "experiment": "Hippocampus 3D U-Net weighted-loss experiment",
        "device": "cpu",
        "dataset": "Medical Segmentation Decathlon Task04 Hippocampus",
        "training_volumes": len(train_names),
        "validation_volumes": len(val_names),
        "epochs_requested": EPOCHS,
        "seed": SEED,
        "class_weights": class_weights.tolist(),
        "best_checkpoint": str(MODEL_PATH.relative_to(PROJECT_DIR)),
        "best_validation_loss": round(best_val_loss, 6),
        "best_validation_dice_foreground": round(best_val_dice, 6),
        "history": history,
        "limitations": [
            "Only 12 training and 2 validation volumes were used.",
            "Class weighting and Dice loss do not guarantee improved performance.",
            "The held-out MSD test set was not used.",
            "Results are experimental and not clinical evidence.",
        ],
        "runtime_seconds": round(time.time() - start_time, 2),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Training completed.")
    print(f"Best checkpoint: {MODEL_PATH}")
    print(f"Metrics report: {REPORT_PATH}")
    print(f"Best foreground Dice: {best_val_dice:.4f}")
    print(f"Runtime: {report['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
