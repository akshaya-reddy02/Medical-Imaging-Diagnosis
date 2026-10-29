
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    EnsureTyped,
    DivisiblePadd,
)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "segmentation"
    / "Task04_Hippocampus"
    / "Task04_Hippocampus"
)
MODEL_PATH = PROJECT_DIR / "models" / "hippocampus_3d_unet.pth"
REPORT_PATH = PROJECT_DIR / "reports" / "hippocampus_segmentation_metrics.json"

SEED = 42


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    image_dir = DATA_DIR / "imagesTr"
    label_dir = DATA_DIR / "labelsTr"

    images = {p.name: p for p in image_dir.glob("*.nii.gz")}
    labels = {p.name: p for p in label_dir.glob("*.nii.gz")}
    names = sorted(set(images) & set(labels))

    if len(names) < 15:
        raise RuntimeError("Not enough matching image-label pairs.")

    random.shuffle(names)
    val_names = names[12:14]

    items = [
        {"image": str(images[name]), "label": str(labels[name])}
        for name in val_names
    ]

    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(
            keys=["image", "label"],
            channel_dim="no_channel",
        ),
        ScaleIntensityd(keys=["image"]),
        DivisiblePadd(keys=["image", "label"], k=4, mode="constant"),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.int64),
    ])

    loader = DataLoader(
        Dataset(data=items, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["model_config"]
    model = UNet(
        spatial_dims=config["spatial_dims"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        channels=tuple(config["channels"]),
        strides=tuple(config["strides"]),
        num_res_units=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dice_metric = DiceMetric(
        include_background=False,
        reduction="none",
    )

    rows = []

    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            image = batch["image"]
            label = batch["label"].long()

            output = model(image)
            prediction = output.argmax(dim=1, keepdim=True)

            actual_classes = sorted(
                int(x) for x in torch.unique(label).tolist()
            )
            predicted_classes = sorted(
                int(x) for x in torch.unique(prediction).tolist()
            )

            actual_counts = {
                str(cls): int((label == cls).sum().item())
                for cls in range(3)
            }
            predicted_counts = {
                str(cls): int((prediction == cls).sum().item())
                for cls in range(3)
            }

            # Calculate per-class Dice directly from class masks.
            per_class_dice = {}
            for cls, class_name in [(1, "anterior"), (2, "posterior")]:
                pred_mask = prediction == cls
                true_mask = label == cls

                intersection = int((pred_mask & true_mask).sum().item())
                denominator = int(pred_mask.sum().item() + true_mask.sum().item())

                per_class_dice[class_name] = (
                    round(2.0 * intersection / denominator, 6)
                    if denominator > 0
                    else None
                )

            # Compare direct Dice with MONAI's metric on one-hot masks.
            pred_onehot = F.one_hot(
                prediction.squeeze(1), num_classes=3
            ).movedim(-1, 1).float()

            label_onehot = F.one_hot(
                label.squeeze(1), num_classes=3
            ).movedim(-1, 1).float()

            dice_metric.reset()
            dice_metric(y_pred=pred_onehot, y=label_onehot)
            monai_dice = dice_metric.aggregate().cpu().numpy().tolist()

            rows.append({
                "validation_index": index,
                "image_file": Path(items[index - 1]["image"]).name,
                "label_file": Path(items[index - 1]["label"]).name,
                "image_shape": list(image.shape),
                "label_shape": list(label.shape),
                "actual_classes_present": actual_classes,
                "predicted_classes_present": predicted_classes,
                "actual_voxel_counts": actual_counts,
                "predicted_voxel_counts": predicted_counts,
                "direct_dice": per_class_dice,
                "monai_dice": monai_dice,
            })

    report = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "checkpoint_validation_dice": checkpoint.get("validation_dice"),
        "diagnostic_results": rows,
        "note": (
            "Diagnostic only. These two volumes are the same validation "
            "subset used during the original training split."
        ),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("SEGMENTATION DIAGNOSTICS")
    print("=" * 60)
    print("Checkpoint epoch:", report["checkpoint_epoch"])
    print()

    for row in rows:
        print("Validation volume:", row["validation_index"])
        print("Image:", row["image_file"])
        print("Actual classes:", row["actual_classes_present"])
        print("Predicted classes:", row["predicted_classes_present"])
        print("Actual voxel counts:", row["actual_voxel_counts"])
        print("Predicted voxel counts:", row["predicted_voxel_counts"])
        print("Direct Dice:", row["direct_dice"])
        print("MONAI Dice:", row["monai_dice"])
        print()

    print("Diagnostic report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
