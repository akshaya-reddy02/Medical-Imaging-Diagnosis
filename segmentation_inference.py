
from pathlib import Path
import argparse
import json

import nibabel as nib
import numpy as np
import torch

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    DivisiblePad,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "hippocampus_3d_unet.pth"

CLASS_NAMES = {
    0: "background",
    1: "anterior hippocampus",
    2: "posterior hippocampus",
}


class HippocampusSegmenter:
    """Load the experimental 3D U-Net and segment a 3D MRI volume."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Segmentation checkpoint not found: {self.model_path}"
            )

        self.device = torch.device("cpu")
        torch.set_num_threads(2)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        config = checkpoint["model_config"]

        self.model = UNet(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=tuple(config["channels"]),
            strides=tuple(config["strides"]),
            num_res_units=1,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch")
        self.validation_dice = checkpoint.get("validation_dice")

        self.load_image = LoadImage(image_only=True)
        self.prepare_image = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensity(),
        ])
        self.pad_image = DivisiblePad(k=4, mode="constant")

    def predict(self, image_path):
        """
        Segment a 3D NIfTI MRI volume.

        Returns a mask matching the original MRI dimensions and a summary.
        """
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(f"MRI image not found: {image_path}")

        # Load the original volume and preserve its spatial dimensions.
        original_image = self.load_image(str(image_path))
        image = self.prepare_image(original_image)

        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                "Expected a single-channel 3D MRI volume. "
                f"Received shape {tuple(image.shape)}."
            )

        original_shape = tuple(int(dim) for dim in image.shape[1:])

        # Pad only for model compatibility.
        padded_image = self.pad_image(image)

        tensor = torch.as_tensor(
            np.ascontiguousarray(padded_image),
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            predicted_mask = torch.argmax(logits, dim=1)[0]

        # Remove padding and restore the original spatial dimensions.
        predicted_mask = predicted_mask[
            :original_shape[0],
            :original_shape[1],
            :original_shape[2],
        ]

        mask = predicted_mask.cpu().numpy().astype(np.uint8)

        if mask.shape != original_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"original MRI shape {original_shape}."
            )

        counts = {
            CLASS_NAMES[class_id]: int(np.count_nonzero(mask == class_id))
            for class_id in CLASS_NAMES
        }

        summary = {
            "image_file": image_path.name,
            "original_shape": list(original_shape),
            "predicted_voxel_counts": counts,
            "predicted_classes": sorted(
                int(value) for value in np.unique(mask)
            ),
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_validation_dice": self.validation_dice,
            "device": str(self.device),
            "clinical_use": False,
            "notice": (
                "Experimental segmentation output only. "
                "Not validated for clinical diagnosis or treatment."
            ),
        }

        return mask, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the experimental hippocampus 3D U-Net."
    )
    parser.add_argument(
        "image",
        help="Path to a 3D NIfTI MRI volume (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--output",
        default="reports/hippocampus_prediction.nii.gz",
        help="Output path for the segmentation mask",
    )
    args = parser.parse_args()

    segmenter = HippocampusSegmenter()
    mask, summary = segmenter.predict(args.image)

    source = nib.load(args.image)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_image = nib.Nifti1Image(
        mask,
        source.affine,
        source.header.copy(),
    )
    mask_image.set_data_dtype(np.uint8)
    nib.save(mask_image, str(output_path))

    summary["output_mask"] = str(output_path)
    print(json.dumps(summary, indent=2))
