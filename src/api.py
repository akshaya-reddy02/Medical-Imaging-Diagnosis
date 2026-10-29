
from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
import logging
import os
import tempfile

import httpx
import nibabel as nib
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from torch import nn
from torchvision import models, transforms

from segmentation_inference import HippocampusSegmenter
from src.audit_log import router as audit_router


BASE_DIR = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = BASE_DIR / "models" / "cardiomegaly_resnet18.pth"

ORTHANC_URL = os.getenv(
    "ORTHANC_URL", "http://127.0.0.1:8042"
).rstrip("/")

ORTHANC_TIMEOUT = 10.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_MRI_BYTES = 100 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.exception("Could not remove temporary output file.")


def load_classifier():
    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold = float(checkpoint.get("decision_threshold", 0.5))

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, threshold, image_transform


async def orthanc_get(path: str):
    """Retrieve JSON data from the local Orthanc REST API."""
    try:
        async with httpx.AsyncClient(
            base_url=ORTHANC_URL,
            timeout=ORTHANC_TIMEOUT,
            trust_env=False,
        ) as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException as exc:
        logger.warning("Orthanc request timed out.")
        raise HTTPException(
            status_code=503,
            detail="Orthanc request timed out.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Orthanc returned HTTP %s.",
            exc.response.status_code,
        )
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="Orthanc resource not found.",
            ) from exc

        raise HTTPException(
            status_code=503,
            detail="Orthanc could not complete the request.",
        ) from exc

    except (httpx.RequestError, ValueError) as exc:
        logger.warning(
            "Orthanc connection or JSON error: %s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Orthanc is unavailable or returned invalid JSON.",
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = None
    app.state.classifier_threshold = 0.5
    app.state.image_transform = None
    app.state.segmenter = None
    app.state.load_errors = {}

    try:
        (
            app.state.classifier,
            app.state.classifier_threshold,
            app.state.image_transform,
        ) = load_classifier()
    except Exception as exc:
        logger.exception("Classifier loading failed.")
        app.state.load_errors["classification"] = type(exc).__name__

    try:
        app.state.segmenter = HippocampusSegmenter()
    except Exception as exc:
        logger.exception("Segmentation model loading failed.")
        app.state.load_errors["segmentation"] = type(exc).__name__

    yield


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Medical Imaging Diagnosis API",
    description=(
        "Educational prototype for experimental chest X-ray "
        "classification, hippocampus MRI segmentation, and "
        "Orthanc DICOM study browsing. "
        "Not for clinical diagnosis or treatment."
    ),
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(audit_router)



@app.get("/")
def root():
    return {
        "message": "Medical Imaging Diagnosis API",
        "documentation": "/docs",
        "clinical_use": False,
    }


@app.get("/health")
def health():
    classification_loaded = app.state.classifier is not None
    segmentation_loaded = app.state.segmenter is not None

    return {
        "status": (
            "ok"
            if classification_loaded and segmentation_loaded
            else "degraded"
        ),
        "classification_loaded": classification_loaded,
        "segmentation_loaded": segmentation_loaded,
        "orthanc_url": ORTHANC_URL,
        "load_errors": app.state.load_errors,
        "clinical_use": False,
    }


@app.get("/orthanc/instances/{instance_id}/preview")
async def get_orthanc_instance_preview(instance_id: str):
    """Return a PNG preview for a local Orthanc DICOM instance."""
    import re
    from fastapi import HTTPException
    from fastapi.responses import Response
    import httpx

    if not re.fullmatch(r"[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{8}){4}", instance_id):
        raise HTTPException(status_code=400, detail="Invalid Orthanc instance ID.")

    try:
        async with httpx.AsyncClient(base_url=ORTHANC_URL, timeout=ORTHANC_TIMEOUT) as client:
            result = await client.get(f"/instances/{instance_id}/preview")
            result.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Preview or DICOM instance not found.") from exc
        raise HTTPException(status_code=502, detail="Orthanc could not generate the preview.") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Orthanc preview request timed out.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to Orthanc.") from exc

    return Response(content=result.content, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/orthanc/studies")
async def list_orthanc_studies():
    """List studies currently stored in Orthanc."""
    studies = await orthanc_get("/studies")

    if not isinstance(studies, list):
        raise HTTPException(
            status_code=502,
            detail="Orthanc returned an unexpected studies response.",
        )

    return {
        "count": len(studies),
        "studies": studies,
        "clinical_use": False,
    }


@app.get("/orthanc/studies/{study_id}")
async def get_orthanc_study(study_id: str):
    """Retrieve metadata for a specific Orthanc study."""
    if not study_id or "/" in study_id or "\\" in study_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid Orthanc study ID.",
        )

    study = await orthanc_get(
        f"/studies/{study_id}"
    )

    if not isinstance(study, dict):
        raise HTTPException(
            status_code=502,
            detail="Orthanc returned an unexpected study response.",
        )

    return {
        "study": study,
        "clinical_use": False,
    }


@app.get("/orthanc/studies/{study_id}/series")
async def list_orthanc_study_series(study_id: str):
    """List the series belonging to a specific study."""
    if not study_id or "/" in study_id or "\\" in study_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid Orthanc study ID.",
        )

    series = await orthanc_get(
        f"/studies/{study_id}/series"
    )

    if not isinstance(series, list):
        raise HTTPException(
            status_code=502,
            detail="Orthanc returned an unexpected series response.",
        )

    return {
        "study_id": study_id,
        "count": len(series),
        "series": series,
        "clinical_use": False,
    }


@app.post("/predict/classification")
async def predict_classification(file: UploadFile = File(...)):
    if app.state.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model is unavailable.",
        )

    try:
        content = await file.read(MAX_IMAGE_BYTES + 1)
    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 20 MB limit.",
        )

    try:
        with Image.open(BytesIO(content)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file.",
        ) from exc

    tensor = app.state.image_transform(image).unsqueeze(0)

    with torch.inference_mode():
        probability = torch.sigmoid(
            app.state.classifier(tensor)
        ).item()

    threshold = app.state.classifier_threshold

    return {
        "target": "Cardiomegaly",
        "probability": round(float(probability), 6),
        "threshold": threshold,
        "predicted_label": int(probability >= threshold),
        "notice": (
            "Experimental result only. Not validated for clinical diagnosis "
            "or treatment."
        ),
        "clinical_use": False,
    }


@app.post("/predict/segmentation")
async def predict_segmentation(file: UploadFile = File(...)):
    if app.state.segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model is unavailable.",
        )

    filename = (file.filename or "").lower()

    if not (
        filename.endswith(".nii")
        or filename.endswith(".nii.gz")
    ):
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Upload a NIfTI MRI file ending in .nii or .nii.gz.",
        )

    suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
    input_path = None
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_input:
            input_path = temp_input.name
            total_bytes = 0

            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_MRI_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="MRI file exceeds the 100 MB limit.",
                    )

                temp_input.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded MRI file is empty.",
            )

        mask, summary = app.state.segmenter.predict(input_path)

        source_image = nib.load(input_path)
        mask_array = np.asarray(mask, dtype=np.uint8)

        if mask_array.shape != source_image.shape:
            raise ValueError(
                "Predicted mask dimensions do not match the input MRI."
            )

        with tempfile.NamedTemporaryFile(
            suffix=".nii.gz",
            delete=False,
        ) as temp_output:
            output_path = temp_output.name

        output_image = nib.Nifti1Image(
            mask_array,
            source_image.affine,
            source_image.header.copy(),
        )
        output_image.set_data_dtype(np.uint8)
        nib.save(output_image, output_path)

        original_name = Path(
            file.filename or "mri.nii.gz"
        ).name

        if original_name.lower().endswith(".nii.gz"):
            original_name = original_name[:-7]
        elif original_name.lower().endswith(".nii"):
            original_name = original_name[:-4]

        download_name = f"{original_name}_predicted_mask.nii.gz"

        return FileResponse(
            path=output_path,
            media_type="application/gzip",
            filename=download_name,
            headers={
                "X-Clinical-Use": "false",
                "X-Predicted-Classes": ",".join(
                    str(value)
                    for value in summary.get("predicted_classes", [])
                ),
            },
            background=BackgroundTask(
                cleanup_file,
                output_path,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("MRI segmentation inference failed.")
        raise HTTPException(
            status_code=500,
            detail=f"MRI segmentation failed: {type(exc).__name__}",
        ) from exc
    finally:
        await file.close()

        if input_path:
            cleanup_file(input_path)
