import cv2
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
import logging
import os
import tempfile

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.audit_log import router as audit_router
from src.resnet import resnet18


BASE_DIR = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = BASE_DIR / "models" / "cardiomegaly_resnet18.pth"

ORTHANC_URL = os.getenv(
    "ORTHANC_URL", "http://127.0.0.1:8042"
).rstrip("/")

ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME", "demo")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD", "DemoOrthanc2026")

ORTHANC_TIMEOUT = 10.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_MRI_BYTES = 100 * 1024 * 1024
ENABLE_SEGMENTATION = os.getenv("ENABLE_SEGMENTATION", "true").lower() == "true"
ENABLE_ORTHANC_MONITOR = os.getenv("ENABLE_ORTHANC_MONITOR", "true").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The model is tiny and requests are rare, so extra inference threads only
# add per-thread allocator arenas. Keeping them at one lowers the process
# RSS on the 512 MB Free Render instance.
torch.set_num_threads(1)
cv2.setNumThreads(1)

CLASSIFIER_MEAN = (0.485, 0.456, 0.406)
CLASSIFIER_STD = (0.229, 0.224, 0.225)


def prepare_classifier_input(image: Image.Image) -> torch.Tensor:
    """Reproduce the previous torchvision transform without importing it.

    Resize to 224x224 with bilinear sampling, convert to a float CHW tensor
    in [0, 1], then apply the ImageNet normalization. This matches the old
    ``transforms.Compose([Resize, ToTensor, Normalize])`` pipeline.
    """
    resized = image.resize((224, 224), resample=Image.BILINEAR)
    array = np.array(resized, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float()
    tensor /= 255.0

    mean = torch.tensor(CLASSIFIER_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CLASSIFIER_STD, dtype=torch.float32).view(3, 1, 1)
    tensor.sub_(mean).div_(std)

    return tensor


def cleanup_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.exception("Could not remove temporary output file.")


def blend_cam_overlay(
    original: np.ndarray,
    cam: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto a uint8 image in bounded memory.

    The native-resolution overlay is built one horizontal band at a time so a
    large radiograph does not create several full-size float32 copies at once.
    The Free Render instance only has 512 MB, and those copies were enough to
    trigger an out-of-memory restart during Grad-CAM generation.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    height, width = cam.shape
    output = np.empty((height, width, 3), dtype=np.uint8)

    band_height = 128
    for top in range(0, height, band_height):
        bottom = min(height, top + band_height)
        base = original[top:bottom].astype(np.float32)
        tint = heatmap[top:bottom].astype(np.float32)
        weight = alpha[top:bottom, :, None]
        base *= 1.0 - weight
        tint *= weight
        base += tint
        np.clip(base, 0.0, 255.0, out=base)
        output[top:bottom] = base.astype(np.uint8)

    return output


def compute_gradcam_plusplus(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> tuple[np.ndarray, float]:
    """Return a Grad-CAM++ map and the positive-class probability.

    Grad-CAM only needs the gradient of the score with respect to the target
    layer's output. Freezing the backbone and detaching that activation lets
    the classifier head build a tiny graph instead of backpropagating through
    (and retaining) the whole ResNet18 graph. On the 512 MB Free Render
    instance the full backward pass was enough to trigger an out-of-memory
    restart, so this keeps the same Grad-CAM++ math with bounded memory.
    """
    captured = {}

    def capture(_module, _inputs, output):
        activation = output.detach().requires_grad_(True)
        captured["activation"] = activation
        return activation

    trainable = [p for p in model.parameters() if p.requires_grad]
    for parameter in trainable:
        parameter.requires_grad_(False)

    handle = model.layer4[-1].register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model(tensor)
            activation = captured["activation"]
            score = logits[:, 0].sum()
            grads = torch.autograd.grad(score, activation)[0]
    finally:
        handle.remove()
        for parameter in trainable:
            parameter.requires_grad_(True)

    activations = activation.detach().cpu().numpy()
    gradients = grads.detach().cpu().numpy()

    grads_power_2 = gradients ** 2
    grads_power_3 = grads_power_2 * gradients
    spatial_axes = (2, 3)
    sum_activations = np.sum(activations, axis=spatial_axes, keepdims=True)
    eps = 1e-6
    aij = grads_power_2 / (
        2.0 * grads_power_2 + sum_activations * grads_power_3 + eps
    )
    aij = np.where(gradients != 0, aij, 0.0)
    weights = np.sum(np.maximum(gradients, 0.0) * aij, axis=spatial_axes)
    cam = np.sum(weights[:, :, None, None] * activations, axis=1)[0]
    cam = np.maximum(cam, 0.0)
    cam -= cam.min()
    cam /= max(1e-7, float(cam.max()))

    probability = torch.sigmoid(logits.detach()).item()
    return np.float32(cam), probability


def load_classifier():
    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = resnet18(num_classes=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Inference never needs parameter gradients. Grad-CAM attaches its own
    # gradient to the detached target-layer activation, so freezing here does
    # not change the saliency math.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    threshold = float(checkpoint.get("decision_threshold", 0.5))

    # Release the checkpoint tensors before serving requests.
    del checkpoint

    return model, threshold, prepare_classifier_input


async def orthanc_get(path: str):
    """Retrieve JSON data from the local Orthanc REST API."""
    import httpx

    try:
        async with httpx.AsyncClient(
            base_url=ORTHANC_URL,
            timeout=ORTHANC_TIMEOUT,
            trust_env=False,
        ) as client:
            response = await client.get(path, auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD))
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
    app.state.orthanc_monitor = None
    app.state.load_errors = {}

    # Load classification model
    try:
        (
            app.state.classifier,
            app.state.classifier_threshold,
            app.state.image_transform,
        ) = load_classifier()

    except Exception as exc:
        logger.exception("Classifier loading failed.")
        app.state.load_errors["classification"] = type(exc).__name__

    # Load segmentation model only when enabled
    if ENABLE_SEGMENTATION:
        try:
            from segmentation_inference import HippocampusSegmenter

            app.state.segmenter = HippocampusSegmenter()

        except Exception as exc:
            logger.exception("Segmentation model loading failed.")
            app.state.load_errors["segmentation"] = type(exc).__name__
    else:
        app.state.load_errors["segmentation"] = (
            "Disabled by deployment configuration"
        )

    # Start Orthanc monitor only when enabled
    monitor = None
    monitor_task = None

    if ENABLE_ORTHANC_MONITOR and app.state.classifier is not None:
        try:
            from src.orthanc_monitor import OrthancMonitor

            monitor = OrthancMonitor(
                base_url=ORTHANC_URL,
                username=ORTHANC_USERNAME,
                password=ORTHANC_PASSWORD,
                classifier=app.state.classifier,
                image_transform=app.state.image_transform,
                threshold=app.state.classifier_threshold,
            )

            app.state.orthanc_monitor = monitor
            monitor_task = asyncio.create_task(monitor.run())

        except Exception as exc:
            logger.exception("Orthanc monitor startup failed.")
            app.state.load_errors["orthanc_monitor"] = type(exc).__name__

    try:
        yield

    finally:
        if monitor is not None:
            monitor.stop()

        if monitor_task is not None:
            monitor_task.cancel()

            try:
                await monitor_task
            except asyncio.CancelledError:
                pass




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
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://medvision-frontend-60an.onrender.com",
],
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
        async with httpx.AsyncClient(
            base_url=ORTHANC_URL,
            timeout=ORTHANC_TIMEOUT,
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
        ) as client:
            result = await client.get(
                f"/instances/{instance_id}/preview"
            )
            result.raise_for_status()

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="Preview or DICOM instance not found.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="Orthanc could not generate the preview.",
        ) from exc

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Orthanc preview request timed out.",
        ) from exc

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Orthanc.",
        ) from exc
        
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

        import nibabel as nib

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


@app.post("/predict/gradcam")
async def predict_gradcam(file: UploadFile = File(...)):
    from fastapi.responses import Response

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
        raise HTTPException(status_code=400, detail="Empty image.")

    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB.")

    try:
        with Image.open(BytesIO(content)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        ) from exc

    # Keep the uploaded image at its native resolution for the overlay so the
    # heatmap and the X-ray share the same coordinate system and aspect ratio.
    # Store it as uint8 and release the encoded upload early: full-resolution
    # float32 copies are what pushed the Free Render instance over 512 MB.
    original_image = image
    original = np.asarray(original_image, dtype=np.uint8)
    del content

    # The classifier input is unchanged: the same 224x224 version used before.
    tensor = app.state.image_transform(
        original_image.resize((224, 224))
    ).unsqueeze(0)

    model = app.state.classifier

    try:
        # Grad-CAM++ instead of first-order Grad-CAM. The classifier is a
        # frozen ImageNet ResNet18 backbone followed by global average pooling
        # and a single linear output neuron. With that architecture the
        # gradient of the logit with respect to layer4[-1] is spatially
        # uniform (every location contributes equally to the global average),
        # so plain Grad-CAM collapses to the linear-head CAM and can put the
        # entire positive response in one border column of the 7x7 feature
        # map. Grad-CAM++ uses second-order gradient weighting, which avoids
        # that collapse and distributes saliency over the cardiac region.
        grayscale_cam, probability = compute_gradcam_plusplus(model, tensor)

        # Grad-CAM is computed on the 7x7 feature map. Match the library's
        # 224x224 scaling first, then resize only the heatmap to the native
        # X-ray dimensions so it aligns with the original image.
        grayscale_cam = cv2.resize(
            grayscale_cam,
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )
        grayscale_cam = cv2.resize(
            grayscale_cam,
            (original_image.width, original_image.height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize the CAM and tint only the regions with meaningful
        # activation. show_cam_on_image() blends the full jet colormap at a
        # fixed 50% weight, which colours even zero-activation background and
        # makes a localized CAM look like it covers the whole image. Keeping
        # the original radiograph outside the significant region makes the
        # heatmap read as an overlay on the anatomy rather than a colour wash.
        cam = cv2.normalize(grayscale_cam, None, 0.0, 1.0, cv2.NORM_MINMAX)
        threshold = max(0.35, float(np.percentile(cam, 75)))
        alpha = np.clip(
            (cam - threshold) / max(1e-6, 1.0 - threshold),
            0.0,
            1.0,
        )
        overlay = blend_cam_overlay(original, cam, alpha)

    except Exception as exc:
        logger.exception("Grad-CAM generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Could not generate Grad-CAM.",
        ) from exc

    output = BytesIO()
    Image.fromarray(overlay).save(output, format="PNG")

    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={
            "X-Model-Probability": str(float(probability)),
            "X-Predicted-Label": str(
                int(probability >= app.state.classifier_threshold)
            ),
            "X-Decision-Threshold": str(app.state.classifier_threshold),
            "Cache-Control": "no-store",
        },
    )