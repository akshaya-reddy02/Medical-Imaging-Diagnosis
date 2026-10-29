
from fastapi.testclient import TestClient

from src import api


def setup_api_state():
    api.app.state.classifier = None
    api.app.state.classifier_threshold = 0.5
    api.app.state.image_transform = None
    api.app.state.segmenter = None
    api.app.state.load_errors = {}


def test_root_endpoint():
    setup_api_state()

    response = TestClient(api.app).get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Medical Imaging Diagnosis API"
    assert data["documentation"] == "/docs"
    assert data["clinical_use"] is False


def test_health_reports_degraded_when_models_unavailable():
    setup_api_state()

    response = TestClient(api.app).get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["classification_loaded"] is False
    assert data["segmentation_loaded"] is False
    assert data["clinical_use"] is False


def test_invalid_orthanc_instance_id_is_rejected():
    setup_api_state()

    response = TestClient(api.app).get(
        "/orthanc/instances/invalid-id/preview"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid Orthanc instance ID."


def test_invalid_orthanc_study_id_is_rejected():
    setup_api_state()

    response = TestClient(api.app).get(
        "/orthanc/studies/invalid%5Cid"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid Orthanc study ID."


def test_classification_returns_503_when_model_unavailable():
    setup_api_state()

    response = TestClient(api.app).post(
        "/predict/classification",
        files={
            "file": (
                "sample.png",
                b"not-a-real-image",
                "image/png",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Classification model is unavailable."


def test_classification_rejects_invalid_image():
    setup_api_state()
    api.app.state.classifier = object()
    api.app.state.image_transform = None

    with TestClient(api.app) as client:
        response = client.post(
            "/predict/classification",
            files={
                "file": (
                    "invalid.png",
                    b"this is not a valid image",
                    "image/png",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or unsupported image file."


def test_classification_rejects_oversized_image():
    setup_api_state()
    api.app.state.classifier = object()

    oversized_content = b"x" * (api.MAX_IMAGE_BYTES + 1)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict/classification",
            files={
                "file": (
                    "large.png",
                    oversized_content,
                    "image/png",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image exceeds the 20 MB limit."

def test_segmentation_returns_503_when_model_unavailable():
    setup_api_state()

    response = TestClient(api.app).post(
        "/predict/segmentation",
        files={
            "file": (
                "sample.nii.gz",
                b"dummy MRI content",
                "application/gzip",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Segmentation model is unavailable."


def test_segmentation_rejects_unsupported_file_type():
    setup_api_state()
    api.app.state.segmenter = object()

    response = TestClient(api.app).post(
        "/predict/segmentation",
        files={
            "file": (
                "sample.png",
                b"dummy content",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Upload a NIfTI MRI file ending in .nii or .nii.gz."
    )
