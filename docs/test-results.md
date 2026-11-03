# MedVision Test Results

## 1. Backend API Tests

Command:

```cmd
.\venv\Scripts\python.exe -m pytest tests\test_api.py -q
```

**Result:** 9 passed, 3 warnings.

The warnings included an HTTP client deprecation notice and a MONAI image-decoding fallback notice.

## 2. Frontend Build

Command:

```cmd
cd frontend
npm.cmd run build
```

**Result:** Successful production build.

## 3. Backend Health Check

Endpoint: `GET /health`

**Result:** Successful.

- Backend status: OK
- Classification model: Loaded
- Segmentation model: Loaded
- Orthanc configuration: Available
- Clinical use: Disabled

## 4. Classification Test

Sample image: `CX_patient00001_study1_view1_frontal.jpg`

- Target: Cardiomegaly
- Predicted label: Negative (0)
- Probability: 37.59%
- Decision threshold: 50%

This is an experimental model result, not a clinical finding.

## 5. Segmentation Evaluation

Validation metrics from two samples:

| Sample | Anterior Dice | Posterior Dice |
|---|---:|---:|
| hippocampus_269 | 0.2047 | 0.4805 |
| hippocampus_048 | 0.1626 | 0.4191 |

The results indicate that segmentation performance needs improvement.

## 6. Grad-CAM Integration

**Result:** Verified.

The backend generated a PNG heatmap overlay, and the React dashboard displayed it successfully.

## 7. Doctor Review and Audit Log

**Result:** Verified.

The Approve/Reject workflow was tested. A review decision was saved and appeared in the audit history.

## 8. Orthanc Integration

**Result:** Basic integration verified.

The Orthanc server was running, study information and preview images were accessible through the backend, and the dashboard displayed the study.

Automatic inference triggered by a newly uploaded Orthanc study still requires a separate end-to-end verification.

## 9. Docker Deployment

**Result:** Backend container deployment verified.

The backend container started successfully, and the health endpoint reported both models loaded.

## 10. Overall Summary

Core backend tests, frontend build, model loading, Grad-CAM display, review logging, Orthanc preview access, and Docker backend startup were verified.

Automatic inference on a newly uploaded Orthanc study remains pending. Segmentation performance also requires further improvement.

All outputs are experimental and are not intended for clinical use.