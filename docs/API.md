# MedVision API Documentation

## Overview

MedVision provides REST API endpoints through FastAPI for medical image classification, MRI segmentation, Grad-CAM visualization, and Orthanc study access.

**Base URL:** `http://localhost:8000`

**Interactive API documentation:** `http://localhost:8000/docs`

All AI outputs are experimental and must not be used for clinical diagnosis or treatment.

## 1. General Endpoints

### GET `/`

Checks whether the API service is accessible.

### GET `/health`

Returns backend health information, including model loading status and Orthanc configuration.

Example response:

```json
{
  "status": "ok",
  "classification_loaded": true,
  "segmentation_loaded": true,
  "clinical_use": false
}
```

## 2. Orthanc Endpoints

### GET `/orthanc/studies`

Retrieves available imaging studies from the Orthanc server.

### GET `/orthanc/studies/{study_id}`

Retrieves information about a specific study.

**Path parameter:** `study_id` — Orthanc study identifier.

### GET `/orthanc/studies/{study_id}/series`

Retrieves the series associated with a specific study.

### GET `/orthanc/instances/{instance_id}/preview`

Retrieves a preview image for a specified Orthanc instance.

**Path parameter:** `instance_id` — Orthanc instance identifier.

## 3. AI Prediction Endpoints

### POST `/predict/classification`

Performs experimental chest X-ray classification for Cardiomegaly.

**Request:**
- Content type: `multipart/form-data`
- Field: `file`
- Accepted input: image file
- Maximum file size: 20 MB

**Example response:**

```json
{
  "target": "Cardiomegaly",
  "probability": 0.3759,
  "threshold": 0.5,
  "predicted_label": 0,
  "notice": "Experimental result only. Not validated for clinical diagnosis or treatment.",
  "clinical_use": false
}
```

`predicted_label` is `1` when the probability meets or exceeds the configured threshold; otherwise it is `0`.

### POST `/predict/segmentation`

Performs experimental 3D MRI segmentation using the 3D U-Net model.

**Request:**
- Content type: `multipart/form-data`
- Field: `file`
- Accepted formats: `.nii` and `.nii.gz`
- Maximum file size: 100 MB

**Response:**
- Downloadable NIfTI segmentation mask (`.nii.gz`)
- `X-Clinical-Use`: `false`
- `X-Predicted-Classes`: predicted class identifiers

### POST `/predict/gradcam`

Generates a Grad-CAM heatmap overlay for an uploaded image.

**Request:**
- Content type: `multipart/form-data`
- Field: `file`
- Maximum file size: 20 MB

**Response:**
- Content type: `image/png`
- `X-Model-Probability`: model probability
- `X-Predicted-Label`: predicted classification label
- `X-Decision-Threshold`: configured threshold
- `Cache-Control`: `no-store`

The PNG overlay highlights image regions that influence the model's output. It is an explainability aid, not proof of a medical finding.

## 4. Common Error Responses

| HTTP status | Meaning |
|---|---|
| 400 | Empty, invalid, or unsupported input |
| 413 | Uploaded file exceeds the size limit |
| 500 | Inference or processing failure |
| 502 | Unexpected response from Orthanc |
| 503 | Required AI model is unavailable |

Error responses include a `detail` field describing the error.

## 5. Testing the API

Open `http://localhost:8000/docs` in a browser.

1. Select an endpoint.
2. Click **Try it out**.
3. Provide the required input or upload file.
4. Click **Execute**.
5. Inspect the response and HTTP status.

## 6. Safety Notice

MedVision is an educational and experimental project. Its models are not clinically validated, and their outputs must not guide patient diagnosis, treatment, or other clinical decisions.