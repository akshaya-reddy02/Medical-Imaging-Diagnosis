# Medical Imaging Diagnosis Platform

An educational medical-imaging AI project developed as part of the Nexvia Technologies Data Science internship. The platform combines medical-image classification, experimental 3D segmentation, Grad-CAM explainability, DICOM study access through Orthanc, and a review workflow with audit logging.

> **Important:** This is an educational prototype, not a clinically validated diagnostic system. AI predictions must not be used to make medical decisions.

## Features

- **Medical image classification:** Experimental ResNet18-based classification.
- **3D segmentation:** Experimental 3D U-Net workflow for hippocampus segmentation.
- **Explainability:** Grad-CAM visualizations for model predictions.
- **DICOM integration:** Orthanc access for sample medical-imaging studies.
- **Backend API:** FastAPI endpoints for imaging workflows and review operations.
- **Frontend:** React and Vite interface for viewing studies, predictions, and review history.
- **Audit logging:** SQLite-based record of review decisions.
- **Data management:** DVC tracking for selected image data and the segmentation dataset archive.
- **Container support:** Docker Compose configuration for supporting services.

## Technology Stack

- Python
- PyTorch and torchvision
- FastAPI
- React and Vite
- Orthanc
- Docker Compose
- DVC
- SQLite
- MONAI — integration and usage must be verified before claiming complete support

## Architecture

1. Sample medical images and DICOM studies provide the input data.
2. Orthanc stores and serves DICOM studies.
3. The FastAPI backend provides imaging and review endpoints.
4. Experimental classification and segmentation workflows generate predictions.
5. Grad-CAM visualizations help illustrate image regions influencing classification predictions.
6. The React frontend displays study information, predictions, and review controls.
7. Review decisions are recorded in the audit database.

## Project Structure

```text
Medical-Imaging-Diagnosis/
├── data/
│   ├── processed/
│   │   ├── train_manifest.csv
│   │   └── val_manifest.csv
│   ├── sample/
│   │   └── images.dvc
│   └── segmentation/
│       └── Task04_Hippocampus/
├── frontend/
├── models/
├── reports/
├── src/
│   ├── api.py
│   └── audit_log.py
├── tests/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Model weights, generated files, and local audit data may be excluded from Git. Dataset files tracked through DVC require access to the corresponding DVC data storage.

## Prerequisites

- Python 3.11
- Node.js and npm
- Docker Desktop
- Git
- DVC

## Installation

Run commands from the project root directory.

### 1. Set up Python

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```cmd
cd frontend
npm install
cd ..
```

### 3. Start the backend

```cmd
venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

Backend: `http://127.0.0.1:8000`

API documentation: `http://127.0.0.1:8000/docs`

### 4. Start the frontend

Open another Command Prompt window, navigate to the project root, and run:

```cmd
cd frontend
npm run dev
```

Frontend: `http://localhost:5173/`

### 5. Start supporting services

Review `docker-compose.yml` and ensure Docker Desktop is running. Start the configured services from the project root:

```cmd
docker compose up -d
```

Orthanc is expected at `http://127.0.0.1:8042/` when configured accordingly.

### 6. Check DVC data

```cmd
venv\Scripts\python.exe -m dvc status
```

To retrieve data on another machine, configure an accessible DVC remote and pull the required tracked data.

## Testing

Run the available automated tests from the project root:

```cmd
venv\Scripts\python.exe -m pytest tests
```

Build the frontend:

```cmd
cd frontend
npm run build
```

Test results should be recorded from the actual environment. A successful frontend build does not establish that every backend, model, or integration workflow is correct.

## Model Evaluation

The project includes experimental classification and segmentation workflows and generated evaluation reports under `reports/`.

Report metrics should be interpreted in light of the dataset, validation method, sample size, and limitations. Do not interpret experimental predictions as clinical diagnoses or assume that results generalize to other hospitals, populations, or imaging devices.

## Data Management

DVC metadata is used to track selected sample-image data and the hippocampus dataset archive. The actual data files are stored outside normal Git tracking.

A DVC remote has not yet been confirmed. Until one is configured and accessible, cloning the Git repository alone may not be sufficient to reproduce the data-dependent workflows.

## Limitations and Responsible Use

- This project is an educational prototype.
- Model performance and generalization require further independent validation.
- Training data may not represent all patient populations, scanners, or acquisition settings.
- Grad-CAM visualizations are explanatory aids, not proof that a prediction is correct.
- Predictions may be wrong, and confidence scores are not necessarily calibrated probabilities.
- Secure deployment, privacy controls, clinical validation, monitoring, and expert oversight would be required before any real-world clinical use.

## Future Improvements

- Complete and document MONAI integration.
- Strengthen automated and end-to-end tests.
- Add reproducible DVC remote storage.
- Document dataset provenance, model evaluation, and limitations.
- Complete a recorded demonstration and architecture/API documentation.
- Improve deployment, monitoring, and operational safeguards.

## Internship Project

Developed for the Nexvia Technologies Data Science internship.

This repository is intended for learning, demonstration, and technical evaluation—not clinical diagnosis.
