# MedVision – Limitations and Future Improvements

## 1. Current Limitations

### Model Accuracy
- The Cardiomegaly classifier is experimental and has not been clinically validated.
- Classification performance may vary across datasets, image quality, scanners, and patient populations.
- The current decision threshold may not be suitable for clinical use.

### Segmentation Performance
- Initial validation Dice scores are low, particularly for the anterior hippocampus.
- Only a small number of validation samples have been evaluated.
- Further training, tuning, and evaluation are required before making performance claims.

### Dataset Limitations
- Training datasets may not represent all hospitals, equipment, demographic groups, or clinical settings.
- Dataset imbalance and labeling quality can affect model performance.
- Results may not generalize to unseen datasets.

### Explainability
- Grad-CAM highlights image regions influencing the model output.
- A heatmap does not establish that a highlighted region is medically meaningful or correct.

### Orthanc Automation
- Orthanc study listing and preview integration have been tested.
- Automatic inference triggered by a newly uploaded study still requires end-to-end verification.

## 2. Future Improvements

- Train and evaluate models using larger and more diverse datasets.
- Improve segmentation through additional training, hyperparameter tuning, and validation.
- Evaluate classification using sensitivity, specificity, precision, recall, F1-score, and ROC-AUC.
- Test generalization on independent datasets.
- Add modality and study-type validation before automatic inference.
- Verify automatic Orthanc ingestion and inference using newly uploaded studies.
- Improve monitoring, error handling, security, and audit reporting.
- Conduct appropriate clinical validation and expert review before any real-world medical use.

## 3. Safety and Intended Use

MedVision is an educational and experimental project. It is not a medical device and is not validated for diagnosis, screening, treatment, or clinical decision-making. All model outputs must be reviewed as experimental demonstrations only.