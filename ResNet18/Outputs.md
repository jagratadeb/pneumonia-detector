# ResNet-18 — Output Files Reference

> All files below are outputs from training a ResNet-18 model for binary pneumonia classification (Normal vs Viral Pneumonia) on the COVID-19 Radiography Database.

---

## Model Files

### `resnet18_full_model.pth`
The **main model file**. Contains both the full architecture and trained weights saved together using `torch.save(model, ...)`. This is the easiest file to reload for inference — no need to redefine the model architecture first.

### `best_resnet18.pth`
A checkpoint saved **automatically during training** at the epoch where validation loss was lowest. Saved as a state dict (weights only). If training was stopped early, this may actually be a slightly better checkpoint than the final model.

### `resnet18_state_dict.pth`
Contains **weights only**, with no architecture. Lighter than the full model file but requires you to manually rebuild the model architecture using `build_resnet18()` before loading. Useful for portability across different PyTorch versions.

---

## Metrics

### `resnet18_results.csv`
A single-row CSV containing all test set evaluation metrics:
- **Accuracy** — overall correct predictions
- **Precision** — of all predicted Pneumonia cases, how many were actually Pneumonia
- **Recall (Sensitivity)** — of all actual Pneumonia cases, how many were correctly detected
- **Specificity** — of all actual Normal cases, how many were correctly identified
- **F1-Score** — harmonic mean of Precision and Recall
- **ROC-AUC** — model's ability to distinguish between classes across all thresholds

---

## Plots

### `resnet18_confusion_matrix.png`
A heatmap showing the breakdown of predictions on the test set across 4 categories — True Positives, True Negatives, False Positives, and False Negatives. Useful for understanding where the model makes mistakes (e.g. missing actual Pneumonia cases vs flagging healthy patients).

### `resnet18_roc_curve.png`
Plots True Positive Rate vs False Positive Rate across all classification thresholds. The AUC (Area Under Curve) score is shown in the legend — closer to 1.0 means better model discrimination. The diagonal dashed line represents a random classifier (AUC = 0.5).

### `resnet18_training_curves.png`
Two side-by-side plots showing how the model improved over each epoch:
- **Left** — Training Loss vs Validation Loss. A large gap between the two indicates overfitting.
- **Right** — Training Accuracy vs Validation Accuracy. Useful for spotting when the model stopped generalizing.

### `resnet18_gradcam.png`
Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations for 6 random test images. Each row shows:
- **Original** — the raw chest X-ray with its true label
- **Grad-CAM** — heatmap of which regions most influenced the model's decision
- **Overlay** — heatmap blended onto the original image

Bright red/yellow regions indicate areas the model focused on most. Useful for validating that the model is looking at clinically relevant regions (lung area) rather than artifacts.
