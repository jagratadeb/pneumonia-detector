# Pneumonia Detector

## 1. Summary

Pneumonia Detector is a research/demo project for binary classification of
chest X-ray images. The deployed checkpoint and application use the following
labels:

- **Normal**
- **Viral Pneumonia**

The deployed application combines three independently trained classifiers:

1. A ResNet-18 convolutional neural network.
2. A k-nearest neighbors (KNN) classifier.
3. An RBF-kernel support vector machine (SVM).

The Streamlit application runs all three models for each uploaded image and
uses the arithmetic mean of their class probabilities as the ensemble result.
It can process multiple images in one session, show individual model scores,
display confidence charts, and export a batch summary as CSV.

The research documentation describes a broader pneumonia-positive grouping that
can include Viral Pneumonia, Lung Opacity, and COVID-19 categories. The
deployed artifact documented here is a two-class `Normal`/`Viral Pneumonia`
artifact; the broader grouping must not be inferred from the application
labels.

This is a research project and is **not a medical device**. Its predictions
must not be used as a diagnosis or as the sole basis for treatment, triage, or
other clinical decisions. A qualified clinician must review the original image
and all relevant clinical information.

## 2. Technical details

### Application

The user interface is implemented in [app.py](app.py) with Streamlit. The
application:

- Accepts PNG and JPEG chest X-ray uploads.
- Supports up to 10 images by default.
- Validates image format, file size, and dimensions.
- Converts images to RGB before inference, including grayscale uploads.
- Loads model artifacts once using Streamlit resource caching.
- Uses CPU when CUDA is unavailable and reports the inference device.
- Shows the predicted class, confidence, per-model probabilities, and an
  ensemble confidence chart.
- Displays an uncertainty warning when ensemble confidence is below 70%.
- Provides a batch results table and CSV download.
- Keeps only non-image prediction results in the current session.

Uploaded image bytes are processed in memory. The application does not save
uploaded images or their metadata, and users should upload only non-sensitive
images.

### Research objective

The research objective is to compare traditional machine-learning methods with
transfer-learning convolutional neural networks under a common evaluation
protocol. The models are assessed using accuracy, precision, recall
(sensitivity), specificity, F1-score, ROC-AUC, and confusion matrices.

### Inference workflow

For each upload, the application validates and decodes the image, sends the
same image through the ResNet-18, KNN, and SVM pipelines, verifies that all
three outputs contain finite two-class scores, and averages those scores.
Model artifacts are cached after the first successful load. Invalid uploads or
incompatible artifacts produce an explicit error instead of a prediction.

### Ensemble decision

For an input image, each model produces a two-class probability vector:

```text
ResNet18 = [p_normal, p_viral_pneumonia]
KNN      = [p_normal, p_viral_pneumonia]
SVM      = [p_normal, p_viral_pneumonia]
```

The final ensemble score is:

```text
ensemble = (ResNet18 + KNN + SVM) / 3
```

The class with the larger ensemble probability is selected. The current
implementation uses equal model weights; these weights have not been
clinically calibrated.

### Repository structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface and inference pipeline |
| `ResNet18/` | ResNet-18 artifacts, metrics, and visualizations |
| `KNN/` | KNN artifact, metrics, and evaluation outputs |
| `SVM/` | SVM artifact, metrics, and evaluation outputs |
| `requirements.txt` | Tested Python package versions |
| `runtime.txt` | Python runtime selected by Streamlit Community Cloud |

### Streamlit deployment

The application is deployed with Python 3.13, as specified in
`runtime.txt`. Keep this file at the repository root when deploying to
Streamlit Community Cloud. Python 3.14 can cause transitive dependencies such
as PyArrow and scikit-learn to be built from source instead of installed from
prebuilt wheels, which can fail during dependency installation.

After adding or changing `runtime.txt`, restart the app or deploy a new commit
so Streamlit Cloud recreates its environment. If the deployment log still
shows Python 3.14, clear the app cache/reboot the app and verify that the
selected branch contains the updated file.

## 3. Model details

### ResNet-18

The deep-learning model is based on `torchvision.models.resnet18` with the
final fully connected layer replaced by a two-output linear layer:

```text
ResNet-18 backbone -> Linear(in_features, 2) -> Softmax
```

The deployment artifact is
`ResNet18/resnet18_state_dict.pth`. The architecture is reconstructed in the
application and the state dictionary is loaded with the selected inference
device.

The documented training configuration for this family of CNN experiments is:

- Optimizer: Adam
- Learning rate: `0.001`
- Batch size: `32`
- Maximum epochs: `20`
- Early-stopping patience: `5`
- Input size: `224 x 224`
- Output classes: Normal and Viral Pneumonia

### KNN

The classical KNN model uses:

- `k = 7`
- Euclidean distance
- Uniform neighbor weighting
- A fitted scaler
- PCA before classification

The stored bundle includes the KNN estimator, scaler, PCA transformer, and
image-size metadata. The reported PCA configuration retains approximately
91.48% of variance.

### SVM

The classical SVM model uses:

- RBF kernel
- `C = 1.0`
- `gamma = scale`
- A fitted scaler
- PCA before classification

The stored bundle includes the SVM estimator, scaler, PCA transformer, and
image-size metadata. The reported PCA configuration retains approximately
91.50% of variance.

### Comparative architectures

The broader comparative study also evaluates ResNet-34, ResNet-50, and
VGGNet-16. The Streamlit application currently loads the ResNet-18, KNN, and
SVM artifacts listed above.

### Reported evaluation metrics

The following metrics are recorded in the repository’s model output files.
They are retrospective test-set results and should not be interpreted as
expected performance on new hospitals, scanners, populations, or acquisition
protocols.

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 98.96% | 94.66% | 96.53% | 99.28% | 95.59% | 99.92% |
| KNN (`k=7`) | 97.17% | 86.96% | 89.11% | 98.23% | 88.02% | 99.04% |
| SVM (RBF) | 98.90% | 94.63% | 96.04% | 99.28% | 95.33% | 99.73% |

The ResNet-18 metrics are recorded in
`ResNet18/resnet18_results.csv`; the KNN and SVM metrics are recorded in their
respective report and JSON artifacts.

For comparison, the study reports the following results for the additional
CNN architectures:

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-34 | 99.02% | 95.12% | 96.53% | 99.35% | 95.82% | 99.92% |
| ResNet-50 | 99.31% | 97.98% | 96.04% | 99.74% | 97.00% | 99.93% |
| VGGNet-16 | 99.02% | 96.95% | 94.55% | 99.61% | 95.74% | 99.90% |

The ResNet-34, ResNet-50, and VGGNet-16 figures are reported study results,
not artifacts loaded or independently re-evaluated by the Streamlit
application. The KNN and SVM values above are the test-set values in their
stored evaluation reports; the source study also contains a comparative table
with validation-oriented values, so metric tables should always be read with
their split and artifact provenance.

## 4. Image preprocessing

### ResNet-18 preprocessing

Every input image is:

1. Converted to RGB.
2. Resized to `224 x 224` pixels.
3. Converted to a PyTorch tensor.
4. Normalized channel-wise using ImageNet statistics:

   ```text
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
   ```

The normalized tensor is given a batch dimension and moved to the selected
inference device. Inference runs under `torch.inference_mode()`.

### Classical-model preprocessing

For KNN and SVM, the image is:

1. Converted to grayscale.
2. Resized to the image dimensions stored in the model bundle.
3. Converted to a floating-point pixel array.
4. Flattened into a feature vector.
5. Extended with four intensity statistics:
   - Mean
   - Standard deviation
   - Minimum
   - Maximum
6. Transformed using the model’s stored scaler and PCA pipeline.

The transformed feature vector is passed to the stored KNN or SVM estimator.

## 5. Research notes and limitations

### Dataset and validation

The research models use the
[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
from Kaggle. Images are organized into normal and pneumonia-affected
categories. The documented split is:

| Subset | Proportion | Purpose |
|---|---:|---|
| Training | 70% | Model fitting and parameter optimization |
| Validation | 15% | Hyperparameter selection and early stopping |
| Testing | 15% | Final performance evaluation |

The subsets are intended to be mutually exclusive. The repository does not
include a data manifest or patient-level split manifest, so the exact
composition of the reported subsets cannot be reconstructed from the
application alone. Reported metrics depend on the exact dataset version, class
grouping, preprocessing, and split used during training.

### Standardization protocol

The comparative experiments use a common protocol:

1. Identical train/validation/test proportions.
2. No use of the test set during model selection.
3. Standardized image dimensions and model-specific scaling.
4. The same six primary evaluation metrics.
5. Documented hyperparameters and training procedures.

Before any real-world use, the models require evaluation on an independent
dataset with representative variation in:

- Hospitals and geographic regions.
- Scanner and acquisition protocols.
- Patient demographics.
- Image quality, positioning, and artifacts.
- Disease prevalence and co-existing conditions.

### Interpretability

The repository includes a ResNet-18 Grad-CAM visualization artifact,
`ResNet18/resnet18_gradcam.png`. A visualization can indicate which regions
influenced a model output, but it does not prove that the model used
clinically valid evidence. Interpretability outputs should be reviewed by
qualified experts.

### Probability and threshold limitations

Softmax and classifier probabilities are model scores, not guaranteed
clinical probabilities. The current ensemble uses equal averaging and a
configurable uncertainty threshold, but it has not been calibrated for a
clinical operating point. Threshold selection should account for the relative
costs of false negatives and false positives.

### Reproducibility

The saved artifacts contain fitted model components needed for inference.
The application repository does not contain the original training scripts or
the complete dataset, so reproducing the reported metrics requires preserving
the training source, dataset version, split manifest, and experiment
configuration in addition to these inference artifacts.
Reproducible research should additionally preserve:

- Dataset version and access details.
- Patient-level or study-level split rules.
- Random seeds.
- Training and validation preprocessing.
- Exact package and hardware versions.
- Model checksums.
- Complete experiment configurations and logs.

### Known implementation boundaries

- The application performs inference only; it does not retrain or fine-tune
  models.
- Grad-CAM is included as a repository artifact, but interactive Grad-CAM
  generation is not part of the current application.
- The displayed confidence values are model scores. They are not calibrated
  probabilities or measures of clinical risk.
- The ensemble average is a simple equal-weight rule and has not been
  validated as an optimal combination.

### Research artifacts

The repository includes model checkpoints, evaluation reports, confusion
matrices, ROC curves, training curves, and a ResNet-18 Grad-CAM visualization.
These artifacts support inspection of performance and training behavior but do
not replace external validation.

### Responsible use

This project should be treated as an educational and research prototype.
Results must be reviewed by a qualified clinician and interpreted together
with the original radiograph, patient history, symptoms, and other diagnostic
evidence. Applicable privacy, security, regulatory, and institutional review
requirements must be addressed before using the system with clinical data.

## 6. References and contributors

### Dataset

Rahman, T. et al. **COVID-19 Radiography Database.** Kaggle, 2020.
[Dataset page](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

### Contributors

- [Jagrata Deb](https://github.com/jagratadeb)
- [Baibhab Majumder](https://github.com/Baibhab2004)
- [Anik Roy](https://github.com/o-yol)
- [Ayushman Nanda](https://github.com/ayushmannanda0)
- [Ayush Pradhan](https://github.com/ayushpradhan1704)
- [Vansh Srivastava](https://github.com/vanshsrivastava09)
