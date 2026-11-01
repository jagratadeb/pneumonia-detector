# ResNet-18 Deployment Guide

This guide describes how to deploy the trained ResNet-18 chest X-ray classifier
from this repository as a beginner-friendly Streamlit web application. The
application accepts a chest X-ray image, runs inference, and displays the
predicted class and probabilities. A FastAPI option is included later for
deployments that need a separate backend API.

> **Medical safety notice:** This model is a research project and is not a
> medical device. It must not be used as the sole basis for diagnosis,
> treatment, triage, or patient-discharge decisions. A qualified clinician must
> review the original image and all model outputs.

## 1. What is being deployed

The ResNet-18 notebook configures:

- Architecture: `torchvision.models.resnet18`
- Initial weights: ImageNet `ResNet18_Weights.IMAGENET1K_V1`
- Replaced head: `nn.Linear(model.fc.in_features, 2)`
- Input: three-channel RGB image, `224 x 224`
- Inference preprocessing:
  1. Convert to RGB.
  2. Resize to `(224, 224)`.
  3. Convert to a tensor.
  4. Normalize with mean `[0.485, 0.456, 0.406]` and standard deviation
     `[0.229, 0.224, 0.225]`.
- Output: two logits; apply `softmax` to obtain class probabilities.
- Class mapping: `0 = Normal`, `1 = Viral Pneumonia`.
- Training configuration: Adam, learning rate `0.001`, batch size `32`,
  maximum `20` epochs, and early stopping patience `5`.

The notebook's ResNet-18 training code loads only `Normal` and
`Viral Pneumonia`. The root README describes a broader pneumonia grouping
elsewhere, but that broader grouping must not be assumed for this artifact.
Confirm the intended label scope with a domain owner before production use.

The reported test metrics in
[`ResNet18/resnet18_results.csv`](ResNet/ResNet18/resnet18_results.csv) are:

| Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 0.9896 | 0.9466 | 0.9653 | 0.9928 | 0.9559 | 0.9992 |

These are retrospective test-set metrics, not a guarantee of performance on
new hospitals, scanners, populations, image formats, or acquisition protocols.

## 2. Run the Streamlit application locally

The ready-to-run application is
[`app.py`](app.py). It loads the recommended state-dict artifact, applies the
same preprocessing used during evaluation, and supports PNG and JPEG uploads.

Create and activate a virtual environment on Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the address printed by Streamlit, normally
`http://localhost:8501`. Upload an image, then review the displayed prediction
and class probabilities.

Stop the application with `Ctrl+C` in the terminal running Streamlit.

## 3. Deploy with Streamlit Community Cloud

Streamlit Community Cloud can deploy this repository directly from GitHub and
does not require Docker.

1. Push `app.py`, `requirements.txt`, and the `ResNet/ResNet18/` model artifact
   to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
3. Select **Create app**.
4. Choose this repository and branch.
5. Set **Main file path** to `app.py`.
6. Select **Deploy** and wait for dependencies and the model to download.
7. Open the generated application URL and test it with a non-sensitive image.

The model file is approximately 45 MB in the current repository, so confirm
that the repository and selected Streamlit plan can accommodate it. If the
model is moved out of Git, store it in private object storage and download it
at startup using authenticated access; never commit credentials to the
repository.

Whenever `app.py`, `requirements.txt`, or the model changes, redeploy and
repeat the smoke test. Do not upload identifiable patient images to a public
demo application.

## 4. Choose and verify the model artifact

The ResNet-18 artifacts are in
[`ResNet/ResNet18/`](ResNet/ResNet18/):

| File | Use |
|---|---|
| `resnet18_state_dict.pth` | Recommended deployment artifact. Rebuild the architecture and load weights. |
| `best_resnet18.pth` | Weights-only validation checkpoint. Use only after verifying its contents and provenance. |
| `resnet18_full_model.pth` | Serialized model and weights. Requires the matching Python/PyTorch class environment and should only be loaded from a trusted source. |

Prefer `resnet18_state_dict.pth` because it avoids coupling deployment to
Python pickle paths and makes the architecture explicit. Keep the model file
outside the public upload directory and pin its checksum in the release
record.

PowerShell checksum command:

```powershell
Get-FileHash .\ResNet\ResNet18\resnet18_state_dict.pth -Algorithm SHA256
```

Linux/macOS checksum command:

```bash
sha256sum ResNet/ResNet18/resnet18_state_dict.pth
```

Do not load an untrusted `.pth` file. PyTorch serialized full models can execute
code during deserialization.

## 5. Prepare the deployment environment

Use Python 3.10 or newer and a supported PyTorch/torchvision pair. Install a
CPU build for a small service or the matching CUDA build when GPU inference is
required. Pin the versions after testing them with the selected artifact.

Create an isolated environment on Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision fastapi "uvicorn[standard]" pillow python-multipart
```

Create an isolated environment on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision fastapi "uvicorn[standard]" pillow python-multipart
```

For a GPU deployment, install the PyTorch wheels for the CUDA version
installed on the host by following the official PyTorch installation selector.
Do not mix arbitrary `torch` and `torchvision` versions.

## 6. Reference inference service

Create `app.py` next to the repository root, or place the equivalent code in
the service package used by your deployment platform:

```python
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


MODEL_PATH = Path("ResNet/ResNet18/resnet18_state_dict.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "Normal", 1: "Viral Pneumonia"}

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


def load_model() -> nn.Module:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
app = FastAPI(title="ResNet-18 Pneumonia Classifier", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax().item())
    return {
        "label": LABELS[predicted_index],
        "class_index": predicted_index,
        "probabilities": {
            LABELS[index]: round(float(probabilities[index]), 6)
            for index in LABELS
        },
        "model": "resnet18",
    }
```

Run it from the repository root:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

For a remotely reachable service, bind behind a reverse proxy or private
network rather than exposing the development server directly:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Start with one worker. Each worker loads its own copy of the model, so multiple
workers can multiply RAM/VRAM usage. Scale workers only after measuring
throughput and available memory.

## 7. Test the deployment

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Prediction request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "accept: application/json" `
  -F "file=@C:\path\to\chest_xray.png"
```

Expected response shape:

```json
{
  "label": "Viral Pneumonia",
  "class_index": 1,
  "probabilities": {
    "Normal": 0.031245,
    "Viral Pneumonia": 0.968755
  },
  "model": "resnet18"
}
```

Before release, test at least:

1. A valid PNG and JPEG.
2. A grayscale image, which must be converted to RGB without failure.
3. An empty upload.
4. A non-image file.
5. An unsupported MIME type.
6. A deliberately oversized upload, after adding the platform's request-size
   limit.
7. A known test image whose result is recorded as a deployment smoke test.

Do not use the test-set images as an operational success metric after release.
Maintain a separate, access-controlled validation set and record model version,
artifact checksum, preprocessing version, and decision threshold for every
evaluation.

## 8. Production hardening

- Put the service behind HTTPS and an authenticated reverse proxy.
- Do not log uploaded images, raw request bodies, patient identifiers, or
  clinical metadata. If request tracing is necessary, use a random request ID.
- Enforce upload size, timeout, rate, and concurrency limits at the proxy and
  application layers.
- Validate the decoded image, not only its filename or MIME type.
- Run the service as a non-root user with a read-only model directory.
- Pin dependencies and rebuild from a locked environment.
- Restrict network access to approved clients; do not make the endpoint public by
  default.
- Monitor latency, error rate, rejected uploads, input dimensions, and class
  distribution without retaining sensitive image content.
- Define an operational fallback: failed inference must be reported as an
  error and routed for human review, never converted into a normal or
  pneumonia prediction.

The API currently returns the argmax class at a threshold implicit in the
largest softmax probability. A clinical deployment should establish and
version an operating threshold using an independent validation study, especially
when the cost of missed pneumonia and false alarms differs.

## 9. Container deployment (optional)

A minimal CPU container can be built with:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision fastapi "uvicorn[standard]" pillow python-multipart

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Build and run from the repository root:

```bash
docker build -t pneumonia-resnet18:1.0 .
docker run --rm -p 8000:8000 pneumonia-resnet18:1.0
```

For production, replace the unpinned install step with a tested lock file,
exclude notebooks and unrelated artifacts from the image where appropriate,
scan the image, and publish the exact image digest alongside the model
checksum. Use a CUDA base image and the matching PyTorch installation only when
GPU inference has been tested on the target host.

## 10. Release checklist

- [ ] Confirm the intended labels are exactly `Normal` and `Viral Pneumonia`.
- [ ] Confirm the artifact checksum and provenance.
- [ ] Verify the architecture and preprocessing code match the training notebook.
- [ ] Run the health and prediction smoke tests.
- [ ] Test malformed, unsupported, empty, and oversized uploads.
- [ ] Evaluate on an independent validation set and document sensitivity,
      specificity, calibration, and the selected operating threshold.
- [ ] Configure HTTPS, authentication, rate limits, request limits, and
      non-sensitive logging.
- [ ] Confirm the service can fail closed and route uncertain or failed cases to
      qualified human review.
- [ ] Record model version, code revision, dependency versions, hardware, and
      artifact checksum.
- [ ] Obtain the required clinical, privacy, security, and regulatory approvals
      before any real-world use.
