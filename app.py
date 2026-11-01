from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "ResNet18"
    / "resnet18_state_dict.pth"
)
KNN_MODEL_PATH = Path(__file__).resolve().parent / "KNN" / "knn_model_latest.pkl"
SVM_MODEL_PATH = Path(__file__).resolve().parent / "SVM" / "svm_rbf_latest.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "Normal", 1: "Viral Pneumonia"}

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


@st.cache_resource
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


@st.cache_resource
def load_classical_models() -> tuple[dict, dict]:
    import pickle

    bundles = []
    for path, model_key in (
        (KNN_MODEL_PATH, "knn_model"),
        (SVM_MODEL_PATH, "svm_model"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        required_keys = {model_key, "scaler", "pca", "img_size"}
        if not required_keys.issubset(bundle):
            missing = sorted(required_keys - bundle.keys())
            raise ValueError(f"{path.name} is missing keys: {', '.join(missing)}")
        bundles.append(bundle)
    return bundles[0], bundles[1]


def classify_image(model: nn.Module, image: Image.Image) -> torch.Tensor:
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    return probabilities


def classical_features(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    grayscale = image.convert("L").resize(image_size)
    pixels = np.asarray(grayscale, dtype=np.float32)
    statistics = np.array(
        [pixels.mean(), pixels.std(), pixels.min(), pixels.max()],
        dtype=np.float32,
    )
    return np.concatenate([pixels.flatten(), statistics]).reshape(1, -1)


def classify_classical(
    bundle: dict, image: Image.Image, model_key: str
) -> np.ndarray:
    image_size = tuple(bundle["img_size"])
    features = classical_features(image, image_size)
    transformed = bundle["pca"].transform(bundle["scaler"].transform(features))
    model = bundle[model_key]
    return np.asarray(model.predict_proba(transformed)[0], dtype=np.float32)


st.set_page_config(
    page_title="Pneumonia Classification",
    page_icon="🩻",
    layout="centered",
)

st.title("Chest X-ray Pneumonia Classification")
st.write("Upload a chest X-ray image to receive a ResNet-18 prediction.")
st.warning(
    "Research/demo use only. This result is not a medical diagnosis and must "
    "not replace review by a qualified clinician."
)

try:
    model = load_model()
    knn_bundle, svm_bundle = load_classical_models()
except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"The uploaded file is not a valid image: {exc}")
        st.stop()

    st.image(image, caption="Uploaded chest X-ray", use_container_width=True)

    with st.spinner("Classifying image..."):
        resnet_probabilities = classify_image(model, image).numpy()
        knn_probabilities = classify_classical(knn_bundle, image, "knn_model")
        svm_probabilities = classify_classical(svm_bundle, image, "svm_model")

    ensemble_probabilities = (
        resnet_probabilities + knn_probabilities + svm_probabilities
    ) / 3
    predicted_index = int(ensemble_probabilities.argmax())

    predicted_label = LABELS[predicted_index]
    st.subheader(f"Ensemble prediction: {predicted_label}")
    st.caption("Final score is the equal-weight average of ResNet18, KNN, and SVM.")

    score_table = {
        LABELS[class_index]: {
            "ResNet18": float(resnet_probabilities[class_index]),
            "KNN": float(knn_probabilities[class_index]),
            "SVM": float(svm_probabilities[class_index]),
            "Ensemble": float(ensemble_probabilities[class_index]),
        }
        for class_index in LABELS
    }
    st.dataframe(
        {
            "Class": list(score_table),
            "ResNet18": [score_table[label]["ResNet18"] for label in score_table],
            "KNN": [score_table[label]["KNN"] for label in score_table],
            "SVM": [score_table[label]["SVM"] for label in score_table],
            "Ensemble": [score_table[label]["Ensemble"] for label in score_table],
        },
        hide_index=True,
        column_config={
            "ResNet18": st.column_config.NumberColumn(format="%.2f"),
            "KNN": st.column_config.NumberColumn(format="%.2f"),
            "SVM": st.column_config.NumberColumn(format="%.2f"),
            "Ensemble": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.caption(f"Running on: {DEVICE}")
