from pathlib import Path
import pickle

import matplotlib.pyplot as plt
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
MAX_UPLOADS = 10
MAX_FILE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 4096
UNCERTAIN_CONFIDENCE = 0.70

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


def validate_image(image: Image.Image) -> None:
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("The image has invalid dimensions.")
    if max(width, height) > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions must not exceed {MAX_IMAGE_DIMENSION} pixels."
        )


def classify_uploaded_image(
    image: Image.Image,
    model: nn.Module,
    knn_bundle: dict,
    svm_bundle: dict,
) -> dict[str, np.ndarray | int]:
    validate_image(image)
    resnet_probabilities = classify_image(model, image).numpy()
    knn_probabilities = classify_classical(knn_bundle, image, "knn_model")
    svm_probabilities = classify_classical(svm_bundle, image, "svm_model")
    probabilities = np.vstack(
        [resnet_probabilities, knn_probabilities, svm_probabilities]
    )
    if probabilities.shape != (3, len(LABELS)) or not np.isfinite(probabilities).all():
        raise RuntimeError("One or more models returned invalid prediction scores.")

    ensemble_probabilities = probabilities.mean(axis=0)
    predicted_index = int(ensemble_probabilities.argmax())
    return {
        "predicted_index": predicted_index,
        "resnet": resnet_probabilities,
        "knn": knn_probabilities,
        "svm": svm_probabilities,
        "ensemble": ensemble_probabilities,
    }


def display_score_chart(result: dict[str, np.ndarray | int]) -> None:
    model_scores = {
        "ResNet18": result["resnet"],
        "KNN": result["knn"],
        "SVM": result["svm"],
        "Ensemble": result["ensemble"],
    }
    figure, axis = plt.subplots(figsize=(8, 3.5))
    positions = np.arange(len(LABELS))
    width = 0.2
    for offset, (model_name, scores) in enumerate(model_scores.items()):
        axis.bar(
            positions + (offset - 1.5) * width,
            scores,
            width,
            label=model_name,
        )
    axis.set_xticks(positions, [LABELS[index] for index in LABELS])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability")
    axis.set_title("Model confidence comparison")
    axis.legend(ncols=2)
    axis.grid(axis="y", alpha=0.25)
    st.pyplot(figure)
    plt.close(figure)


st.set_page_config(
    page_title="Pneumonia Classification",
    page_icon="🩻",
    layout="centered",
)

st.title("Chest X-ray Pneumonia Classification")
st.write(
    "Upload up to 10 chest X-ray images. ResNet18, KNN, and SVM scores are "
    "combined into one ensemble prediction for each image."
)
st.warning(
    "Research/demo use only. This result is not a medical diagnosis and must "
    "not replace review by a qualified clinician."
)
with st.sidebar:
    st.header("About this model")
    st.write("Classes: Normal and Viral Pneumonia")
    st.write("Models: ResNet18, KNN, and SVM")
    st.write("Decision: equal-weight average of model probabilities")
    st.write(f"Maximum uploads: {MAX_UPLOADS}")
    st.write(f"Maximum file size: {MAX_FILE_SIZE_MB} MB per image")
    st.write(f"Inference device: {DEVICE}")
    st.info(
        "Use non-sensitive images only. This research/demo tool is not a "
        "medical diagnosis."
    )

try:
    model = load_model()
    knn_bundle, svm_bundle = load_classical_models()
except (
    AttributeError,
    EOFError,
    FileNotFoundError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    pickle.UnpicklingError,
    RuntimeError,
    ValueError,
) as exc:
    st.error(
        "The models could not be loaded. Verify that all model files are "
        f"present and compatible. Details: {exc}"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose up to 10 chest X-ray images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_file:
    if len(uploaded_file) > MAX_UPLOADS:
        st.error(f"Please select no more than {MAX_UPLOADS} images.")
        st.stop()

    for file in uploaded_file:
        st.divider()
        st.subheader(file.name)
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(
                f"This file is larger than {MAX_FILE_SIZE_MB} MB and was skipped."
            )
            continue

        try:
            image = Image.open(file).convert("RGB")
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"This file is not a valid image and was skipped: {exc}")
            continue

        st.image(image, caption="Uploaded chest X-ray", use_container_width=True)
        try:
            with st.spinner(f"Classifying {file.name}..."):
                result = classify_uploaded_image(
                    image, model, knn_bundle, svm_bundle
                )
        except (RuntimeError, ValueError) as exc:
            st.error(f"Could not classify {file.name}: {exc}")
            continue

        predicted_index = int(result["predicted_index"])
        ensemble_probabilities = result["ensemble"]
        predicted_label = LABELS[predicted_index]
        confidence = float(ensemble_probabilities[predicted_index])
        st.subheader(f"Ensemble prediction: {predicted_label}")
        st.metric("Confidence", f"{confidence:.1%}")
        if confidence < UNCERTAIN_CONFIDENCE:
            st.warning(
                "The ensemble confidence is low. Treat this result as uncertain "
                "and request qualified human review."
            )

        score_table = {
            LABELS[class_index]: {
                "ResNet18": float(result["resnet"][class_index]),
                "KNN": float(result["knn"][class_index]),
                "SVM": float(result["svm"][class_index]),
                "Ensemble": float(ensemble_probabilities[class_index]),
            }
            for class_index in LABELS
        }
        st.dataframe(
            {
                "Class": list(score_table),
                "ResNet18": [
                    score_table[label]["ResNet18"] for label in score_table
                ],
                "KNN": [score_table[label]["KNN"] for label in score_table],
                "SVM": [score_table[label]["SVM"] for label in score_table],
                "Ensemble": [
                    score_table[label]["Ensemble"] for label in score_table
                ],
            },
            hide_index=True,
            column_config={
                "ResNet18": st.column_config.NumberColumn(format="%.1%"),
                "KNN": st.column_config.NumberColumn(format="%.1%"),
                "SVM": st.column_config.NumberColumn(format="%.1%"),
                "Ensemble": st.column_config.NumberColumn(format="%.1%"),
            },
        )
        display_score_chart(result)
