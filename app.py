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
    figure, axis = plt.subplots(figsize=(6, 2.4))
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
    axis.set_ylabel("Score")
    axis.set_title("Model confidence comparison")
    axis.legend(ncols=2, fontsize="small")
    axis.grid(axis="y", alpha=0.25)
    st.pyplot(figure)
    plt.close(figure)


st.set_page_config(
    page_title="Pneumonia Classification",
    page_icon="🩻",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding: 1rem 1rem 2rem;
    }
    h1 {
        font-size: clamp(1.5rem, 5vw, 2.2rem) !important;
        margin-bottom: 0.25rem !important;
    }
    h2, h3 {
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
    }
    [data-testid="stMetric"] {
        padding: 0.35rem 0.5rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 0.9rem;
        white-space: nowrap;
    }
    [data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }
    .prediction-badge {
        border-radius: 0.45rem;
        color: white;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0.25rem 0 0.5rem;
        padding: 0.3rem 0.35rem;
        text-align: center;
    }
    .normal-badge {
        background: #198754;
    }
    .pneumonia-badge {
        background: #dc3545;
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 0.65rem 0.55rem 1.25rem;
        }
        [data-testid="stImage"] img {
            max-height: 150px;
        }
    }
    @media (min-width: 641px) and (max-width: 1000px) {
        .block-container {
            padding: 0.7rem 0.65rem 1.5rem;
        }
        [data-testid="stImage"] img {
            max-height: 150px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Chest X-ray Classification")
st.caption(
    "Upload up to 10 images. ResNet18, KNN, and SVM are combined for each result."
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

    for row_start in range(0, len(uploaded_file), 2):
        result_columns = st.columns(2, gap="small")
        for column, file in zip(
            result_columns, uploaded_file[row_start : row_start + 2]
        ):
            with column:
                with st.container(border=True):
                    st.caption(file.name)
                    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                        st.error(
                            f"Over {MAX_FILE_SIZE_MB} MB. Skipped.",
                            icon="⚠️",
                        )
                        continue

                    try:
                        image = Image.open(file).convert("RGB")
                        image.load()
                    except (UnidentifiedImageError, OSError) as exc:
                        st.error(f"Invalid image: {exc}", icon="⚠️")
                        continue

                    try:
                        with st.spinner("Classifying..."):
                            result = classify_uploaded_image(
                                image, model, knn_bundle, svm_bundle
                            )
                    except (RuntimeError, ValueError) as exc:
                        st.error(f"Could not classify: {exc}", icon="⚠️")
                        continue

                    predicted_index = int(result["predicted_index"])
                    ensemble_probabilities = result["ensemble"]
                    predicted_label = LABELS[predicted_index]
                    confidence = float(ensemble_probabilities[predicted_index])
                    badge_class = (
                        "normal-badge"
                        if predicted_index == 0
                        else "pneumonia-badge"
                    )
                    summary_column, image_column = st.columns([1, 1], gap="small")
                    with summary_column:
                        st.markdown(
                            f'<div class="prediction-badge {badge_class}">'
                            f"{predicted_label}</div>",
                            unsafe_allow_html=True,
                        )
                        st.metric("Confidence", f"{confidence:.1%}")
                        if confidence < UNCERTAIN_CONFIDENCE:
                            st.warning(
                                "Low confidence — human review recommended."
                            )
                    with image_column:
                        st.image(image, caption="Uploaded image", width=150)

                    score_table = {
                        LABELS[class_index]: {
                            "ResNet18": round(
                                float(result["resnet"][class_index]), 6
                            ),
                            "KNN": round(
                                float(result["knn"][class_index]), 6
                            ),
                            "SVM": round(
                                float(result["svm"][class_index]), 6
                            ),
                            "Ensemble": round(
                                float(ensemble_probabilities[class_index]), 6
                            ),
                        }
                        for class_index in LABELS
                    }
                    st.dataframe(
                        {
                            "Class": list(score_table),
                            "ResNet": [
                                score_table[label]["ResNet18"]
                                for label in score_table
                            ],
                            "KNN": [
                                score_table[label]["KNN"] for label in score_table
                            ],
                            "SVM": [
                                score_table[label]["SVM"] for label in score_table
                            ],
                            "Final": [
                                score_table[label]["Ensemble"]
                                for label in score_table
                            ],
                        },
                        hide_index=True,
                        height=115,
                        column_config={
                            "ResNet": st.column_config.NumberColumn(
                                format="%.2%"
                            ),
                            "KNN": st.column_config.NumberColumn(format="%.2%"),
                            "SVM": st.column_config.NumberColumn(format="%.2%"),
                            "Final": st.column_config.NumberColumn(
                                format="%.2%"
                            ),
                        },
                    )
                    with st.expander("Confidence chart"):
                        display_score_chart(result)
