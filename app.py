from pathlib import Path

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


def classify_image(model: nn.Module, image: Image.Image) -> tuple[int, torch.Tensor]:
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    return int(probabilities.argmax().item()), probabilities


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
except (FileNotFoundError, RuntimeError) as exc:
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
        predicted_index, probabilities = classify_image(model, image)

    predicted_label = LABELS[predicted_index]
    st.subheader(f"Prediction: {predicted_label}")

    columns = st.columns(2)
    for column, class_index in zip(columns, LABELS):
        column.metric(
            LABELS[class_index],
            f"{float(probabilities[class_index]) * 100:.2f}%",
        )

    st.caption(f"Running on: {DEVICE}")
