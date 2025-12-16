import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection System")
st.write("Upload a brain MRI image to detect tumor presence.")

# Model download (runs only once)
MODEL_PATH = "best_vgg16_model_functional.h5"
MODEL_URL = "https://drive.google.com/drive/my-drive/best_vgg16_model_functional.h5"  # replace with your Google Drive file ID

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Detect Tumor"):
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)[0][0]
        confidence = float(pred * 100)

        if pred > 0.5:
            st.error(f"🧠 Tumor Detected — Confidence: {confidence:.2f}%")
        else:
            st.success(f"✔ No Tumor Detected — Confidence: {100 - confidence:.2f}%")

