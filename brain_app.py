import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload a brain MRI image and the model will predict if a tumor is present.")

# Cache the model load using st.cache_resource (recommended for TF models)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_vgg16_model_functional.h5")
    return model

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Detect Tumor"):
        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        pred = model.predict(img_array, verbose=0)[0][0]
        confidence = float(pred * 100)

        # Display result
        if pred > 0.5:
            st.error(f"🧠 Tumor Detected — Confidence: {confidence:.2f}%")
        else:
            st.success(f"✔ No Tumor Detected — Confidence: {100 - confidence:.2f}%")
