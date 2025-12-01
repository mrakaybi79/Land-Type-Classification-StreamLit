import streamlit as st
import tensorflow as tf
import numpy as np
import rasterio

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_v3_Advanced_CNN.h5")

model = load_model()

# ----------------------------
# Classes
# ----------------------------
classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    "AnnualCrop2", "AnnualCrop3", "UnknownClass"
]

# ----------------------------
# Read TIF image
# ----------------------------
def load_tif(path):
    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
        img = img.astype("float32") / 10000.0
    return img

# ----------------------------
# Streamlit App
# ----------------------------
st.title("EuroSAT Land Classification 🌍")
st.write("Upload a .tif file and get the predicted land use class.")

uploaded_file = st.file_uploader("Upload a TIF image", type=["tif", "tiff"])

if uploaded_file is not None:
    with rasterio.open(uploaded_file) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
        img = img.astype("float32") / 10000.0

    st.write("Image loaded successfully!")

    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch)
    idx = np.argmax(pred)
    conf = float(np.max(pred))

    st.subheader("Prediction")
    st.write(f"**Class:** {classes[idx]}")
    st.write(f"**Confidence:** {conf:.4f}")
