import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("model.pkl")

st.title("🧬 Blood Group Detection using Fingerprint")

# Allow bmp also
file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "png", "jpeg", "bmp"])

if file is not None:
    try:
        # Read image using PIL (works for BMP)
        image = Image.open(file).convert('L')
        img = np.array(image)

        # Safety check
        if img is None or img.size == 0:
            st.error("❌ Image not loaded properly")
        else:
            # Preprocessing
            img = cv2.resize(img, (128, 128))
            img = cv2.GaussianBlur(img, (5,5), 0)

            # Feature extraction
            features = img.flatten().reshape(1, -1)

            # Prediction
            result = model.predict(features)

            # Show image
            st.image(file, caption="Uploaded Fingerprint", use_column_width=True)

            # Result
            st.success(f"✅ Predicted Blood Group: {result[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")