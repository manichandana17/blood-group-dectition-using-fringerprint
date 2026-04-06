import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image

st.title("🧬 Blood Group Detection using Fingerprint")

# ✅ Safe model loading
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl file not found! Please upload it to GitHub.")
else:
    model = joblib.load(MODEL_PATH)

    # File upload
    file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "png", "jpeg", "bmp"])

    if file is not None:
        try:
            image = Image.open(file).convert('L')
            img = np.array(image)

            if img is None or img.size == 0:
                st.error("❌ Image not loaded properly")
            else:
                # ⚠️ IMPORTANT: match training size
                img = cv2.resize(img, (32, 32))   # or 128x128 based on your training
                img = cv2.GaussianBlur(img, (5,5), 0)

                features = img.flatten().reshape(1, -1)

                result = model.predict(features)

                st.image(file, caption="Uploaded Fingerprint", use_column_width=True)
                st.success(f"✅ Predicted Blood Group: {result[0]}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
