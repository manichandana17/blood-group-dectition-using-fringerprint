import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image

st.set_page_config(page_title="Blood Group Detection", layout="centered")

st.title("🧬 Blood Group Detection using Fingerprint")

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found. Run train.py first.")
else:
    model = joblib.load(MODEL_PATH)

    file = st.file_uploader("Upload Fingerprint Image", type=["jpg","png","jpeg","bmp"])

    if file is not None:
        try:
            image = Image.open(file).convert('L')
            img = np.array(image)

            if img is None or img.size == 0:
                st.error("❌ Invalid image")
            else:
                # SAME preprocessing as training
                img = cv2.resize(img, (64, 64))
                img = cv2.Canny(img, 100, 200)

                features = img.flatten().reshape(1, -1)
                features = features / 255.0

                result = model.predict(features)

                probs = model.predict_proba(features)
                confidence = np.max(probs) * 100

                st.image(file, caption="Uploaded Fingerprint", use_column_width=True)
                st.success(f"✅ Predicted Blood Group: {result[0]}")
                st.info(f"📊 Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"❌ Error: {e}")
