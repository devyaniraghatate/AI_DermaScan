import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import pandas as pd
import os
from datetime import datetime

# ----------------------------------
# App Configuration
# ----------------------------------
APP_NAME = "🧠 AI Facial Skin Condition Analyzer"

MODEL_PATH = "efficientnetb0_facial_skin_final.h5"
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
CSV_LOG_PATH = "predictions_log.csv"

IMG_SIZE = 224
DISPLAY_SIZE = (512, 512)

MODEL_ACCURACY = 98.40
CONF_THRESHOLD = 0.60

CLASS_NAMES = ["Clear Skin", "Dark Spots", "Puffy Eyes", "Wrinkles"]

# ----------------------------------
# Streamlit Page Config
# ----------------------------------
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    page_icon="🧠"
)

# ----------------------------------
# Load Model & Face Detector
# ----------------------------------
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    return model, face_cascade

model, face_cascade = load_assets()

# ----------------------------------
# Face Preprocessing
# ----------------------------------
def preprocess_face(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face

# ----------------------------------
# Save Prediction to CSV
# ----------------------------------
def save_to_csv(image_name, label, confidence):
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image_Name": image_name,
        "Prediction": label,
        "Confidence (%)": round(confidence * 100, 2),
        "Model_Accuracy (%)": MODEL_ACCURACY
    }

    df = pd.DataFrame([record])

    if os.path.exists(CSV_LOG_PATH):
        df.to_csv(CSV_LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_LOG_PATH, index=False)

# ----------------------------------
# UI Header
# ----------------------------------
st.title(APP_NAME)
st.markdown(
    f"""
    **Classes:** Clear Skin | Dark Spots | Puffy Eyes | Wrinkles  
    **Model Accuracy:** **{MODEL_ACCURACY}%**
    """
)

# ----------------------------------
# Image Upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a clear face image",
    type=["jpg", "jpeg", "png", "webp"]
)

# ----------------------------------
# Image Processing
# ----------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(DISPLAY_SIZE)

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=6,
        minSize=(120, 120)
    )

    col1, col2 = st.columns(2)

    # -------- Original Image --------
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    if len(faces) == 0:
        st.error("❌ No face detected. Please upload a clear face image.")
    else:
        for (x, y, w, h) in faces:
            pad = int(0.12 * w)
            face_roi = img_bgr[
                max(0, y + pad):min(y + h - pad, img_bgr.shape[0]),
                max(0, x + pad):min(x + w - pad, img_bgr.shape[1])
            ]

            processed_face = preprocess_face(face_roi)

            preds = model.predict(processed_face, verbose=0)[0]
            confidence = float(np.max(preds))
            class_id = int(np.argmax(preds))

            label = "Uncertain" if confidence < CONF_THRESHOLD else CLASS_NAMES[class_id]

            # Draw bounding box
            cv2.rectangle(
                img_rgb,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img_rgb,
                f"{label} ({confidence*100:.2f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            save_to_csv(uploaded_file.name, label, confidence)

        # -------- Prediction Result --------
        with col2:
            st.subheader("✅ Predicted Result")
            st.image(img_rgb, use_container_width=True)

            st.markdown(
                f"""
                <div style="text-align:center; font-size:18px; font-weight:600;">
                    📈 Model Accuracy:
                    <span style="color:green;">{MODEL_ACCURACY}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------- Probabilities --------
        st.subheader("📊 Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.progress(float(preds[i]))
            st.write(f"{class_name}: **{preds[i]*100:.2f}%**")

        st.success("✅ Prediction saved to CSV successfully!")

        # ----------------------------------
        # CSV Download Button
        # ----------------------------------
        if os.path.exists(CSV_LOG_PATH):
            with open(CSV_LOG_PATH, "rb") as file:
                st.download_button(
                    label="⬇️ Download Prediction Log (CSV)",
                    data=file,
                    file_name="facial_skin_predictions.csv",
                    mime="text/csv"
                )
