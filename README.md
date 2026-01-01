# DermalScan: AI Facial Skin Condition & Aging Detection App

## Overview
DermalScan is a deep learning–based facial skin condition analysis system that detects and classifies visible skin-related features from facial images. The application identifies and categorizes facial skin conditions into **Clear Skin**, **Dark Spots**, **Puffy Eyes**, and **Wrinkles** using a pretrained **EfficientNetB0** model.

The system combines **computer vision (face detection)**, **transfer learning**, and a **Streamlit-based web interface** to deliver real-time predictions with confidence scores, annotated bounding boxes, and downloadable prediction logs.

---

## Key Capabilities
- Face detection using Haar Cascade classifiers
- EfficientNetB0-based deep learning classification
- Percentage-based prediction confidence
- Confidence thresholding to avoid uncertain predictions
- Annotated bounding boxes and labels
- CSV-based prediction logging with timestamps
- Web-based UI for easy image upload and analysis

---

## Model Performance
The model was trained using transfer learning with EfficientNetB0 and achieved the following results:

| Metric | Value |
|------|------|
| Training Accuracy | **98.01%** |
| Training Loss | **0.1708** |
| Validation Accuracy | **100%** |
| Validation Loss | **0.1151** |

> These results indicate strong generalization on the validation set, with stable convergence and low loss.

---

## Supported Classes
- Clear Skin  
- Dark Spots  
- Puffy Eyes  
- Wrinkles  

---

## System Workflow
1. User uploads a facial image via the Streamlit interface  
2. Image is resized for display and converted for processing  
3. Haar Cascade detects facial regions  
4. Detected face is tightly cropped to reduce background and texture bias  
5. Face is preprocessed using **EfficientNet-specific preprocessing**  
6. Model predicts class probabilities  
7. Predictions below confidence threshold are marked as **Uncertain**  
8. Results are displayed with bounding boxes and confidence scores  
9. Prediction details are logged into a CSV file  

---

## Face Preprocessing Strategy
To ensure compatibility with EfficientNetB0, the following preprocessing pipeline is applied:

- Resize face to **224 × 224**
- Convert BGR → RGB
- Apply `preprocess_input()` from EfficientNet
- Expand dimensions for batch inference

This ensures the inference pipeline exactly matches the training distribution.

---

## Confidence Thresholding
To improve prediction reliability, a confidence threshold is enforced:

- **Threshold:** `0.60`
- Predictions below this threshold are labeled as **Uncertain**
- Prevents misleading classifications on low-quality or ambiguous inputs

---

## CSV Logging
Each prediction is automatically logged into a CSV file with:

- Timestamp  
- Image name  
- Predicted class  
- Confidence percentage  
- Model accuracy reference  

This enables traceability, evaluation, and result auditing.

---

## User Interface (Streamlit)
The web interface provides:

- Image upload (JPG, PNG, JPEG, WEBP)
- Side-by-side original and predicted image views
- Bounding boxes with predicted labels
- Confidence visualization for each class
- Progress bars for probability distribution
- Real-time feedback and error handling

---

## Project Structure (Typical)
```
├── app.py
├── Dataset
├── AI_Facial_Skin_Aging_detection.ipynb
├── haarcascade_frontalface_default.xml
├── predictions_log.csv
├── requirements.txt
└── README.md
```


---

## Tech Stack

### Model & ML
- TensorFlow
- Keras
- EfficientNetB0

### Computer Vision
- OpenCV
- Haar Cascade Classifier
- NumPy

### Frontend
- Streamlit
- HTML (via Streamlit Markdown)

### Data Handling
- Pandas
- CSV logging

---

## Evaluation Highlights
- Accurate face localization
- High-confidence classification
- Low inference latency (≤ 5 seconds per image)
- Robust preprocessing aligned with training
- Reliable logging and export functionality

---

## Limitations
- Performance depends on face visibility and lighting
- Haar Cascade may fail on extreme poses or occlusions
- High validation accuracy may indicate dataset simplicity or limited diversity

---

## Future Improvements
- Replace Haar Cascade with deep face detectors (e.g., RetinaFace, MTCNN)
- Add multi-face support with independent logging
- Improve dataset diversity for better generalization
- Deploy using Docker or cloud-based inference
- Add Grad-CAM for explainable predictions

---

## Conclusion
DermalScan demonstrates a complete end-to-end AI pipeline for facial skin condition analysis, integrating deep learning, computer vision, and web deployment. The project emphasizes **correct preprocessing**, **confidence-aware predictions**, and **practical usability**, making it suitable for academic evaluation, internships, and real-world experimentation.
