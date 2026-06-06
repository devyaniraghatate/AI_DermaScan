# 🧬 DermalScanAI

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![EfficientNetB0](https://img.shields.io/badge/Model-EfficientNetB0-success)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Project-orange)
![MIT License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

DermalScanAI is a deep learning-powered facial skin condition analysis system that identifies and classifies visible skin conditions from facial images. The application combines computer vision, transfer learning, and web deployment to deliver real-time predictions through an interactive Streamlit interface.

The system classifies facial images into four categories:

- Clear Skin
- Dark Spots
- Puffy Eyes
- Wrinkles

Developed using EfficientNetB0, TensorFlow, OpenCV, and Streamlit, this project demonstrates an end-to-end machine learning workflow, from data preparation and model training to deployment and inference.

---

## Key Features

- Automated face detection using OpenCV Haar Cascade Classifiers
- Skin condition classification using EfficientNetB0
- Transfer learning-based deep learning approach
- Confidence-based prediction scoring
- Real-time image analysis through Streamlit
- Annotated image visualization
- CSV-based prediction logging
- Exportable prediction results

---

## Model Performance

The model was trained using transfer learning with EfficientNetB0 and achieved the following results:

| Metric | Value |
|----------|----------|
| Training Accuracy | 98.01% |
| Training Loss | 0.1708 |
| Validation Accuracy | 100% |
| Validation Loss | 0.1151 |

These results demonstrate strong classification performance and effective learning on the prepared dataset.

---

## System Workflow

```text
Input Image
     │
     ▼
Face Detection
     │
     ▼
Face Cropping
     │
     ▼
Image Preprocessing
     │
     ▼
EfficientNetB0 Model
     │
     ▼
Prediction & Confidence Score
     │
     ▼
Visualization & Export
```

---

## Dataset & Preprocessing

The dataset consists of facial images categorized into four skin condition classes.

### Classes

- Clear Skin
- Dark Spots
- Puffy Eyes
- Wrinkles

### Preprocessing Pipeline

- Face extraction and cropping
- Image resizing to 224 × 224
- Image normalization
- Data augmentation
- EfficientNet preprocessing
- Dataset balancing

### Augmentation Techniques

- Horizontal flipping
- Rotation
- Scaling
- Geometric transformations

These preprocessing techniques improved model robustness and helped reduce overfitting.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/devyaniraghatate/AI_DermaScan.git

cd AI_DermaScan
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run dermal_app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

---

## Project Structure

```text
AI_DermaScan/
│
├── dermal_app.py
├── best_dermal_model.h5
├── AI_Facial_Skin_Aging_Detection.ipynb
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── predictions_log.csv
├── README.md
│
├── logs/
└── exports/
```

---

## Results

The application provides:

- Skin condition prediction
- Confidence score for each prediction
- Probability distribution across all classes
- Annotated facial image output
- CSV prediction reports
- Historical prediction logs

The complete workflow enables users to upload an image, receive model predictions, and export the results for further analysis.

---

## Future Improvements

- Integrate RetinaFace or MTCNN for improved face detection
- Expand dataset diversity for better generalization
- Add explainable AI techniques such as Grad-CAM
- Support real-time webcam-based analysis
- Deploy using Docker and cloud services
- Extend support for additional skin condition categories

---

## Author

**Devyani Raghatate**  
B.Tech Artificial Intelligence Engineering

🔗 GitHub: https://github.com/devyaniraghatate

---

## License

This project is licensed under the MIT License.

⭐ If you found this project useful, consider giving it a star.
