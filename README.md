# 🚢 ShipSense AI

![ShipSense AI Banner](assets/Cargoship.jpg)


**ShipSense AI** is a comprehensive, end-to-end machine learning system designed to classify underwater vessels using passive sonar acoustic signals (WAV audio files). The system bridges the gap between digital signal processing and artificial intelligence to identify marine vehicles based on the sound signatures they produce underwater.

---

## 🎯 Project Overview

The project successfully identifies 5 distinct acoustic classes:
1. **CargoShip:** Large commercial cargo vessels.
2. **KaiYuan:** Specific naval vessel class.
3. **SpeedBoat:** High-speed small watercraft.
4. **UUV:** Unmanned Underwater Vehicles.
5. **Noise:** Ambient underwater background noise.

---

## ⚙️ Features & Architecture

### 1. Data Pipeline & Processing
Built a robust data preprocessing pipeline to handle raw acoustic data using `librosa`:
- **Resampling & Normalization:** Standardized all audio files to a uniform 22,050 Hz sampling rate to ensure consistency.
- **Segmentation:** Split continuous long recordings into manageable, fixed-length segments (approx. 3 seconds each) to maximize the training dataset and ensure real-time inference capabilities.

### 2. Feature Engineering
Employed two distinct feature extraction strategies to accommodate different families of machine learning models:
- **Classical Machine Learning Features (282 Dimensions):** Extracted a comprehensive suite of 282 statistical features (MFCCs, Deltas, Chroma, Spectral Contrast, Zero-Crossing Rate, RMS Energy).
- **Deep Learning Features (Log-Mel Spectrograms):** Transformed the 1D audio waveform into a 2D visual representation (128x128 Log-Mel Spectrogram), perfectly formatted for computer vision techniques.

### 3. Model Training paths
Trained and evaluated multiple models to find the most accurate classifier:
- **Classical ML Models:** Random Forest (RF), Support Vector Machine (SVM), K-Nearest Neighbors (KNN).
- **Deep Learning Models (PyTorch):** Custom Shallow CNN (built from scratch for fast inference) and ResNet-18 (Transfer Learning for robust classification).

### 4. Interactive Web Application
A production-ready Streamlit web application (`app.py`) providing:
- **Modern UI/UX:** A stunning dark-mode interface featuring custom CSS, glassmorphism elements, gradients, and a responsive layout.
- **Real-Time Inference:** Upload a `.wav` file segment and toggle between "Classical ML" and "Deep Learning" models for live comparison.
- **Rich Results:** Outputs confidence scores, full probability distributions, and dynamic visual indicators.

---

## 📁 Directory Structure

```text
PVR LAB P/
├── app.py                    # Streamlit web app (ShipSense AI UI)
├── requirements.txt          # Python Dependencies
├── assets/                   # Ship images for the UI
├── data/                     # Data (raw, processed, classical_ml models, DL models, tensors)
│   ├── classical_ml/         # Serialized ML models (*.pkl)
│   └── models/               # PyTorch deep learning weights (*.pth)
├── notebooks/                # Jupyter notebooks for EDA, processing, and training
└── README.md                 # This documentation
```

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.9+
- Pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/shipsense-ai.git
   cd "PVR LAB P"
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv pvr_env
   # On Windows:
   pvr_env\Scripts\activate
   # On Mac/Linux:
   source pvr_env/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   *The app will automatically open in your default web browser.*

---

## 🌐 Deployment & Dataset Uploads

Please refer to the `deployment_guide.md` file for step-by-step instructions on deploying the application to **Streamlit Cloud** via **GitHub**.

**Important Note on Datasets:** 
Due to size limits on GitHub and Streamlit Cloud, the raw dataset folders (`data/raw/` and `data/processed/`) are typically excluded via `.gitignore`. The application only requires the trained models and pre-computed features for inference. If you wish to upload or share the entire multi-gigabyte datasets, refer to the "Handling Large Dataset Folders" section in the `deployment_guide.md`.

---

## 🤝 Contributors
- Developed by **Raghvendra Goyal**
- Powered by **Soulware** & Logic by **Mathmind 2.0**
