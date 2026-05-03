# ShipSense AI - Project Summary

## Project Overview
**ShipSense AI** is a comprehensive, end-to-end machine learning system designed to classify underwater vessels using passive sonar acoustic signals (WAV audio files). The system bridges the gap between digital signal processing and artificial intelligence to identify marine vehicles based on the sound signatures they produce underwater.

The project successfully identifies 5 distinct acoustic classes:
1. **CargoShip:** Large commercial cargo vessels.
2. **KaiYuan:** Specific naval vessel class.
3. **SpeedBoat:** High-speed small watercraft.
4. **UUV:** Unmanned Underwater Vehicles.
5. **Noise:** Ambient underwater background noise.

---

## 1. Data Pipeline & Processing
We built a robust data preprocessing pipeline to handle raw acoustic data:
- **Resampling & Normalization:** Standardized all audio files to a uniform 22,050 Hz sampling rate to ensure consistency.
- **Segmentation:** Split continuous long recordings into manageable, fixed-length segments (approx. 3 seconds each) to maximize the training dataset and ensure real-time inference capabilities.
- **Structured Storage:** Organized the dataset logically into `raw`, `processed` (resampled and segmented), and `tensors` for efficient training.

## 2. Feature Engineering
We employed two distinct feature extraction strategies to accommodate different families of machine learning models:

### A. Classical Machine Learning Features (282 Dimensions)
We extracted a comprehensive suite of 282 statistical features from the audio signals to capture the temporal and spectral characteristics of the vessel sounds:
- **MFCCs (40) + Deltas (40) + Delta-Deltas (40):** Captures the timbral texture of the engine and propeller noise.
- **Chroma Features (12):** Captures the pitch class profile.
- **Spectral Contrast (7):** Distinguishes between peaks and valleys in the spectrum.
- **Zero-Crossing Rate (ZCR) & RMS Energy:** Captures the signal's rate of sign-changes and loudness.
*(All features were aggregated using their Mean and Standard Deviation to form a 1D vector).*

### B. Deep Learning Features (Log-Mel Spectrograms)
- Transformed the 1D audio waveform into a 2D visual representation (128x128 Log-Mel Spectrogram) mimicking how human ears perceive sound frequencies, perfectly formatted for computer vision techniques.

## 3. Model Architecture & Training
We trained and evaluated multiple models to find the most accurate classifier, establishing two distinct prediction paths:

- **Classical ML Models:** 
  - Random Forest (RF)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  *(All models were paired with a StandardScaler to normalize the 282 feature vectors).*

- **Deep Learning Models (PyTorch):**
  - **Custom Shallow CNN:** A lightweight convolutional neural network built from scratch (3 Convolutional blocks + Max Pooling + Fully Connected layers) for fast, efficient inference.
  - **ResNet-18 (Transfer Learning):** Leveraged a pre-trained ResNet architecture adapted to process the 3-channel (RGB) representations of our Mel Spectrograms, achieving highly robust classification.

## 4. Web Application (Streamlit UI)
To make the AI accessible, we built a fully functional, production-ready web application:
- **Modern UI/UX:** A stunning dark-mode interface featuring custom CSS, glassmorphism elements, gradients, and a responsive layout.
- **Real-Time Inference:** Users can upload a `.wav` file segment directly into the app.
- **Model Selection:** Users can seamlessly toggle between "Classical ML" and "Deep Learning" models for live comparison.
- **Rich Results:** The app performs live feature extraction, runs the prediction, and presents the output with confidence scores, full probability distributions across all 5 classes, and dynamic visual indicators (emojis, vessel images).
- **Educational Tabs:** The application includes dedicated sections detailing the model architectures, feature engineering stats, and the complete pipeline flow.

## 5. What We Accomplished
1. Successfully handled raw unstructured acoustic data.
2. Implemented complex signal processing using `librosa`.
3. Designed and trained both traditional ML and modern Deep Learning algorithms.
4. Serialized and optimized the models (`.pkl` and `.pth` files) for deployment.
5. Deployed the entire ecosystem into a beautiful, user-friendly interactive web application (`app.py`).
