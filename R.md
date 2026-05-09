# 🌊 ShipSense AI: Deep Underwater Vessel Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)

![ShipSense AI Banner](assets/Cargoship.jpg)

**ShipSense AI** is a state-of-the-art acoustic classification system designed to identify maritime vessel signatures from underwater hydrophone recordings. By transforming 1D acoustic signals into 2D Mel-spectrograms, the system leverages **Deep Residual Networks (ResNet-18)** and ensemble machine learning to achieve high-precision classification across 5 distinct maritime classes.

---

## 🚀 Key Features
- **Hybrid Modeling**: Comprehensive benchmarking between Deep Learning (ResNet-18, Custom CNN) and Classical ML (SVM, RF, KNN).
- **Acoustic Engineering**: Advanced preprocessing pipeline using `librosa` for Mel-spectrogram and MFCC extraction.
- **Robust Evaluation**: Full-spectrum metrics including Ablation studies, ROC curves, and detailed Error Analysis.
- **Interactive UI**: Real-time vessel inference application built with Streamlit.
- **Academic Standard**: IEEE-formatted research report and LaTeX source included.

---

## 🚢 Dataset & Requirements
The project is built on the **ShipsEar dataset** (Santos-Domínguez et al., 2016).
- **Classes**: CargoShip, KaiYuan, SpeedBoat, UUV, Ambient Noise.
- **Dataset Link**: [DRIVE_LINK_HERE] (Includes Raw and Processed segments).
- **Environment**: CUDA-enabled GPU recommended for training.

---

## 📂 Repository Structure
```text
.
├── data/
│   ├── classical_ml/          # Serialized KNN, SVM, RF, and Scaler models
│   ├── models/                # Saved ResNet-18 and Custom CNN weights (.pth)
│   ├── splits.json            # Train/Val/Test metadata mapping
│   └── tensors/               # Pre-computed feature tensors for fast loading
├── figures/                   # Multi-stage visualizations (EDA, Preprocessing, Results)
├── notebooks/                 # End-to-end experimental pipeline
│   ├── EDA.ipynb              # Exploratory Data Analysis & Visualizations
│   ├── Preprocessing.ipynb    # Signal cleaning & Segmentation logic
│   ├── FeatureExtraction.ipynb # Mel-spectrogram & MFCC generation
│   ├── Classical_ML.ipynb     # Scikit-learn model benchmarking
│   ├── Deep_Learning_CNN.ipynb # Neural network training & optimization
│   └── Final_Evaluation.ipynb # Master metrics & performance analysis
├── Reports/                   # High-fidelity Research Documentation (PDF & MD)
│   ├── PVR_Final_IEEE_Paper.pdf # Formal Academic Report
│   └── PVR_R1-R6.pdf          # Module-specific technical discussions
├── app.py                     # Streamlit Deployment Interface
├── requirements.txt           # Dependency Manifest
└── SUBMISSION.md              # Project Submission Requirement Map
```

---

## 🛠️ Installation & Setup

1. **Clone & Navigate**:
   ```bash
   git clone https://github.com/raghvendragoyal01/Shipsense-AI.git
   cd Shipsense-AI
   ```

2. **Environment Configuration**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch the UI**:
   ```bash
   streamlit run app.py
   ```

---

## 📈 Performance Summary
The system achieves exceptional discriminative power, though careful attention must be paid to the data leakage discussed in the reports.

| Model | Accuracy | Macro F1 | AUC-ROC |
| :--- | :--- | :--- | :--- |
| **ResNet-18** | **99.92%** | **0.9994** | **1.00** |
| **Random Forest** | 99.92% | 0.9994 | 1.00 |
| **Custom CNN** | 93.09% | 0.7332 | 0.92 |

> [!IMPORTANT]
> **Data Leakage Warning**: The 99%+ scores are a result of 50% segment overlap. For true acoustic generalization, refer to the "Overfitting Analysis" in `Reports/PVR_R5.pdf`.

---

## 🤝 Contributors & Credits
- **Lead Developer**: Raghvendra Goyal
- **Algorithm Design**: Soulware & Mathmind 2.0 Logic
- **Dataset**: Santos-Domínguez et al. (ShipsEar)

---
