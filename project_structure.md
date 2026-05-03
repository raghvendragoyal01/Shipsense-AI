# PVR LAB P — Full Project Structure

> **ShipSense AI** — Underwater Acoustic Vessel Classification System  
> Built with Python | Streamlit UI | TensorFlow + Classical ML

---

## 📁 Root Directory

```
PVR LAB P/
├── app.py                    # Streamlit web app (ShipSense AI UI)
├── requirements.txt          # Dependencies: streamlit, tensorflow-cpu, numpy, pandas, pillow
├── verify.py                 # Verification/test script
├── assets/                   # [EMPTY] — Ship images expected here (CargoShip.jpg, etc.)
├── data/                     # All data (raw → processed → features → models)
├── features/                 # [EMPTY] — Feature outputs placeholder
├── notebooks/                # All Jupyter notebooks (ML pipeline)
├── pvr_env/                  # Python virtual environment
└── report/                   # [EMPTY] — Report/output placeholder
```

---

## 📁 data/

```
data/
├── splits.json               # Train/Val/Test split definitions (209 KB)
├── raw/                      # Original WAV recordings per class
│   ├── CargoShip/            # Raw WAV files for CargoShip class
│   ├── KaiYuan/              # Raw WAV files for KaiYuan class
│   ├── SpeedBoat/            # Raw WAV files for SpeedBoat class
│   ├── Uuv/                  # Raw WAV files for Uuv class
│   └── noise/                # Raw WAV files for noise/background
├── processed/
│   ├── dataset_manifest.csv  # Manifest of full dataset (24 KB)
│   ├── resampled_manifest.csv# Manifest of resampled dataset (36 KB)
│   ├── segment_manifest.csv  # Manifest of all audio segments (558 KB)
│   ├── resampled/            # Resampled WAV files by class
│   │   ├── CargoShip/
│   │   ├── KaiYuan/
│   │   ├── SpeedBoat/
│   │   ├── Uuv/
│   │   └── noise/
│   └── segments/             # Segmented WAV files (~3-second chunks per source file)
│       └── [Thousands of .wav files: e.g. CargoShip_DATA0145.WAV_seg0.wav, ...]
├── classical_ml/             # Serialized Classical ML artifacts
│   ├── best_knn.pkl          # Trained K-Nearest Neighbors model (6.2 MB)
│   ├── best_rf.pkl           # Trained Random Forest model (687 KB)
│   ├── best_svm.pkl          # Trained SVM model (597 KB)
│   ├── scaler.pkl            # Feature scaler (StandardScaler) (7 KB)
│   ├── features_train.npy    # Extracted features — training set (6.2 MB)
│   ├── features_val.npy      # Extracted features — validation set (1.3 MB)
│   ├── features_test.npy     # Extracted features — test set (1.3 MB)
│   ├── y_train.npy           # Labels — training set (22 KB)
│   ├── y_val.npy             # Labels — validation set (5 KB)
│   └── y_test.npy            # Labels — test set (5 KB)
├── models/                   # Trained Deep Learning models
│   ├── Custom_CNN.pth        # Custom CNN (PyTorch) (508 KB)
│   └── ResNet18_Transfer.pth # ResNet18 Transfer Learning (PyTorch) (42.7 MB)
└── tensors/                  # NumPy tensor cache for DL training
    ├── X_features.npy        # Full feature tensor (256 MB)
    ├── y_labels.npy          # Full label array (32 KB)
    └── class_mapping.npy     # Class index ↔ name mapping
```

---

## 📁 notebooks/

```
notebooks/
├── EDA.ipynb                 # Exploratory Data Analysis (2.02 MB)
├── Preprocessing.ipynb       # Signal resampling & segmentation (2.32 MB)
├── FeatureExtraction.ipynb   # Feature engineering from audio (1.25 MB)
├── Classical_ML.ipynb        # KNN / RF / SVM training & evaluation (45 KB)
├── Deep_Learning_CNN.ipynb   # Custom CNN + ResNet18 training (1.81 MB)
├── Final_Evaluation.ipynb    # Cross-model comparison & metrics (96 KB)
├── Inference.ipynb           # Single-sample prediction demo (6 KB)
└── Model_Testing_History.csv # Manual tracking of model experiments (305 B)
```

---

## 🔗 Pipeline Flow (Connecting Points)

```
RAW DATA                    PROCESSING                  FEATURES / MODELS          APP
─────────                   ──────────                  ─────────────────          ───
data/raw/                   Preprocessing.ipynb  ──►   data/processed/            app.py
  [5 classes of .WAV]   │   (resample + segment)  │      resampled/               (Streamlit UI)
                         │                         │      segments/                    │
                         │   FeatureExtraction.ipynb─►  data/tensors/              loads model
                         │   (extract MFCC, etc.)  │      X_features.npy         from data/models/
                         │                         │      y_labels.npy
                         │   Classical_ML.ipynb    ─►  data/classical_ml/
                         │   (KNN, RF, SVM)             *.pkl models
                         │                              *.npy features & labels
                         │   Deep_Learning_CNN.ipynb─►  data/models/
                         │   (Custom CNN, ResNet18)      *.pth model weights
                         │
                         │   Final_Evaluation.ipynb  (compares all models)
                         │   Inference.ipynb          (single-sample demo)
                         │
                         └─► EDA.ipynb               (analysis of raw data)
```

---

## 🏷️ Classes (5 Vessel Types)

| Class | Description |
|-------|-------------|
| `CargoShip` | Large cargo vessel |
| `KaiYuan` | Specific vessel type |
| `SpeedBoat` | Fast small boat |
| `Uuv` | Unmanned Underwater Vehicle |
| `noise` | Ambient background noise |

---

## 📝 Key Files for UI Design

| File | Role | Notes |
|------|------|-------|
| `app.py` | Main UI entry point | Streamlit; expects `final_model.h5` in root |
| `data/splits.json` | Data splits | Train/Val/Test definitions |
| `data/classical_ml/*.pkl` | Classical models | KNN, RF, SVM + scaler |
| `data/models/*.pth` | Deep learning weights | Custom CNN, ResNet18 |
| `data/tensors/class_mapping.npy` | Label map | Class index ↔ name |
| `notebooks/Model_Testing_History.csv` | Experiment log | Model comparison history |
| `assets/` | **[EMPTY]** | Ship images needed here for UI |

> [!WARNING]
> `app.py` references `final_model.h5` which does **not exist** in the project. The saved models are `.pth` (PyTorch), but app.py uses `tf.keras` to load. This is a **critical mismatch** to resolve.

> [!NOTE]
> `assets/` folder is empty — the app references `assets/{ClassName}.jpg` for displaying ship images post-prediction. These images need to be created/added.
