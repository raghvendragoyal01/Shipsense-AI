# ShipSense AI: Underwater Vessel Noise Classification

![ShipSense AI Banner](assets/Cargoship.jpg)

ShipSense AI is a deep learning-based classification system designed to identify underwater vessel noise signatures using the ShipsEar dataset. It leverages Mel-spectrograms and Residual Networks (ResNet-18) to achieve high-precision classification across five categories: CargoShip, KaiYuan, SpeedBoat, UUV, and Ambient Noise.

## 🚢 Dataset
The project utilizes the **ShipsEar dataset** (Santos-Domínguez et al., 2016). Due to large file sizes, the raw and processed audio data are hosted externally.
- **Raw/Processed Data**: [DRIVE_LINK_HERE]
- **Classes**: CargoShip, KaiYuan, SpeedBoat, UUV, Ambient Noise.

## 📂 Repository Structure
```text
.
├── data/
│   ├── classical_ml/          # Pre-trained KNN, SVM, RF models
│   ├── models/                # Saved ResNet-18 and Custom CNN weights
│   ├── splits.json            # Train/Val/Test split metadata
│   └── dataset_manifest.csv   # Global metadata for all audio files
├── figures/                   # Generated EDA, Preprocessing, and Result plots
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── Preprocessing.ipynb    # Audio standardization and segmentation
│   └── Final_Evaluation.ipynb # Model benchmarking and metrics
├── Reports/                   # PDF Research Reports (R1-R6) and IEEE Paper
├── 01_project_overview.md     # Technical project overview
├── 02_integrity_report.md     # Data quality and health report
├── 03_eda_discussion.md       # EDA insights and discussion
├── 04_segmentation_report.md  # Segmentation logic and imbalance analysis
├── 05_results_analysis.md     # Model performance and leakage analysis
├── 06_error_analysis.md       # Confusion patterns and literature review
├── 07_ieee_report.tex         # LaTeX source for the final research paper
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── SUBMISSION.md              # Requirement mapping for project submission
```

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ShipSense-AI.git
   cd ShipSense-AI
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Execution Order
To reproduce the project results, run the notebooks/scripts in the following sequence:
1. **`Reports/PVR_R1.pdf`**: Project Overview.
2. **`notebooks/EDA.ipynb`**: Review exploratory data analysis and class distributions.
3. **`notebooks/Preprocessing.ipynb`**: Observe audio resampling and 3s segmentation logic.
4. **`notebooks/Final_Evaluation.ipynb`**: Execute to train/evaluate models and generate metrics.
5. **`app.py`**: Launch the Streamlit UI to test real-time inference:
   ```bash
   streamlit run app.py
   ```

## 📊 Results Summary
- **Top Accuracy**: 99.92% (ResNet-18, Random Forest, SVM)
- **Macro-F1**: 0.9994
- **Note**: High performance is partially attributed to data leakage from overlapping segments; see `05_results_analysis.md` for a full deep dive.

## 📜 Citation
If you use this work, please cite the original ShipsEar paper:
> Santos-Domínguez, D., Torres-Guijarro, S., Cardenal-López, A., & Pena-Gimenez, A. (2016). ShipsEar: An underwater vessel noise database. Applied Acoustics.

---

## 🤝 Contributors
- Developed by **Raghvendra Goyal**
- Powered by **Soulware** & Logic by **Mathmind 2.0**
