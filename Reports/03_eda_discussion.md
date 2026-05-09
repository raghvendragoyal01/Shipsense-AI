# Exploratory Data Analysis (EDA) Technical Discussion

## Introduction
This document presents a comprehensive technical discussion based on the Exploratory Data Analysis (EDA) phase of the underwater vessel noise classification project. The observations are derived directly from the extracted figures, dataframes, and outputs of the executed `EDA.ipynb` notebook. The goal is to analyze the raw acoustic signatures and dataset statistics to inform subsequent feature engineering and model development strategies.

## 1. Waveform Characteristics
Analysis of the time-domain waveforms across the five target classes reveals distinct amplitude envelopes and temporal structures indicative of their respective physical acoustic sources. 

![Waveforms & Spectrograms](./figures/eda/waveform_2.png)

*   **CargoShip:** The waveforms exhibit a dense, continuous high-amplitude envelope, characteristic of large displacement vessels with constant RPM diesel propulsion. The temporal structure lacks sharp transients but maintains a massive baseline acoustic energy.
*   **KaiYuan:** As naval vessels, the KaiYuan class waveforms show a structured, rhythmic amplitude modulation. This periodicity aligns with specific propeller shaft rates and machinery cycles, distinctly different from civilian cargo vessels.
*   **SpeedBoat:** The high-speed watercraft waveforms are characterized by high-variance, transient spikes in the time domain. These rapid amplitude fluctuations correspond to aggressive maneuvering, hull slamming against the water surface, and high-RPM propeller cavitation bursts.
*   **Uuv (Unmanned Underwater Vehicle):** The submersibles display an exceptionally smooth, low-amplitude waveform. Lacking large mechanical propulsion or surface wave interaction, their time-domain signature is remarkably flat, exhibiting minor modulations strictly from internal electric thrusters.
*   **noise:** The ambient underwater noise waveforms lack any deterministic structure. They resemble Gaussian white or pink noise, exhibiting uniform variance without any prominent structural peaks, serving as the baseline acoustic floor.

## 2. Spectrogram Observations
The time-frequency representation via Log-Mel Spectrograms provides profound insights into the spectral distribution of energy for each class, which is vital for deep learning pattern recognition.

*   **CargoShip:** The spectrograms are dominated by intense energy bands concentrated heavily in the ultra-low frequency region. Several distinct, parallel narrowband horizontal lines (tonals) are visible, representing the fundamental and harmonic frequencies of the ship's massive engine and large-diameter propeller blades.
*   **KaiYuan:** These spectrograms display a complex interplay of mid-to-low frequency tonals. The harmonic spacing is notably tighter and potentially more complex than that of the CargoShip, indicating specialized multi-stage machinery inherent to naval designs.
*   **SpeedBoat:** The spectral energy for SpeedBoats extends significantly higher up the frequency axis, often dominating the mid-to-high frequency bands (1 kHz to 8 kHz). The spectrograms show broad, vertical energy smears rather than precise horizontal tonals, illustrating broad-spectrum propeller cavitation noise rather than internal engine hum.
*   **Uuv:** The spectrograms for UUVs are exceedingly sparse. Energy is highly localized into a few extremely faint, high-frequency continuous wave (CW) tonals. The lack of broadband cavitation noise is a defining characteristic of this stealthy class in the frequency domain.
*   **noise:** The ambient noise spectrograms display a uniform distribution of background energy with occasional, random broadband disruptions (e.g., distant wave action), clearly lacking any persistent, anthropogenic tonal structures.

## 3. Class Imbalance Analysis
An analysis of the dataset manifest and the class distribution bar charts generated during the EDA phase reveals a specific imbalance in the working subset. The total file count processed during the strict subset extraction is 140 files. The exact distribution is computed as follows:

![Class Distribution](./figures/eda/class_distribution_1.png)

*   **KaiYuan:** 30 files
*   **noise:** 30 files
*   **SpeedBoat:** 30 files
*   **Uuv:** 30 files
*   **CargoShip:** 20 files

While four out of the five classes are perfectly balanced at exactly 30 instances each, the **CargoShip** class is distinctly underrepresented, comprising only 20 instances. This 33% reduction relative to the majority classes presents a minor but significant statistical imbalance. If left unaddressed, trained models (especially data-hungry deep learning architectures) might develop a bias against predicting the CargoShip class due to its lower prior probability during the gradient update phases.

## 4. Signal-to-Noise Ratio (SNR) Observations
The calculation of the Average Power Spectral Density (PSD) using Welch's Method across all classes provides a direct mathematical visualization of the Signal-to-Noise Ratio (SNR) across the frequency spectrum. 

![Average PSD / SNR Profile](./figures/eda/plot_cell_5_3.png)

When comparing the PSD curves of the anthropogenic classes against the 'noise' baseline, distinct SNR profiles emerge. The CargoShip and KaiYuan classes demonstrate a massive positive SNR in the low-frequency bands, heavily exceeding the ambient noise floor by a large margin (frequently >15-20 dB). SpeedBoats exhibit a moderate but consistently positive SNR spanning the mid-to-high frequency ranges due to intense cavitation. Conversely, the UUV class demonstrates a critically low, near-zero SNR across most of the broadband spectrum. The UUV PSD curve almost overlaps entirely with the ambient noise curve, except at a few highly specific narrowband frequencies corresponding to their electric motors. This extremely low SNR indicates that UUV detection is highly sensitive to background conditions and requires highly focused feature extraction.

## 5. Key Takeaways for Feature Engineering
Based on the empirical acoustic observations from the EDA notebook, the following critical decisions must be integrated into the feature extraction and modeling pipeline:

1.  **Low-Frequency Emphasis:** Because massive vessels (CargoShip, KaiYuan) operate heavily in the low-frequency domain, while UUVs exhibit specific tonals, utilizing Mel-Frequency Cepstral Coefficients (MFCCs) is highly justified. The Mel scale's logarithmic spacing naturally emphasizes lower frequencies, capturing vital tonal data better than linear scales.
2.  **Addressing Broadband vs. Narrowband:** SpeedBoats rely on broadband cavitation, while UUVs and CargoShips rely on narrowband tonals. Therefore, extracting Spectral Contrast and Chroma features in addition to MFCCs will provide classical machine learning algorithms with the statistical tools needed to differentiate between sharp tonal peaks and flat broadband noise.
3.  **Handling Imbalance:** The underrepresentation of the CargoShip class (20 vs 30 files) dictates that global accuracy is an insufficient evaluation metric. Stratified k-fold cross-validation is mandatory during training. Model evaluation must prioritize the macro F1-score and carefully analyze confusion matrices to monitor the recall rate specifically for the CargoShip class.
4.  **Temporal Dynamics:** The transient, burst-like nature of SpeedBoat signals versus the continuous, steady nature of CargoShips dictates that temporal derivatives (Delta and Delta-Delta MFCCs) are essential. These dynamic features will capture the rate of change of the acoustic spectrum, effectively modeling the behavioral differences between a rapid surface maneuver and steady, constant cruising.
