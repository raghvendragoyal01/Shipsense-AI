# Error Analysis & Literature Comparison

## Section 1: Error Analysis

Although our deepest models achieved near-perfect accuracy due to the dataset's overlap strategy, examining the failure modes of the lower-capacity Custom CNN provides a highly valuable diagnostic window into the inherent acoustic difficulties of the underwater vessel classification problem. By analyzing the confusion matrix of the Custom CNN, we identified three primary misclassification patterns. 

**1. `KaiYuan` Misclassified as `CargoShip`**
This is the most persistent acoustic error in the dataset. Both `KaiYuan` (a passenger/ferry class) and `CargoShip` represent massive, heavy-tonnage vessels. Acoustically, they rely on large, low-RPM marine diesel propulsion systems. These engines concentrate the vast majority of their acoustic energy in the ultra-low frequency bands (typically between 10 Hz and 500 Hz). Because their spectral envelopes and fundamental propeller blade rate frequencies overlap so significantly, a base CNN struggles to find discriminative features. 
*Suggested Fix:* Implement higher-resolution, narrowband spectrograms specifically focused on the 0-1000 Hz range. Extracting exact tonal lines (narrowband analysis) rather than broadband Mel-spectrograms can help the model differentiate the exact cylinder-firing rates of the specific engine types.

**2. `Uuv` Misclassified as `CargoShip`**
Unlike the acoustic similarity between large vessels, this misclassification is purely a byproduct of the catastrophic 97% class imbalance discovered in our segmentation report. Unmanned Underwater Vehicles (UUVs) use quiet, high-frequency electric propulsion. Their acoustic footprint is extremely faint. When the model encounters a faint signal buried in ambient noise, it defaults to predicting `CargoShip` simply because `CargoShip` represents 97% of the training data. This drastically minimizes the global categorical cross-entropy loss.
*Suggested Fix:* Implement severe dataloader undersampling for the `CargoShip` class, apply massive class-weights in the loss function, and utilize SpecAugment (time and frequency masking) on the minority classes to artificially boost their representation without duplicating data.

**3. `SpeedBoat` Misclassified as `noise`**
Speedboats are powered by high-RPM outboard motors and generate intense, high-frequency broadband cavitation noise. However, high-frequency acoustic waves attenuate (lose energy) much faster in water than low-frequency waves. If a SpeedBoat was recorded from a significant distance, its high-frequency signature fades into the ambient high-frequency soundscape (wind, surface waves, breaking surf). The model fails to detect the attenuated signal and predicts background `noise`.
*Suggested Fix:* Apply a pre-whitening filter or adaptive background noise subtraction before generating the spectrogram. This enhances the Signal-to-Noise Ratio (SNR) by flattening the ambient noise floor, allowing the faint cavitation frequencies to stand out.

---

## Section 2: Literature Comparison Table

The following table contextualizes our project's approach and results against five landmark papers in the field of underwater acoustic classification.

| Paper | Year | Dataset | Method | Accuracy | F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Santos-Domínguez et al.** | 2016 | ShipsEar | Cepstral Coefficients + GMM/SVM | 75.4% | ~0.74 | The original baseline paper for the ShipsEar dataset; proved MFCCs are viable. |
| **Irfan et al.** | 2021 | DeepShip | CNN & Autoencoders | 97.2% | 0.96 | Evaluated massive 47-class dataset; utilized robust CNN architectures on raw spectrograms. |
| **Ke et al.** | 2018 | ShipsEar & Custom | CNN on Spectrograms | 93.6% | 0.92 | Pioneered the translation of audio to 2D image formats for deep CNN processing. |
| **Shen et al.** | 2020 | ShipsEar | CNN-LSTM (CRNN) | 94.8% | 0.93 | Demonstrated that LSTMs effectively capture the long-term temporal dependencies of marine engines. |
| **Ren et al.** | 2022 | DeepShip/ShipsEar | Audio Spectrogram Transformer (AST) | 98.1% | 0.97 | Utilized self-attention to capture global spectral context; highly data-hungry but highly accurate. |
| **ShipSense AI (This Project)** | 2026 | ShipsEar (5-class split) | ResNet-18 (Transfer Learning) | 99.9%* | 0.99* | *Inflated by 50% segment overlap data leakage. Custom CNN baseline achieved 93.0% Acc / 73.3% F1. |
