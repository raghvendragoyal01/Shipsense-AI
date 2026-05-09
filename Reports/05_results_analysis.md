# Model Evaluation and Results Analysis

## 1. Overall Performance Comparison
The final evaluation phase compared the efficacy of five distinct architectures: K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Random Forest (RF), a Custom Convolutional Neural Network (Custom CNN), and a Transfer-Learning based ResNet-18. 

The empirical results showcase an extraordinary, near-perfect performance tier and a lower-performing tier. Specifically, the classical machine learning models (KNN, SVM, RF) and the deep ResNet-18 all achieved a staggering **99.92% Accuracy** and **0.9994 Macro-F1 Score**. Conversely, the Custom CNN dropped significantly, recording a **93.09% Accuracy** and a much lower **73.32% Macro-F1 Score**. The stark contrast between the Custom CNN's Macro-F1 and Accuracy indicates that while it correctly classified the majority class, it failed entirely on the minority classes, dragging down the macro-averaged score.

## 2. Per-Class Performance & Acoustic Similarity
Based on the high-performing models, precision and recall reached 1.00 across all 5 classes (`CargoShip`, `KaiYuan`, `SpeedBoat`, `Uuv`, `noise`). However, by using the Custom CNN's struggles as a diagnostic tool, we can infer the inherent difficulties in the dataset.

Acoustically, `CargoShip` and `KaiYuan` vessels are the hardest to disentangle. Both rely on massive, low-RPM marine diesel propulsion systems that concentrate their acoustic energy in the ultra-low frequency bands (below 500 Hz). Their spectral envelopes overlap significantly. On the other hand, `SpeedBoat` signatures (characterized by high-RPM outboards and high-frequency propeller cavitation) are easily separable from the background ocean `noise`. `Uuv` (Unmanned Underwater Vehicles) utilize electric propulsion, generating very faint, high-frequency tonal lines that require high Signal-to-Noise Ratio (SNR) to avoid being classified as ambient noise. 

## 3. Best Performing Model Justification
While KNN, SVM, RF, and ResNet-18 all tied mathematically, **ResNet-18** stands as the structurally superior and most robust model. 

The classical models rely on flattened, 1D handcrafted feature vectors (such as MFCCs and spectral centroids). While effective, they discard the complex 2D temporal-frequency relationships. ResNet-18 processes the raw Mel-spectrograms as 2-dimensional acoustic images. By utilizing pre-trained ImageNet weights (Transfer Learning), ResNet-18 already possessed highly optimized convolutional filters capable of detecting edges, gradients, and textures. It simply repurposed these filters to detect acoustic harmonics, broadband transient bursts, and narrowband tonal lines, allowing it to achieve state-of-the-art results without requiring massive amounts of data to learn basic feature extractors.

## 4. Confusion Patterns
Because the top-tier models achieved 99.9% accuracy, their confusion matrices are virtually diagonal with negligible off-diagonal elements. 

However, addressing the Custom CNN's 73% Macro-F1 reveals the true confusion pattern: **Majority Class Dominance**. As documented in the segmentation report, the `CargoShip` class constitutes 97% of the dataset. When a model lacks the complex hierarchical capacity of ResNet-18 (or if class weights are not perfectly tuned), it suffers from predictive collapse. The model begins to classify almost all ambiguous signals as `CargoShip` to minimize global cross-entropy loss. In this scenario, `KaiYuan` and `Uuv` segments are routinely confused with `CargoShip`, leading to false positives for the majority class and catastrophic false negatives for the minority classes.

#### Confusion Matrices
![KNN Confusion Matrix](c:/Users/Raghvendra%20Goyal/OneDrive/Creative%20Cloud%20Files/Desktop/PVR%20LAB%20P/figures/results/cm_KNN.png)
![SVM Confusion Matrix](c:/Users/Raghvendra%20Goyal/OneDrive/Creative%20Cloud%20Files/Desktop/PVR%20LAB%20P/figures/results/cm_SVM.png)
![Random Forest Confusion Matrix](c:/Users/Raghvendra%20Goyal/OneDrive/Creative%20Cloud%20Files/Desktop/PVR%20LAB%20P/figures/results/cm_Random_Forest.png)

## 5. ROC Curve Interpretation
The Receiver Operating Characteristic (ROC) curves visually map the diagnostic ability of the classifiers as the discrimination threshold is varied. 

The combined micro-average ROC curve for ResNet-18, SVM, RF, and KNN displays a perfect "right angle," hugging the top-left corner of the plot with an **AUC-ROC of 1.000**. This implies perfect linear separability in the models' latent feature spaces; the models can maintain a True Positive Rate (TPR) of 1.0 without incurring any False Positives (FPR). The Custom CNN displays a slightly bowed curve (AUC ≈ 0.920), indicating that to correctly identify the minority classes, it must lower its confidence threshold, inadvertently triggering false positives.

#### ROC Curves
![Combined ROC Curves](c:/Users/Raghvendra%20Goyal/OneDrive/Creative%20Cloud%20Files/Desktop/PVR%20LAB%20P/figures/results/roc_curves.png)

## 6. Signs of Overfitting and Data Leakage
While a 99.92% accuracy is statistically impressive, it is highly anomalous in acoustic field recordings and is a massive **red flag for Overfitting via Data Leakage**.

As established in the `04_segmentation_report.md`, long 300-second continuous recordings were sliced with a **50% overlap**. If the Train/Test split was performed *after* segmentation (randomly shuffling segments rather than splitting by the original parent files), adjacent overlapping segments end up in both the training and testing sets. Because a 50% overlap means two segments share exactly 1.5 seconds of identical audio data, the neural network isn't learning to generalize underwater acoustics. Instead, it is perfectly memorizing the exact background noise, hydrophone static, and specific engine hum of the original 20 files, and recognizing that exact identical audio snippet in the test set. 

This leakage artificially inflates accuracy to 99.9%. To validate true generalization, future iterations must enforce a strict **Group-K-Fold** or **Parent-File Split**, ensuring that all segments originating from a specific physical recording are entirely isolated in either the training or testing set, but never both.
