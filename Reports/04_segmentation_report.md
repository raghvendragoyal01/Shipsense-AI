# Audio Segmentation Strategy Report

## Introduction
This document evaluates the audio segmentation strategy implemented during the data preprocessing phase of the underwater vessel noise classification project. By analyzing the `data/processed/segment_manifest.csv` and the generated `04_segment_counts.csv`, we can directly infer the exact mathematical approach used to slice the continuous audio recordings into discrete samples, and subsequently evaluate its impact on dataset statistics.

## 1. Segment Length Justification
Based on the metadata embedded in the preprocessed files and the explicit output of the data engineering pipeline, a fixed **segment length of 3.0 seconds** (equivalent to 66,150 frames at 22,050 Hz) was utilized across the entire dataset. 

In the domain of passive sonar and underwater acoustics, this 3.0-second window is an optimal choice. It strikes a crucial balance based on the wide-sense stationarity assumption. A 3-second window is sufficiently long to capture at least one full mechanical cycle of slow-turning, low-RPM marine diesel engines (typical in the CargoShip class), ensuring that fundamental frequencies and their harmonics are fully represented in the resulting spectrograms. Conversely, it is short enough that the acoustic signature of a highly dynamic source, such as a rapidly maneuvering SpeedBoat, remains relatively stationary within the temporal window, preventing spectral smearing.

## 2. Overlap Strategy
By cross-referencing the original dataset durations with the final segment counts, we can explicitly infer the overlap strategy (hop length) employed during segmentation. 

The original dataset contained 20 CargoShip recordings with an average duration of approximately 300.94 seconds. If a zero-overlap, contiguous sliding window of 3.0 seconds were applied, we would expect approximately `(20 * 300.94) / 3 = 2,006` segments. 

However, the final `04_segment_counts.csv` reveals exactly **3,980** segments for the CargoShip class. This mathematically dictates that a **50% overlap (1.5-second hop length)** was utilized. A 50% overlap effectively doubles the data yield from continuous recordings. For the remaining classes (KaiYuan, SpeedBoat, Uuv, noise), the original files were already approximately 3.0 seconds in length. Consequently, the sliding window extracted exactly 1 segment per file, yielding exactly 30 segments per class.

## 3. Impact on Class Imbalance
While the overlapping strategy maximized data yield for long files, it has inadvertently introduced a severe, catastrophic class imbalance into the deep learning pipeline. The generated counts in `04_segment_counts.csv` illustrate this drastically:

| Class | Train Count | Val Count | Test Count | Total Segments |
| :--- | :--- | :--- | :--- | :--- |
| **CargoShip** | 2,786 | 597 | 597 | **3,980** |
| **KaiYuan** | 21 | 4 | 5 | **30** |
| **SpeedBoat** | 21 | 5 | 4 | **30** |
| **Uuv** | 21 | 4 | 5 | **30** |
| **noise** | 21 | 5 | 4 | **30** |

Prior to segmentation, the dataset was relatively balanced (20 CargoShip files vs. 30 for all other classes). Post-segmentation, the CargoShip class constitutes **over 97%** of the entire training dataset (2,786 out of 2,870 total training files). If a neural network is trained directly on this distribution, it will overwhelmingly predict CargoShip to minimize global categorical cross-entropy loss, entirely failing to generalize to naval vessels or UUVs.

## 4. Comparison with Literature
In contemporary acoustic literature concerning the ShipsEar dataset (Santos-Domínguez et al.), researchers typically segment continuous recordings into 3 to 5-second windows. The 3.0-second length aligns perfectly with state-of-the-art methodology for capturing marine tonal frequencies. Furthermore, a 50% overlap is a widely accepted technique in Environmental Sound Classification (ESC) to augment training data and capture transient events that might otherwise be split abruptly at the window boundaries. Therefore, the physical parameters of the windowing function are highly defensible and align with peer-reviewed methods.

## 5. Final Recommendations
While the physical parameters (3.0s length, 50% overlap) are optimal for capturing stationary vessel acoustics, the resulting statistical distribution is highly problematic. Based on the actual segment counts, the current strategy is **sub-optimal for immediate, unweighted model training**. 

To resolve the catastrophic 97% class imbalance, the following interventions are strongly recommended before initiating model training:
1.  **Dataloader Undersampling:** Implement a weighted random sampler within the PyTorch/TensorFlow dataloader to drastically undersample the CargoShip class during training, ensuring the model sees a balanced batch distribution of all 5 classes.
2.  **Class Weights:** Apply heavy class weights to the loss function (e.g., `weight = 3980/30 ≈ 132.6` for the minority classes) to penalize the model heavily for misclassifying KaiYuan, UUV, SpeedBoat, and noise signatures.
3.  **Dynamic Hop Length (Alternative):** In future preprocessing iterations, apply a 0% overlap (hop length = 3.0s) exclusively to the CargoShip class to halve its segment count to ~2000, while applying heavy augmentation (pitch shifting, time-masking, background noise mixing) to the minority classes to artificially inflate their counts. 

By applying weighted loss and undersampling during the training phase, the current `data/processed/segments` directory can still be utilized successfully without requiring a full re-segmentation of the raw audio.
