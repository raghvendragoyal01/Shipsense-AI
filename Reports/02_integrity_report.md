# Dataset Integrity Report

## 0. Dataset Overview (Raw Data)
The raw data directory (`data/raw`) contains **5469** `.wav` files with a total size of **4196.76 MB**.

| Class Directory | WAV Files | Size (MB) |
| --- | --- | --- |
| CargoShip | 20 | 908.1 |
| KaiYuan | 2047 | 1235.44 |
| noise | 300 | 181.06 |
| SpeedBoat | 200 | 120.71 |
| Uuv | 2902 | 1751.46 |

## 1. Total File Count
The dataset contains a total of **140** files.

## 2. Per-class File Count
| Class | Count |
| --- | --- |
| KaiYuan | 30 |
| SpeedBoat | 30 |
| noise | 30 |
| Uuv | 30 |
| CargoShip | 20 |

## 3. Duration Statistics
| Class | Mean Duration (s) | Std Dev (s) |
| --- | --- | --- |
| CargoShip | 300.94 | 0.0 |
| KaiYuan | 3.0 | 0.0 |
| SpeedBoat | 3.0 | 0.0 |
| Uuv | 3.0 | 0.0 |
| noise | 3.0 | 0.0 |

## 4. Sample Rate Consistency
 **Consistent**: All files share the same sample rate of 52734 Hz.

## 5. Corrupted or Empty Files
 No missing, empty, or corrupted files were found.

## 6. Final Data Quality Verdict
 **Ready for Training**: The dataset passes all integrity checks. No corrupted or empty files detected, and the sample rate is consistent across the dataset.