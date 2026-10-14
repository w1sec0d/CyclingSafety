# CyclingSafety

**Multiclass classification of urban cycling safety events using smartphone IMU data, neural networks, and GPS-based risk maps.**

English · [Español](README_ES.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![License: MIT](https://img.shields.io/badge/License-MIT-green)

CyclingSafety explores how low-cost smartphone sensors and machine learning can help identify safety-critical events during urban cycling. Accelerometer and gyroscope signals are processed to classify events such as pothole impacts, evasive maneuvers, and emergency braking. Predictions are combined with GPS coordinates to visualize where events occurred and where road-severity indicators are concentrated.

## Preliminary model comparison

The notebook compares feature-engineered MLP baselines with 1D convolutional neural networks (CNNs) trained on raw IMU windows. CNN experiments include a manually designed architecture and Bayesian hyperparameter search with KerasTuner (AutoML-style tuning).

> **Important evaluation caveat:** these scores are outputs from the current notebook run, but the split is **not yet leakage-safe**. Although one output heading calls it a route-level hold-out, the code applies stratified `train_test_split` to individual windows after segmentation. The windows contain 128 samples with a stride of 64 (50% overlap), so highly similar windows from the same recording may appear in multiple splits. In addition, synthetic augmentations are generated from all annotated event windows before splitting, then all augmented copies are assigned to training; this can place derivatives of validation/test windows in the training set. Treat the metrics below as **exploratory and potentially optimistic**, not as robust estimates of performance on unseen rides.

| Architecture | Task | Current window-split accuracy | Macro-F1 |
|---|---|---:|---:|
| MLP — hand-crafted features | Heuristic, 3 classes | 98.59% | 0.9671 |
| MLP — hand-crafted features | Supervised, 4 classes | 97.91% | 0.8942 |
| 1D-CNN — manual architecture | Heuristic, 3 classes | 95.90% | 0.9027 |
| 1D-CNN — manual architecture | Supervised, 4 classes | 96.13% | 0.6936 |
| 1D-CNN — KerasTuner hyperparameter search | Heuristic, 3 classes | 98.30% | 0.9542 |
| **1D-CNN — KerasTuner hyperparameter search** | **Supervised, 4 classes** | **99.14%** | **0.9468** |

For supervised models, the table uses class-decision thresholds tuned on the validation set and then applied to the current test split. The tuned supervised CNN reported 100% recall for the three critical-event classes in this run, with precision of 0.92 for potholes, 0.94 for evasive maneuvers, and 0.76 for emergency braking. These results also require confirmation after correcting the split procedure.

**Recommended next step:** split by original ride/recording before creating windows, keep all windows from a recording in a single train/validation/test partition, generate augmentations from training windows only, tune hyperparameters and class thresholds on validation data, and evaluate once on untouched test rides.

## Dataset at a glance

- **2,760,216** inertial sensor samples in the supplied Colab run, resampled to **50 Hz**: 2,065,293 from natural recordings and 694,923 from controlled recordings.
- **48 CSV recordings/logs** in that run: 37 natural and 11 controlled.
- Approximately **15.3 hours** based on the resampled sample count.
- **335 manually annotated event timestamps:** 152 pothole impacts, 96 evasive maneuvers, and 87 emergency-braking events.
- Data captured with a Xiaomi POCO X6 using Sensor Logger; the device's accelerometer, gyroscope, and GPS data were exported for processing.

These totals reflect the supplied notebook export; reconcile them with earlier project summaries before using the figures in a formal report or publication.

The public Bike&Safe dataset was assessed but excluded from the final model dataset because its acceleration scale was not sufficiently compatible with the project's own recordings.

## How it works

1. **Capture:** Record accelerometer, gyroscope, and GPS data with Sensor Logger during controlled and natural rides.
2. **Preprocess:** Combine recordings, resample inertial signals to 50 Hz, interpolate missing sensor values, propagate GPS coordinates for geospatial association, and correct/deduplicate manual event annotations.
3. **Segment and engineer features:** Divide the time series into sliding windows of 128 samples (2.56 seconds) with 50% overlap, then extract statistical features from inertial signals.
4. **Train and compare:** Train feature-engineered MLPs and 1D-CNNs on raw windows. Compare a manual CNN architecture with KerasTuner-based Bayesian hyperparameter search across heuristic three-class and supervised four-class tasks. See the evaluation caveat above before interpreting the current scores.
5. **Map:** Join model predictions to GPS coordinates and create interactive event maps and road-severity heatmaps using Folium.
6. **Run at the edge:** Build an ESP32 prototype with an MPU-6050 IMU, NEO-6M GPS, and microSD. Quantize the supervised 1D-CNN to INT8, deploy it with TensorFlow Lite Micro, run local inference, and publish geotagged event telemetry to ThingsBoard over MQTT.

## Visual outputs

The repository includes interactive HTML maps generated from the analysis:

- [Classified CSE events](outputs/map_mlp_events.html)
- [Road-severity heatmap](outputs/map_severity_heatmap.html)
- [Combined map with toggleable layers](outputs/map_combined.html)

If GitHub does not render the interactive HTML directly, download the file and open it in a web browser.

## Implementation scope

The project combines the Python model-development workflow with an experimental embedded IoT prototype. The repository primarily contains the notebook-based preprocessing, MLP/CNN experiments, and Folium maps; the hardware and firmware work is described in the accompanying project report. The prototype firmware was developed and tested in Arduino IDE.

| Layer / component | Implementation scope |
|---|---|
| Sensor node | Built an ESP32 prototype with an MPU-6050 six-axis IMU, NEO-6M GPS, and microSD storage. |
| Edge inference | Post-training INT8 quantization of the supervised 1D-CNN and porting to TensorFlow Lite Micro for local inference on the ESP32. |
| Communication and backend | The firmware publishes geotagged event predictions using MQTT to a ThingsBoard dashboard. This was an experimental prototype, not a production deployment. |
| Visualization | Python/Folium event maps and severity heatmaps, plus a ThingsBoard dashboard for incoming node telemetry. |

The project report records a reduction in model artifact size from 2,394.95 KB to 227.20 KB (approximately 90.5%) after INT8 quantization. Its model benchmark changed from 99.26% to 99.20% accuracy and from 0.9531 to 0.9487 macro-F1; these are model-comparison figures in the report, not a guarantee of equivalent field performance. The report also notes that field classification performance on the ESP32 node was lower than with smartphone-based data, so cross-device calibration and validation remain important. The firmware and hardware documentation are not part of the repository's main notebook workflow at present. The preliminary notebook metrics above still carry the window-split evaluation caveat.

## Repository structure

```text
CyclingSafety/
├── notebooks/
│   ├── Cycling_Safety.ipynb       # MLP and 1D-CNN experiments
│   ├── Cycling_Safety.py          # Python export of the notebook
│   └── research/
├── utils/
│   └── combine_recordings.py
├── features/
├── models/
├── outputs/
└── README.md
```

The same notebook includes manual and KerasTuner-selected 1D-CNN experiments and a six-configuration comparison against both MLP baselines. The current split/augmentation sequence needs correction before these metrics can be treated as an independent benchmark.

## Getting started

### Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

### Install

```bash
git clone https://github.com/w1sec0d/CyclingSafety.git
cd CyclingSafety
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

### Get the dataset

The raw recordings are not stored in this repository because of their size. Download them from the [CyclingSafety dataset folder on Google Drive](https://drive.google.com/drive/folders/1-24wbZkLPmLIyj7BUMyjm0R-Bd0dZiIx?usp=sharing), then place them in the expected `data/raw/` folders.

### Preprocess recordings

```bash
python utils/combine_recordings.py --mode artificial
python utils/combine_recordings.py --mode natural
```

### Run the main notebook

```bash
jupyter notebook notebooks/Cycling_Safety.ipynb
```

You can also use the [Google Colab notebook](https://drive.google.com/file/d/1XYW1Dx_ZvmpX7s5-YWOH7kgPcS5dwEK0/view?usp=sharing). In Colab, download the dataset and mount/copy it to the paths expected by the notebook before running the cells.

## Limitations and next steps

- Redo train/validation/test splits at the original recording/ride level before windowing to prevent leakage from overlapping windows.
- Generate augmented samples from training windows only, never from validation/test event windows.
- Rerun all MLP/CNN comparisons and report class-wise precision, recall, macro-F1, and confusion matrices.
- Expand labeled recordings in varied real-world conditions and evaluate generalization across routes and cities.
- Improve the precision of minority event classes, especially pothole detection.
- Recalibrate and evaluate the ESP32 model with data captured from the embedded MPU-6050, since the report notes a performance gap versus smartphone-based inference.
- Add the ESP32 firmware, wiring diagram, and build instructions to the repository so the embedded prototype can be reproduced.
- Improve robustness of local microSD logging and connectivity handling for field rides.
- Integrate MQTT transport and a ThingsBoard dashboard for end-to-end telemetry monitoring.

## License

Distributed under the [MIT License](LICENSE).
