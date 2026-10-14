# CyclingSafety

**Multiclass classification of urban cycling safety events using smartphone IMU data, neural networks, and GPS-based risk maps.**

English · [Español](README_ES.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![License: MIT](https://img.shields.io/badge/License-MIT-green)

CyclingSafety explores how low-cost smartphone sensors and machine learning can help identify safety-critical events during urban cycling. Accelerometer and gyroscope signals are processed to classify events such as pothole impacts, evasive maneuvers, and emergency braking. Predictions are combined with GPS coordinates to visualize where events occurred and where road-severity indicators are concentrated.

## Model comparison and results

The notebook compares shallow MLP classifiers using hand-crafted statistical features with 1D convolutional neural networks (CNNs) trained on raw inertial windows. The CNN experiments include a manually designed architecture and Bayesian hyperparameter search with KerasTuner (AutoML-style tuning).

| Architecture | Classification setup | Route-level hold-out test accuracy | Macro-F1 |
|---|---|---:|---:|
| MLP — hand-crafted features | Heuristic, 3 classes | 98.59% | 0.9671 |
| MLP — hand-crafted features | Supervised, 4 classes | 97.91% | 0.8942 |
| 1D-CNN — manual architecture | Heuristic, 3 classes | 95.90% | 0.9027 |
| 1D-CNN — manual architecture | Supervised, 4 classes | 96.13% | 0.6936 |
| 1D-CNN — KerasTuner hyperparameter search | Heuristic, 3 classes | 98.30% | 0.9542 |
| **1D-CNN — KerasTuner hyperparameter search** | **Supervised, 4 classes** | **99.14%** | **0.9468** |

The best reported supervised result came from the KerasTuner-selected 1D-CNN: 99.14% accuracy and 0.9468 macro-F1 on 1,627 held-out windows. The notebook reports 100% recall for the three critical-event classes for this model, with class-level precision of 0.92 for potholes, 0.94 for evasive maneuvers, and 0.76 for emergency braking. The manually designed supervised CNN scored lower on macro-F1 (0.6936), illustrating why accuracy alone is not enough when normal riding greatly outnumbers critical events.

For supervised configurations, class-decision thresholds were tuned on the validation set before the final test-set evaluation. These are offline results from a route-level hold-out split, not evidence of real-world production or city-wide deployment.

## Dataset at a glance

- **2,760,216** inertial sensor samples after preprocessing, harmonized to **50 Hz**.
- **48 rides** in the exported Colab run: 37 natural urban rides and 11 controlled rides designed to capture labeled events.
- Approximately **15.3 hours** of recordings based on the resampled sample count.
- **335 manually annotated events:** 152 pothole impacts, 96 evasive maneuvers, and 87 emergency-braking events.
- Data captured with a Xiaomi POCO X6 using Sensor Logger; the device's accelerometer, gyroscope, and GPS data were exported for processing.

The public Bike&Safe dataset was assessed but excluded from the final model dataset because its acceleration scale was not sufficiently compatible with the project's own recordings.

## How it works

1. **Capture:** Record accelerometer, gyroscope, and GPS data with Sensor Logger during controlled and natural rides.
2. **Preprocess:** Combine recordings, resample inertial signals to 50 Hz, interpolate missing sensor values, propagate GPS coordinates for geospatial association, and correct/deduplicate manual event annotations.
3. **Segment and engineer features:** Divide the time series into sliding windows of 128 samples (2.56 seconds) with 50% overlap, then extract statistical features from inertial signals.
4. **Train and evaluate:** Train MLP classifiers on engineered features and 1D-CNNs on raw windows. Compare a manually designed CNN with KerasTuner-based Bayesian hyperparameter search across heuristic three-class and supervised four-class tasks.
5. **Map:** Join model predictions to GPS coordinates and create interactive event maps and road-severity heatmaps using Folium.

## Visual outputs

The repository includes interactive HTML maps generated from the analysis:

- [Classified CSE events](outputs/map_mlp_events.html)
- [Road-severity heatmap](outputs/map_severity_heatmap.html)
- [Combined map with toggleable layers](outputs/map_combined.html)

If GitHub does not render the interactive HTML directly, download the file and open it in a web browser.

## Implementation scope

The current repository documents and implements the smartphone-based sensing workflow, IMU preprocessing, MLP training/evaluation, and GPS-based maps. A four-layer IoT/Edge Intelligence architecture is also proposed:

| Layer | Scope |
|---|---|
| Perception | Smartphone IMU/GPS acquisition is implemented; an ESP32-based sensor node is a proposed extension. |
| Communication | MQTT-based transmission is part of the proposed architecture. |
| Backend | ThingsBoard integration is proposed, not presented here as a completed deployment. |
| Visualization | Folium-based event maps and heatmaps are generated by the current analysis. |

**Important:** on-device ESP32/TinyML inference and live MQTT/ThingsBoard integration are future implementation goals. The results reported above come from offline model training and validation on collected recordings.

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

The same notebook includes the manual and KerasTuner-selected 1D-CNN experiments and a final six-row comparison against both MLP baselines.

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
jupyter notebook notebooks/MLP_Cycling_Safety.ipynb
```

You can also use the [Google Colab notebook](https://drive.google.com/file/d/1XYW1Dx_ZvmpX7s5-YWOH7kgPcS5dwEK0/view?usp=sharing). In Colab, download the dataset and mount/copy it to the paths expected by the notebook before running the cells.

## Limitations and next steps

- Expand labeled recordings in varied real-world conditions and evaluate generalization across routes and cities.
- Improve the precision of minority event classes, especially pothole detection.
- Implement and benchmark the ESP32/TinyML edge-inference path.
- Integrate MQTT transport and a ThingsBoard dashboard for end-to-end telemetry monitoring.

## License

Distributed under the [MIT License](LICENSE).
