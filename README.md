# EmotionAI — Face Detection & Emotion Recognition System

A professional-grade real-time emotion recognition pipeline built for a university final project.
Combines a custom CNN and MobileNetV2 transfer learning model with live face detection,
Grad-CAM explainability, persistent face identity, and a full-featured web dashboard.

---

## Features

- **Two trained models** — Custom 3-block CNN and fine-tuned MobileNetV2
- **Real-time webcam pipeline** — MediaPipe face detection with OpenCV Haar Cascade fallback
- **Temporal smoothing** — 5-frame weighted average prevents flickering labels
- **Grad-CAM heatmaps** — visualise what each model attends to per emotion
- **Face identity registry** — SQLite-backed persistent face recognition (dlib / face_recognition)
- **Live dashboard** — Flask + SocketIO with Chart.js timeline and donut charts
- **Session export** — PDF (ReportLab) and CSV session reports
- **Registration UI** — slide-in panel to name unknown faces during a session

---

## Project Structure

```
emotion_ai/
├── data/                        # Place fer2013.csv here
├── models/
│   ├── train_custom_cnn.py      # Train CNN from scratch
│   ├── train_transfer.py        # Train MobileNetV2 transfer learning
│   ├── evaluate.py              # Comparison + metrics
│   └── saved/                   # .keras model files + evaluation PNGs
├── core/
│   ├── detector.py              # Face detection (MediaPipe + OpenCV fallback)
│   ├── predictor.py             # Emotion inference pipeline + annotation
│   ├── gradcam.py               # Grad-CAM heatmap generation
│   ├── session_tracker.py       # Per-frame emotion history
│   └── face_registry.py         # SQLite face identity registry
├── web/
│   ├── app.py                   # Flask + SocketIO server
│   ├── templates/dashboard.html # Main UI (dark theme)
│   └── static/
│       ├── style.css
│       └── dashboard.js
├── utils/
│   ├── dataset_loader.py        # FER2013 CSV loader + augmentation
│   ├── report_generator.py      # PDF session reports
│   └── metrics_plot.py          # Training curves, confusion matrix, bar charts
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
cd emotion_ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `dlib` (required by `face_recognition`) needs cmake and a C++ compiler.
> On macOS: `brew install cmake`. On Ubuntu: `sudo apt install cmake build-essential`.

### 2. Download the dataset

Download **FER2013** from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013

Place `fer2013.csv` in the `data/` directory:
```
emotion_ai/data/fer2013.csv
```

---

## Training

### Custom CNN
```bash
python models/train_custom_cnn.py
```
Saves to `models/saved/custom_cnn.keras`. Runs ~50–70 epochs before early stopping.

### MobileNetV2 (two-phase fine-tuning)
```bash
python models/train_transfer.py
```
Saves to `models/saved/mobilenet_finetuned.keras`. Phase 1: 20 epochs, Phase 2: 30 epochs.

---

## Evaluation

```bash
python models/evaluate.py
```

Produces:
- Console classification report (accuracy, precision, recall, F1 per class)
- `models/saved/cm_custom_cnn.png` — confusion matrix
- `models/saved/cm_mobilenet.png` — confusion matrix
- `models/saved/metrics_*.png` — per-class metric bar charts
- `models/saved/evaluation_results.json` — machine-readable summary

---

## Running the Dashboard

```bash
python web/app.py
```

Open **http://localhost:5000** in your browser.

**What you'll see:**
- Left: Live webcam feed with annotated bounding boxes, emotion badges, confidence bars
- Top-right: Scrolling emotion timeline chart (last 60 s)
- Bottom-right: Emotion distribution donut chart
- Top bar: Dominant emotion, session duration, total faces, average confidence
- Model toggle: switch between Custom CNN and MobileNetV2 live
- 🔥 Grad-CAM button: overlay heatmap showing model attention regions
- 👥 FAB button: view / delete registered faces
- Export buttons: download CSV or PDF session report

---

## Face Registration

When the system detects an unknown face for 3+ seconds, a slide-in panel appears
with a zoomed crop of the face and a name input field.

- **Remember Me** — saves the face + name to the SQLite registry (`data/faces.db`)
- **Skip** — ignores this face for the rest of the session

Registered faces are recognised instantly in subsequent frames (and future sessions).

---

## Emotion Classes & Colours

| Emotion  | Class | Colour  |
|----------|-------|---------|
| Angry    | 0     | #FF4444 |
| Disgust  | 1     | #9B59B6 |
| Fear     | 2     | #E67E22 |
| Happy    | 3     | #F1C40F |
| Sad      | 4     | #3498DB |
| Surprise | 5     | #1ABC9C |
| Neutral  | 6     | #95A5A6 |

---

## Model Architecture

### Custom CNN
```
Input (48×48×1)
→ Conv64 → BN → ReLU → Conv64 → BN → ReLU → MaxPool → Dropout(0.25)
→ Conv128 → BN → ReLU → Conv128 → BN → ReLU → MaxPool → Dropout(0.25)
→ Conv256 → BN → ReLU → MaxPool → Dropout(0.25)
→ Flatten → Dense(512) → BN → ReLU → Dropout(0.5)
→ Dense(7, softmax)
```

### MobileNetV2
```
Input (96×96×3) — grayscale stacked to RGB
→ MobileNetV2 (ImageNet pretrained, phase 2: last 30 layers unfrozen)
→ GlobalAveragePooling2D
→ Dense(256) → BN → ReLU → Dropout(0.4)
→ Dense(7, softmax)
```

---

## Screenshots

> *(Add screenshots of the dashboard, Grad-CAM overlays, and evaluation charts here)*

---

## Notes

- If no webcam is found, place a sample video at `data/sample.mp4` — it will loop automatically.
- If no model file is found at startup, the dashboard still runs but emotion inference is disabled.
  Train at least one model before launching.
- Grad-CAM is automatically disabled when FPS drops below 15 to keep the stream smooth.
- face_recognition runs in a background thread and caches embeddings per face track to
  avoid recomputing every frame.

---

## Authors

- Elia Ghazal ([@eliaghazal](https://github.com/eliaghazal))
- George Khayat ([@georgekhayat](https://github.com/georgekhayat))
