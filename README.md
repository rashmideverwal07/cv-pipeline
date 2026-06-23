# Real-Time Sign Language to Text Pipeline

A real-time computer vision system that translates **ASL sign language into text** live from a webcam — running at 20+ FPS on CPU with no GPU required.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)

---

## Features

- **24 ASL letters** detected in real time (A–Y, excluding J and Z which require motion)
- **Word builder** — hold any gesture for 1.2 seconds to add the letter
- **Autocorrect** — automatically fixes spelling when word is completed
- **Confidence filter** — only accepts predictions above confidence threshold
- **Subtitle overlay** — YouTube-style live caption on video feed
- **20+ FPS** on standard laptop CPU — no GPU needed
- **Custom gestures** — map any sign to custom words (SPACE, BACK, GREAT)

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| OpenCV | 4.8 | Webcam capture and drawing |
| MediaPipe | 0.10.35 | Hand landmark detection (21 points) |
| Scikit-learn | latest | Random Forest classifier |
| NumPy | latest | Array operations |
| pyspellchecker | latest | Word autocorrect |
| Streamlit | latest | Web dashboard |

---

## Model Performance

> Note: Model is trained per-user on their own hand data for best accuracy.
> Below results are from the author's trained model.

| Metric | Value |
|---|---|
| Training samples | ~9,500+ |
| Gesture classes | 22–27 |
| Test accuracy | 99–100% |
| Precision (avg) | 1.00 |
| Recall (avg) | 1.00 |
| Speed | 20+ FPS on CPU |

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Rashmi7905/cv-pipeline.git
cd cv-pipeline
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download MediaPipe model files
Run in PowerShell:
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

### 5. Collect your own gesture data
```bash
python collect_data.py
```
This opens your webcam and records hand landmarks for each gesture.
Each gesture is recorded for 15 seconds — move your hand slightly
during recording to capture variation.

### 6. Train the model
```bash
python train_model.py
```
Trains a Random Forest classifier on your collected data.
Typical accuracy: 99–100%.

### 7. Run the app
```bash
python main.py
```

### 8. View the dashboard
```bash
streamlit run app.py
```

---

## Controls

| Gesture | Action |
|---|---|
| Any ASL letter (hold 1.2s) | Add letter to word |
| open_palm (hold 1.2s) | Complete word (SPACE) |
| fist (hold 1.2s) | Delete last letter (BACK) |
| thumbs_up (hold 1.2s) | Custom word (GREAT) |
| Q key | Quit |

---

## Project Structure

cv-pipeline/

├── main.py              # Main real-time application

├── collect_data.py      # Webcam data collection script

├── train_model.py       # Random Forest training script

├── app.py               # Streamlit web dashboard

├── requirements.txt     # Python dependencies

└── README.md            # This file

These files are NOT in the repo (generated locally):
gesture_data.csv       → created by collect_data.py
gesture_model.pkl      → created by train_model.py
hand_landmarker.task   → downloaded in step 4

---

##  How It Works

Webcam frame

↓

OpenCV reads BGR frame → flip → convert to RGB

↓

MediaPipe Hand Landmarker

→ detects 21 landmarks on hand

→ each landmark has x, y, z coordinates

→ 21 × 3 = 63 features per frame

↓

Random Forest Classifier

→ takes 63 numbers as input

→ predicts which of 27 gestures

→ returns confidence probability

↓

Smoothing (deque of 12 frames)

→ majority vote prevents flickering

↓

Hold Timer (1.2 seconds)
→ prevents accidental additions

↓

Word Builder + Autocorrect

→ builds words and sentences

↓

Display on screen at 20+ FPS

---

## Why You Need to Collect Your Own Data

The gesture model (`gesture_model.pkl`) is trained on the author's hand
and webcam setup. Due to differences in:
- Hand size and shape
- Skin tone and lighting
- Camera angle and distance

The model works best when trained on YOUR own data.
The entire collection + training process takes about 10 minutes.

---

## Future Improvements

- [ ] Add J and Z using LSTM on landmark sequences
- [ ] Train on multiple people's hands for generalization  
- [ ] Two-hand gesture support
- [ ] Web deployment using Streamlit + WebRTC
- [ ] Larger vocabulary beyond ASL alphabet

---

## Author

**Rashmi** — B.Tech CSE, IGDTUW Delhi