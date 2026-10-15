# `README.md` for Emosonic

Ithu unga project-ku README file. Ithula project description, setup, usage ellame irukum. Copy panni `README.md` file-a create pannunga.

#  Emosonic

**Emotion-Based Music Recommendation System using CNN**

A deep learning project that detects human emotion from facial images using a Convolutional Neural Network (CNN) trained on the FER2013 dataset, and recommends mood-matched songs based on the detected emotion.



##  Overview

Emosonic combines two things:

1. **Emotion Detection** — A CNN model classifies a face image into one of 7 emotions:
   - Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
2. **Music Recommendation** — The detected emotion is mapped to a music mood, and top songs are recommended from a music dataset.



##  Model Architecture

A Sequential CNN with the following blocks:

| Block | Layers |
|-------|--------|
| Block 1 | Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25) |
| Block 2 | Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25) |
| Block 3 | Conv2D(128) → BN → MaxPool → Dropout(0.25) |
| Head | Flatten → Dense(256) → BN → Dropout(0.5) → Dense(7, softmax) |

- **Input:** 48×48 grayscale images
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Data Augmentation:** Rotation, horizontal flip, width/height shift



## Dataset

- **FER2013** — Facial Expression Recognition dataset
  - Train: 28,709 images
  - Test: 7,178 images
  - 7 emotion classes
- **music.csv** — Music dataset with audio features (valence, energy, popularity, etc.)

### Expected folder structure

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── test/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── music.csv
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/emosonic.git
cd emosonic
```

### 2. Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train the model

```bash
python train.py
```

This will:
- Load and preprocess FER2013 data
- Train the CNN for up to 50 epochs
- Save the best model to `output/models/cnn_model.keras`
- Save training plots (accuracy & loss) to `output/plots/`

### Step 2 — Evaluate the model

```bash
python evaluate.py
```

This will:
- Load the trained model
- Generate a confusion matrix
- Print a classification report (precision, recall, f1-score)

### Step 3 — Predict emotion & get song recommendations

```bash
python predict.py path/to/face_image.jpg
```

This will:
- Detect the emotion from the given face image
- Display all emotion probabilities
- Print the top 5 recommended songs for the detected mood

### Step 4 (Optional) — Test recommendations only

```bash
python recommend.py




##  Project Structure

emosonic/
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── config.py                 # Paths, constants, emotion-mood mapping
├── data_loader.py            # Load & preprocess FER2013 dataset
├── model.py                  # CNN model definition
├── train.py                  # Train the CNN model
├── evaluate.py               # Confusion matrix & classification report
├── recommend.py              # Music loading & recommendation logic
├── predict.py                # Face image → emotion → songs
│
├── data/
│   ├── train/                # Training images (7 emotion folders)
│   ├── test/                 # Testing images (7 emotion folders)
│   └── music.csv             # Music metadata dataset
│
└── output/
    ├── models/               # Saved models (.keras)
    ├── plots/                # Training plots & confusion matrix
    └── predictions/          # Prediction outputs




## 🎵 Emotion → Mood Mapping

| Detected Emotion | Music Mood | Reasoning |
|------------------|------------|-----------|
| Angry            | Angry      | Intense, high-energy music |
| Disgust          | Angry      | Intense music |
| Fear             | Sad        | Dark, low-valence music |
| Happy            | Happy      | Upbeat, positive music |
| Neutral          | Calm       | Relaxed, low-energy music |
| Sad              | Sad        | Slow, melancholic music |
| Surprise         | Happy      | Upbeat, energetic music |

**Mood is derived from audio features:**
- `valence ≥ 0.5` and `energy ≥ 0.5` → Happy
- `valence ≥ 0.5` and `energy < 0.5` → Calm
- `valence < 0.5` and `energy ≥ 0.5` → Angry
- `valence < 0.5` and `energy < 0.5` → Sad



##  Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~60% |
| Best Val Accuracy | ~59.7% |
| Macro F1 | ~0.53 |
| Weighted F1 | ~0.57 |

**Per-class highlights:**
-  Happy — Best performing (F1 ≈ 0.82)
-  Surprise — Good (F1 ≈ 0.73)
-  Fear — Weak (F1 ≈ 0.23)
-  Disgust — Weak (F1 ≈ 0.38)



##  Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** OpenCV
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn, Plotly
- **ML Utilities:** scikit-learn


##  Notes

- FER2013 dataset must be downloaded and placed in `data/train/` and `data/test/` manually.
- The model uses **grayscale 48×48** input — the same as FER2013.
- For best results, use GPU (CUDA) during training.
- Adjust `EPOCHS`, `BATCH_SIZE` in `config.py` if needed.

### Future Enhancements
Improve emotion recognition accuracy.
Integrate real-time facial emotion detection.
Enhance music recommendations using the Spotify API.
Develop a user-friendly web application


