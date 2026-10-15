import os

# ============ PATHS ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')
MUSIC_CSV = os.path.join(BASE_DIR, 'data', 'music.csv')

BASE_OUT = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_OUT, 'models')
PLOTS_DIR = os.path.join(BASE_OUT, 'plots')
PRED_DIR = os.path.join(BASE_OUT, 'predictions')

for d in [MODELS_DIR, PLOTS_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)

CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_model.keras')

# ============ CONSTANTS ============
IMG_SHAPE = 48
BATCH_SIZE = 64
NUM_CLASSES = 7

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

EMOTION_TO_MOOD = {
    'Angry':    'Angry',
    'Disgust':  'Angry',
    'Fear':     'Sad',
    'Happy':    'Happy',
    'Neutral':  'Calm',
    'Sad':      'Sad',
    'Surprise': 'Happy',
}