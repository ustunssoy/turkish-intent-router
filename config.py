"""Proje konfigürasyonu."""

from pathlib import Path

# Dizinler
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

for d in [MODEL_DIR, DATA_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# Model
BERTURK_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
EMBEDDING_MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
MAX_LENGTH = 128
NUM_LABELS = 6

# Eğitim
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SEED = 42

# Niyet sınıfları
INTENT_LABELS = {
    0: "bilgi_sorgulama",
    1: "siparis_takibi",
    2: "sikayet",
    3: "iptal_iade",
    4: "teknik_destek",
    5: "genel_sohbet",
}

LABEL_TO_ID = {v: k for k, v in INTENT_LABELS.items()}

# RAG
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 3
