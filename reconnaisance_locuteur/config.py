import os
from pathlib import Path
import torch


# CONFIGURATION PROJET

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_ROOT = PROJECT_ROOT / "wav"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# Créer dossiers
for folder in ["selected_agents", "radio_simulated", "features", "models"]:
    (OUTPUT_ROOT / folder).mkdir(parents=True, exist_ok=True)


# CONFIGURATION GPU - OPTIMISÉ GTX 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
    
    # Optimisations pour 4GB VRAM
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
else:
    print("GPU non disponible")

# Paramètres optimisés pour 4GB VRAM
BATCH_SIZE = 16 if GPU_AVAILABLE else 8  
NUM_WORKERS = 2  # Pour GTX 1650
PIN_MEMORY = GPU_AVAILABLE


# PARAMÈTRES AUDIO


SAMPLE_RATE = 16000
DURATION = 3.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)


# SÉLECTION AGENTS


NUM_AGENTS = 50
MIN_SAMPLES_PER_AGENT = 200


# SIMULATION RADIO P25


LOW_CUTOFF = 300
HIGH_CUTOFF = 3000
SNR_DB = 15

# Compression dynamique
THRESHOLD_DB = -20
RATIO = 4
ATTACK_MS = 5
RELEASE_MS = 50


# EXTRACTION FEATURES


N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

FEATURES_LIST = [
    'spectral_centroid',
    'spectral_rolloff',
    'spectral_bandwidth',
    'zero_crossing_rate'
]

TOTAL_FEATURES = N_MFCC * 3 + len(FEATURES_LIST) * 3


# DATASET CONSTRUCTION


SAMPLES_PER_AGENT = 300
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ENTRAÎNEMENT MODÈLE


SVM_KERNEL = 'rbf'
SVM_C = 10
SVM_GAMMA = 'scale'
CV_FOLDS = 5

# Chemins modèles
MODEL_PATH = OUTPUT_ROOT / "models" / "svm_police_radio.pkl"
SCALER_PATH = OUTPUT_ROOT / "models" / "scaler.pkl"
LABEL_ENCODER_PATH = OUTPUT_ROOT / "models" / "label_encoder.pkl"


# AFFICHAGE



print("  CONFIGURATION PROJET POLICE RADIO")

print(f"Projet: {PROJECT_ROOT}")
print(f"Dataset: {DATASET_ROOT}")
print(f"{NUM_AGENTS} agents × {SAMPLES_PER_AGENT} échantillons")
print(f"{TOTAL_FEATURES} features audio")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")

