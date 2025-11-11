"""
Constants for the Dilbert-o-Matic MLOps Pipeline
Centralized configuration to avoid hardcoding values across the codebase.
"""
import os
from pathlib import Path

# PROJECT PATHS
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_STORE = PROJECT_ROOT / "mlflow_store"
DVC_REMOTE = PROJECT_ROOT / "dvc_remote"

# Model paths
MODEL_V1_DIR = MODELS_DIR / "t5_jargon_v1"
MODEL_CHECKPOINTS_DIR = MODELS_DIR / "t5_jargon_v1_checkpoints"

# DATA PATHS
TRAINING_DATA_FILE = PROJECT_ROOT / "training_data.jsonl"
NEW_JARGON_CANDIDATES_FILE = PROJECT_ROOT / "new_jargon_candidates.txt"
KNOWN_JARGON_FILE = PROJECT_ROOT / "known_jargon.csv"

# MODEL CONFIG
BASE_MODEL_NAME = "google/flan-t5-small"
MODEL_MAX_LENGTH = 128
MODEL_MIN_LENGTH = 20

# Training parameters
TRAINING_BATCH_SIZE = 4
TRAINING_EPOCHS = 20
TRAINING_LEARNING_RATE = 5e-4
TRAINING_WARMUP_RATIO = 0.15

# Inference parameters
INFERENCE_NUM_BEAMS = 4

# GROQ API (Weak Labeler)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_key_here")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_WEAK_LABELER_PROMPT = """
You are a JSON data generator. Your sole output must be a single, valid JSON object.
For the given corporate jargon word/phrase '{jargon}', generate 10 realistic corporate/workplace input-output pairs.

Return a single JSON object in the following format:
{{
  "pairs": [
    {{"input": "simple phrase", "output": "jargon-filled phrase"}},
    {{"input": "...", "output":..."}}
  ]
}}
"""

# QA / VALIDATION
QA_GOLDEN_TEST_EXAMPLES = 20
QA_MIN_COSINE_DISTANCE = 0.5
QA_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# DVC CONFIG
DVC_REMOTE_NAME = "local_storage"
DVC_REMOTE_PATH = str(DVC_REMOTE)

# BENTOML CONFIG
BENTO_MODEL_NAME = "jargon_translator"
BENTO_SERVICE_NAME = "jargon_translator_service"
BENTO_PRODUCTION_TAG = "production"