import os
from pathlib import Path

# Project root = one directory above this file (src/)
PROJECT_ROOT = Path(os.path.dirname(__file__)).parent

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH                 = MODELS_DIR  / "image_classifier.joblib"
TRAIN_HOG_PATH             = MODELS_DIR  / "train_hog.joblib"
TEST_HOG_PATH              = MODELS_DIR  / "test_hog.joblib"

CONFUSION_MATRIX_PATH      = OUTPUTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUTS_DIR / "classification_report.txt"
PREDICTIONS_PATH           = OUTPUTS_DIR / "predictions.png"
