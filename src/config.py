from pathlib import Path
import random
import torch

# ---------------------------
# Paths
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
PROJECT_CODENET = DATA_ROOT / "Project_CodeNet"
MUTATED_CODENET = DATA_ROOT / "Mutated_CodeNet"
GRAPHS_DIR = MUTATED_CODENET / "graphs_multiple_no_timestamp"

LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in [LOG_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Training hyperparameters
# ---------------------------
USE_CODEBERT = False

EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-5

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ---------------------------
# Reproducibility helper
# ---------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
