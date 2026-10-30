import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
VERL_SOURCE_ROOT = TRAINING_ROOT / "verl"

for candidate in (REPO_ROOT, TRAINING_ROOT, VERL_SOURCE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
