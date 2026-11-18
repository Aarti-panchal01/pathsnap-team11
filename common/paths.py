"""Resolve project and data locations without embedding machine-specific paths."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_data_dir = os.environ.get("PATHSNAP_DATA_DIR")
if not _data_dir:
    raise RuntimeError("Set PATHSNAP_DATA_DIR to the PathSnap dataset directory.")

DATA_DIR = Path(_data_dir).expanduser().resolve()
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
