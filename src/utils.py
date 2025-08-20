
import os
import joblib
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_obj(obj, path):
    ensure_dir(Path(path).parent)
    joblib.dump(obj, path)

def load_obj(path):
    return joblib.load(path)
