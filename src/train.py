
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
from .data import load_dataset
from .models import build_model
from .features import ALL_FEATURES
from .config import RANDOM_STATE, TEST_SIZE, MODEL_DIR, MODEL_FILENAME, DEFAULT_TARGET
from .utils import ensure_dir, save_obj

def main(args):
    df = load_dataset(args.data_path)
    df = df.sample(frac=0.1, random_state=RANDOM_STATE)

    target = args.target or DEFAULT_TARGET
    assert target in df.columns, f"Target '{target}' not in dataframe columns"

    X = df[ALL_FEATURES].copy()
    y = df[target].astype(int).values

    model = build_model(args.algorithm)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    roc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    pr = cross_val_score(model, X, y, cv=cv, scoring="average_precision").mean()
    print(f"CV ROC AUC: {roc:.4f} | PR AUC: {pr:.4f}")

    # Fit on full training set and persist
    model.fit(X, y)
    ensure_dir(MODEL_DIR)
    model_path = Path(MODEL_DIR) / MODEL_FILENAME
    save_obj(model, model_path)
    print(f"Saved model to {model_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/default_of_credit_card_clients.csv")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--algorithm", type=str, choices=["logreg","rf"], default="logreg")
    args = parser.parse_args()
    main(args)
