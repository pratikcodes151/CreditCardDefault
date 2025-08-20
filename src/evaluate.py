
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    brier_score_loss, f1_score
)
from .data import load_dataset
from .features import ALL_FEATURES
from .config import DEFAULT_TARGET
from .utils import load_obj

def main(args):
    model = load_obj(args.model_path)
    df = load_dataset(args.data_path)
    target = args.target or DEFAULT_TARGET

    X = df[ALL_FEATURES]
    y = df[target].astype(int).values

    proba = model.predict_proba(X)[:,1]
    preds = (proba >= args.threshold).astype(int)

    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    f1 = f1_score(y, preds)

    print(f"ROC AUC: {roc:.4f}")
    print(f"PR  AUC: {pr:.4f}")
    print(f"Brier : {brier:.4f}")
    print(f"F1@{args.threshold:.2f}: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/model.pkl")
    parser.add_argument("--data-path", type=str, default="data/default_of_credit_card_clients.csv")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
