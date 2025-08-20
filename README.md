
# Credit Card Default — End‑to‑End ML Project

An opinionated, production‑ready starter for building a classifier that predicts the probability of default on the classic **UCI "Default of Credit Card Clients"** dataset.

## What's inside
- Reproducible training with scikit‑learn Pipelines
- Careful preprocessing for categorical codes and skewed numeric features
- Class‑imbalance handling (`class_weight='balanced'` + threshold tuning)
- Cross‑validation, metrics (ROC AUC, PR AUC, calibration)
- Model persistence
- CLI + Streamlit app for interactive scoring
- Tests to keep you honest

## Quickstart
```bash
# 1) Create & activate a virtual environment (any tool is fine)
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Auto)fetch dataset & train the model
python src/train.py --data-path data/default_of_credit_card_clients.csv --target default_payment_next_month

# 4) Evaluate
python src/evaluate.py --model-path models/model.pkl --data-path data/default_of_credit_card_clients.csv --target default_payment_next_month

# 5) Try the Streamlit app
streamlit run app/streamlit_app.py
```

### Dataset
If `data/default_of_credit_card_clients.csv` doesn't exist, training will attempt to download the public dataset from UCI automatically. You can also place a CSV with the same columns manually.

### Project layout
```
cc-default-starter/
├─ app/
│  └─ streamlit_app.py
├─ data/                      # dataset download target (gitignored)
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ features.py
│  ├─ models.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ predict.py
│  └─ utils.py
├─ tests/
│  └─ test_smoke.py
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
└─ README.md
```

### Notes
- Threshold tuning uses validation data to maximize F1 by default, but you can choose a different objective via CLI.
- All randomness is controlled by `RANDOM_STATE` in `src/config.py`.
- This starter is intentionally compact; extend with experiment tracking (e.g., MLflow) if you like.
