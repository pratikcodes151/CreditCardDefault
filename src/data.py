
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import io

from .config import UCI_URL, CSV_FALLBACK_URL, DEFAULT_DATA_PATH

UCI_SHEET_INDEX = 0

def _download_uci_xls():
    try:
        r = requests.get(UCI_URL, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        return None

def _download_csv_fallback():
    try:
        r = requests.get(CSV_FALLBACK_URL, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def load_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)

    p.parent.mkdir(parents=True, exist_ok=True)

    # Try UCI XLS, then fallback CSV
    xls = _download_uci_xls()
    if xls is not None:
        try:
            df = pd.read_excel(io.BytesIO(xls), sheet_name=UCI_SHEET_INDEX, header=1)
            # Normalize column names
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            # The target column is 'default_payment_next_month'
            df.to_csv(p, index=False)
            return df
        except Exception:
            pass

    csv = _download_csv_fallback()
    if csv is not None:
        df = pd.read_csv(io.BytesIO(csv))
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        df.to_csv(p, index=False)
        return df

    raise FileNotFoundError(f"Could not load dataset to {p}. Place the CSV manually.")
