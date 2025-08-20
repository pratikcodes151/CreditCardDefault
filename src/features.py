
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

CATEGORICAL = [
    "sex", "education", "marriage",
    "pay_0","pay_2","pay_3","pay_4","pay_5","pay_6"
]

NUMERIC = [
    "limit_bal","age",
    "bill_amt1","bill_amt2","bill_amt3","bill_amt4","bill_amt5","bill_amt6",
    "pay_amt1","pay_amt2","pay_amt3","pay_amt4","pay_amt5","pay_amt6"
]

ALL_FEATURES = CATEGORICAL + NUMERIC

def _clip_skew(X: pd.DataFrame) -> pd.DataFrame:
    # Mild winsorization to reduce impact of extreme values on numeric columns
    X = X.copy()
    for col in NUMERIC:
        if col in X.columns:
            lo, hi = X[col].quantile([0.01, 0.99])
            X[col] = X[col].clip(lo, hi)
    return X

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("clip", FunctionTransformer(_clip_skew)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    categorical_pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop"
    )
    return pre
