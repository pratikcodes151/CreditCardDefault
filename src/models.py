
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from .features import build_preprocessor
from .config import CALIBRATED, RANDOM_STATE

def build_model(algorithm: str = "logreg") -> Pipeline:
    pre = build_preprocessor()

    if algorithm == "logreg":
        base = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
            solver="lbfgs",
        )
    elif algorithm == "rf":
        base = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        raise ValueError("Unknown algorithm. Use 'logreg' or 'rf'.")

    clf = base
    if CALIBRATED and algorithm != "rf":  # RF is often reasonably calibrated
        clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe
