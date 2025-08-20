import sys
import os
import streamlit as st
import pandas as pd
from pathlib import Path

# 👇 Add project root to sys.path so "src" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_obj
from src.features import ALL_FEATURES


st.set_page_config(page_title="Credit Card Default Predictor", layout="centered")

st.title("Credit Card Default Predictor")
st.caption("Trained on the UCI Default of Credit Card Clients dataset")

model_path = Path("models/model.pkl")
if not model_path.exists():
    st.warning("⚠️ No trained model found at models/model.pkl. Train first: `python -m src.train`")
else:
    model = load_obj(model_path)

    st.subheader("Enter applicant/account data")
    vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(ALL_FEATURES):
        with cols[i % 3]:
            if feat in ["sex", "education", "marriage", "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]:
                vals[feat] = st.number_input(feat, value=0, step=1)
            else:
                vals[feat] = st.number_input(feat, value=0.0, step=100.0)

    if st.button("Score"):
        df = pd.DataFrame([vals], columns=ALL_FEATURES)
        proba = float(model.predict_proba(df)[:, 1][0])
        pred = int(proba >= 0.5)
        st.metric("Probability of Default", f"{proba:.2%}")
        st.write("Prediction:", "Default" if pred == 1 else "No Default")
