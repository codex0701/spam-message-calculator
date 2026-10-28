"""Streamlit demo: paste a message, get spam / not spam."""
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"

st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Spam Message Classifier")
st.write("Paste a message and the model will predict whether it is spam.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
text = st.text_area("Message", height=120, placeholder="e.g. WINNER! Claim your free prize now")

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        prob = model.predict_proba([text])[0][1]
        if prob >= 0.5:
            st.error(f"🚨 Spam  (confidence {prob:.0%})")
        else:
            st.success(f"✅ Not spam  (spam probability {prob:.0%})")
