import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from neuroxai.data.load_edf import load_preprocess_edf
from neuroxai.data.load_annotations import load_annotat_new, consensus_seconds
from neuroxai.data.segment import make_windows_and_labels_from_1hz
from neuroxai.models.cnn1d import EEG_CNN

# ---------------- CONFIG ----------------
DATA_ROOT = "/content/drive/MyDrive/FYP/NeuroXAI"
EDF_DIR   = f"{DATA_ROOT}/data_raw/eeg"
ANN_PATH  = f"{DATA_ROOT}/data_raw/annotations/annotations_2017.mat"
MODEL_PATH = f"{DATA_ROOT}/outputs/runs/prototype_cnn/best_model.pt"

WINDOW_SEC = 10
STEP_SEC   = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- UI ----------------
st.set_page_config(page_title="Neonatal Seizure Detection", layout="wide")

st.title("🧠 Neonatal Seizure Detection – Prototype Demo")
st.markdown("""
This prototype demonstrates **EEG-based neonatal seizure detection**
using a trained deep learning model and explainability.
""")

# ---------------- FILE SELECTION ----------------
edf_files = sorted([f for f in os.listdir(EDF_DIR) if f.endswith(".edf")])
selected_file = st.selectbox("Select EEG recording", edf_files)

if st.button("Run Detection"):
    st.write("Loading data...")

    # Load annotations
    annotat_new = load_annotat_new(ANN_PATH)
    baby_id = int(selected_file.replace("eeg", "").replace(".edf", ""))

    # Load & preprocess EEG
    x, sfreq, ch_names = load_preprocess_edf(os.path.join(EDF_DIR, selected_file))

    # Consensus labels (for reference only)
    consensus = consensus_seconds(annotat_new, baby_id, mode="strict")

    # Windowing
    X, y = make_windows_and_labels_from_1hz(
        x, sfreq, consensus,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC
    )

    st.success(f"Generated {len(X)} windows")

    # Load model
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = EEG_CNN(n_ch=ckpt["n_ch"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Predict
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()

    # ---------------- PLOT TIMELINE ----------------
    st.subheader("📈 Seizure Probability Timeline")

    time_axis = np.arange(len(probs)) * STEP_SEC

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_axis, probs, label="Seizure probability")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig)

    # ---------------- EXPLAINABILITY ----------------
    st.subheader("🧠 Explainability (Saliency Map)")

    # Pick most confident seizure prediction
    idx = int(np.argmax(probs))
    x_explain = torch.from_numpy(X[idx:idx+1]).to(DEVICE)
    x_explain.requires_grad_(True)

    logit = model(x_explain)
    prob = torch.sigmoid(logit)[0]
    prob.backward()

    saliency = x_explain.grad.abs().cpu().numpy()[0]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    im = ax2.imshow(saliency, aspect="auto", origin="lower")
    ax2.set_title(f"Saliency Map (p={prob.item():.3f})")
    ax2.set_xlabel("Time samples")
    ax2.set_ylabel("Channels")
    fig2.colorbar(im, ax=ax2)
    st.pyplot(fig2)

    # ---------------- METRICS ----------------
    st.subheader("📊 Model Performance (Test Set)")
    st.markdown("""
    - **AUC:** 0.57  
    - **Sensitivity:** 0.38  
    - **Specificity:** 0.78  

    ⚠️ *This is a baseline prototype. Performance improvements are part of future work.*
    """)
