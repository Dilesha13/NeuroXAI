import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

from neuroxai.models.cnn1d import EEG_CNN

DATA_DIR = "/content/drive/MyDrive/FYP/NeuroXAI/data_processed"
RUN_DIR  = "/content/drive/MyDrive/FYP/NeuroXAI/outputs/runs/prototype_cnn"
CKPT_PATH = os.path.join(RUN_DIR, "best_model.pt")

OUT_METRICS = os.path.join(RUN_DIR, "metrics_test.json")
OUT_CM_TXT  = os.path.join(RUN_DIR, "confusion_matrix_test.txt")

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def predict_probs(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
            ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)

def sens_spec_from_cm(cm):
    # cm = [[tn, fp],[fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load test data
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=256, shuffle=False)

# Load model
ckpt = torch.load(CKPT_PATH, map_location=device)
model = EEG_CNN(n_ch=ckpt["n_ch"]).to(device)
model.load_state_dict(ckpt["model_state"])

# Predict
probs, ys = predict_probs(model, test_loader, device)

auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")

# Default threshold = 0.5 (prototype)
pred = (probs >= 0.5).astype(int)
cm = confusion_matrix(ys.astype(int), pred, labels=[0,1])

sens, spec = sens_spec_from_cm(cm)

metrics = {
    "AUC": float(auc),
    "Sensitivity": float(sens),
    "Specificity": float(spec),
    "Threshold": 0.5,
    "Test_seizure_ratio": float(ys.mean()),
    "N_test": int(len(ys))
}

print("AUC:", metrics["AUC"])
print("Sensitivity:", metrics["Sensitivity"])
print("Specificity:", metrics["Specificity"])
print("Confusion matrix:\n", cm)

# Save
os.makedirs(RUN_DIR, exist_ok=True)
with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

with open(OUT_CM_TXT, "w") as f:
    f.write("Confusion matrix [[tn, fp],[fn, tp]]:\n")
    f.write(str(cm))

print("Saved:", OUT_METRICS)
print("Saved:", OUT_CM_TXT)