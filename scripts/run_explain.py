import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neuroxai.models.cnn1d import EEG_CNN

DATA_DIR = "/content/drive/MyDrive/FYP/NeuroXAI/data_processed"
RUN_DIR  = "/content/drive/MyDrive/FYP/NeuroXAI/outputs/runs/prototype_cnn"
CKPT_PATH = os.path.join(RUN_DIR, "best_model.pt")

OUT_FIG = os.path.join(RUN_DIR, "saliency_example.png")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test data
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

# Pick one seizure example if possible
idxs = np.where(y_test == 1)[0]
idx = int(idxs[0]) if len(idxs) > 0 else 0

x = torch.from_numpy(X_test[idx:idx+1]).to(device)
x.requires_grad_(True)

# Load model
ckpt = torch.load(CKPT_PATH, map_location=device)
model = EEG_CNN(n_ch=ckpt["n_ch"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Forward + backward
logit = model(x)
prob = torch.sigmoid(logit)[0]
prob.backward()

# Saliency = absolute gradient
saliency = x.grad.detach().cpu().numpy()[0]
saliency = np.abs(saliency)

# Plot
plt.figure(figsize=(10, 4))
plt.imshow(saliency, aspect="auto", origin="lower")
plt.colorbar(label="|Gradient|")
plt.xlabel("Time samples")
plt.ylabel("Channels")
plt.title(f"Saliency map (predicted p={prob.item():.3f}, true label={y_test[idx]})")
plt.tight_layout()

os.makedirs(RUN_DIR, exist_ok=True)
plt.savefig(OUT_FIG, dpi=150)
plt.close()

print("Saved explainability figure:", OUT_FIG)