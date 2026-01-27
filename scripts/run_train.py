import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from neuroxai.models.cnn1d import EEG_CNN  # we will create this next

DATA_DIR = "/content/drive/MyDrive/FYP/NeuroXAI/data_processed"
OUT_DIR  = "/content/drive/MyDrive/FYP/NeuroXAI/outputs/runs/prototype_cnn"
os.makedirs(OUT_DIR, exist_ok=True)

# Load arrays
X_train = np.load(f"{DATA_DIR}/X_train.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
X_val   = np.load(f"{DATA_DIR}/X_val.npy")
y_val   = np.load(f"{DATA_DIR}/y_val.npy")

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)            # (N, C, T)
        self.y = torch.from_numpy(y).float()    # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def compute_pos_weight(y):
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No seizure windows in training set.")
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(EEGDataset(X_val,   y_val),   batch_size=128, shuffle=False)

model = EEG_CNN(n_ch=X_train.shape[1]).to(device)

pos_weight = compute_pos_weight(y_train).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def eval_loss(loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total += loss.item() * len(yb)
            n += len(yb)
    return total / n

best_val = float("inf")

for epoch in range(1, 11):  # 10 epochs prototype
    model.train()
    total, n = 0.0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(yb)
        n += len(yb)

    train_loss = total / n
    val_loss = eval_loss(val_loader)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        ckpt_path = os.path.join(OUT_DIR, "best_model.pt")
        torch.save({"model_state": model.state_dict(), "n_ch": X_train.shape[1]}, ckpt_path)
        print("  Saved best model:", ckpt_path)

print("Done. Best val loss:", best_val)
