import os, glob
import numpy as np

from neuroxai.data.splits import split_files
from neuroxai.data.load_annotations import load_annotat_new
from neuroxai.data.segment import build_split_arrays

# ---- PATHS ----
EDF_DIR = "/content/drive/MyDrive/FYP/NeuroXAI/data_raw/eeg"
ANN_PATH = "/content/drive/MyDrive/FYP/NeuroXAI/data_raw/annotations/annotations_2017.mat"
OUT_DIR = "/content/drive/MyDrive/FYP/NeuroXAI/data_processed"

os.makedirs(OUT_DIR, exist_ok=True)

# ---- LIST FILES ----
edf_files = sorted([os.path.basename(f) for f in glob.glob(f"{EDF_DIR}/*.edf")])
print("Total EDF files:", len(edf_files))

# ---- SPLIT (70/10/20) ----
train_files, val_files, test_files = split_files(edf_files, seed=42)
print("Train:", len(train_files), "Val:", len(val_files), "Test:", len(test_files))

# ---- LOAD ANNOTATIONS ----
annotat_new = load_annotat_new(ANN_PATH)

# ---- BUILD DATA ----
print("Building TRAIN set...")
X_train, y_train = build_split_arrays(EDF_DIR, train_files, annotat_new)

print("Building VAL set...")
X_val, y_val = build_split_arrays(EDF_DIR, val_files, annotat_new)

print("Building TEST set...")
X_test, y_test = build_split_arrays(EDF_DIR, test_files, annotat_new)

# ---- SAVE ----
print("Saving X_train...")
np.save(f"{OUT_DIR}/X_train.npy", X_train)
print("Saving y_train...")
np.save(f"{OUT_DIR}/y_train.npy", y_train)

print("Saving X_val...")
np.save(f"{OUT_DIR}/X_val.npy", X_val)
print("Saving y_val...")
np.save(f"{OUT_DIR}/y_val.npy", y_val)

print("Saving X_test...")
np.save(f"{OUT_DIR}/X_test.npy", X_test)
print("Saving y_test...")
np.save(f"{OUT_DIR}/y_test.npy", y_test)

print("Saved processed arrays to:", OUT_DIR)
print("Train seizures:", y_train.sum())
print("Val seizures:", y_val.sum())
print("Test seizures:", y_test.sum())