import os, glob
import numpy as np

from neuroxai.data.splits import split_files
from neuroxai.data.load_annotations import load_annotat_new
from neuroxai.data.build_dataset import build_split_arrays

# ---- PATHS ----
EDF_DIR = "/content/drive/MyDrive/FYP/NeuroXAI/data_raw/eeg"
ANN_PATH = "/content/drive/MyDrive/FYP/NeuroXAI/data_raw/annotations/annotations_2017.mat"
OUT_DIR = "/content/drive/MyDrive/FYP/NeuroXAI/data_processed"

os.makedirs(OUT_DIR, exist_ok=True)

# ---- LIST FILES ----
edf_files = sorted([os.path.basename(f) for f in glob.glob(f"{EDF_DIR}/*.edf")])
print("Total EDF files:", len(edf_files))