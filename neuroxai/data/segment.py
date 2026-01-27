import os
import numpy as np

from .load_edf import load_preprocess_edf
from .load_annotations import consensus_seconds


def make_windows_and_labels_from_1hz(
    x, sfreq, consensus_1hz,
    window_sec=10, step_sec=5,
    min_seizure_seconds=1
):
    """
    x: (channels, samples)
    sfreq: sampling frequency in Hz
    consensus_1hz: (N_seconds,) 0/1 label per second
    window labeled seizure if >= min_seizure_seconds seizure seconds in that window.
    """
    win_samp = int(window_sec * sfreq)
    step_samp = int(step_sec * sfreq)

    X, y = [], []

    max_start = x.shape[1] - win_samp
    for start in range(0, max_start + 1, step_samp):
        end = start + win_samp

        # Convert sample window to seconds indices (1Hz labels)
        t0 = int(np.floor(start / sfreq))
        t1 = int(np.ceil(end / sfreq))

        # Clip to available annotation length
        t0 = max(t0, 0)
        t1 = min(t1, len(consensus_1hz))

        seg = x[:, start:end]

        seizure_seconds = int(consensus_1hz[t0:t1].sum())
        label = 1 if seizure_seconds >= min_seizure_seconds else 0

        X.append(seg)
        y.append(label)

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def file_to_baby_id(fname):
    """
    Converts 'eeg1.edf' -> 1
    """
    base = os.path.basename(fname)
    num = base.replace("eeg", "").replace(".edf", "")
    return int(num)
