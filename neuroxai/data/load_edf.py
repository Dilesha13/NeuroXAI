import mne
import numpy as np

def load_preprocess_edf(edf_path, l_freq=0.5, h_freq=30.0, target_sfreq=100):
    """
    Safe prototype preprocessing:
    - bandpass 0.5–30 Hz
    - resample to 100 Hz
    - channel-wise z-score
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(l_freq, h_freq)
    raw.resample(target_sfreq)

    x = raw.get_data().astype(np.float32)  # (ch, samples)
    sfreq = float(raw.info["sfreq"])

    # channel-wise normalization (per recording)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return x, sfreq, raw.ch_names
