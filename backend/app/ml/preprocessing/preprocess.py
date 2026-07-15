from pathlib import Path
import numpy as np
import mne

from app.core.config import settings
from app.core.model_manifest import BIPOLAR_PAIRS, BIPOLAR_NAMES, CANON_ELECTRODES
from app.ml.preprocessing.channel_utils import canonicalize_ch_name


def normalize_and_pick_eeg(raw: mne.io.BaseRaw):
    raw = raw.copy()
    raw.rename_channels({ch: canonicalize_ch_name(ch) for ch in raw.ch_names})
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.pick_types(eeg=True)

    missing = sorted(list(set(CANON_ELECTRODES) - set(raw.ch_names)))
    if missing:
        raise RuntimeError(f"Missing required electrodes: {missing}")

    raw.pick(CANON_ELECTRODES)
    return raw


def make_bipolar_montage(raw: mne.io.BaseRaw) -> np.ndarray:
    raw = normalize_and_pick_eeg(raw)

    raw.filter(
        settings.l_freq,
        settings.h_freq,
        fir_design="firwin",
        verbose="ERROR",
    )
    raw.resample(settings.sfreq_target, npad="auto")

    raw = mne.set_bipolar_reference(
        raw,
        anode=[a for a, b in BIPOLAR_PAIRS],
        cathode=[b for a, b in BIPOLAR_PAIRS],
        ch_name=BIPOLAR_NAMES,
        drop_refs=True,
        copy=False,
    )

    if len(raw.ch_names) != 18:
        raise RuntimeError("Bipolar montage did not produce 18 channels")

    if raw.ch_names != BIPOLAR_NAMES:
        raise RuntimeError("Bipolar channel order mismatch")

    data = raw.get_data().astype(np.float32)

    total_seconds = data.shape[1] / settings.sfreq_target
    if total_seconds < settings.min_recording_seconds:
        raise RuntimeError(f"Recording too short: {total_seconds:.1f}s")

    # Channel-wise normalization
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    data = (data - mean) / std

    return data


def _effective_step_sec() -> int:
    """
    Lets you speed up demo inference without changing training preprocessing code.
    Falls back safely to step_sec_nonseizure.
    """
    fast_step = getattr(settings, "inference_step_sec", None)
    if fast_step is not None and fast_step > 0:
        return int(fast_step)

    return int(settings.step_sec_nonseizure)


def compute_timeline(total_samples: int):
    win_samp = int(settings.win_sec * settings.sfreq_target)
    step_sec = _effective_step_sec()
    step_samp = int(step_sec * settings.sfreq_target)

    if total_samples < win_samp:
        raise RuntimeError("Recording too short for one analysis window")

    timeline = []
    st = 0
    idx = 0

    while st + win_samp <= total_samples:
        ed = st + win_samp
        timeline.append({
            "window_index": idx,
            "start_sec": st / settings.sfreq_target,
            "end_sec": ed / settings.sfreq_target,
            "start_sample": st,
            "end_sample": ed,
        })
        st += step_samp
        idx += 1

    return timeline


def iter_window_batches(data: np.ndarray, batch_size: int = 16):
    """
    Yield batches of EEG windows.
    Input shape: [18, total_samples]
    Output shape per batch: [B, 18, win_samples]
    """
    if data.ndim != 2 or data.shape[0] != 18:
        raise RuntimeError(f"Expected data shape [18, total_samples], got {tuple(data.shape)}")

    win_samp = int(settings.win_sec * settings.sfreq_target)
    timeline = compute_timeline(data.shape[1])

    total_windows = len(timeline)
    for batch_start in range(0, total_windows, batch_size):
        batch_meta = timeline[batch_start: batch_start + batch_size]

        batch_windows = []
        cleaned_meta = []

        for item in batch_meta:
            st = item["start_sample"]
            ed = item["end_sample"]
            window = data[:, st:ed]

            if window.shape != (18, win_samp):
                continue

            batch_windows.append(window)
            cleaned_meta.append({
                "window_index": item["window_index"],
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
            })

        if not batch_windows:
            continue

        yield np.stack(batch_windows, axis=0).astype(np.float32), cleaned_meta


def preprocess_edf_signal(edf_path: Path):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    orig_sfreq = float(raw.info["sfreq"])

    # Optional demo safety crop
    max_minutes = getattr(settings, "max_recording_minutes", None)
    if max_minutes is not None and max_minutes > 0:
        max_seconds = float(max_minutes) * 60.0
        if raw.times[-1] > max_seconds:
            raw.crop(tmin=0.0, tmax=max_seconds)

    data = make_bipolar_montage(raw)
    duration_seconds = float(data.shape[1] / settings.sfreq_target)

    return data, {
        "duration_seconds": duration_seconds,
        "sampling_rate_original": orig_sfreq,
    }


def preprocess_edf_for_inference(edf_path: Path, batch_size: int = 16):
    """
    Returns:
    - preprocessed full signal
    - computed timeline
    - metadata

    Window batching is handled later using iter_window_batches(...).
    """
    data, meta = preprocess_edf_signal(edf_path)
    timeline = compute_timeline(data.shape[1])
    return data, timeline, meta