import random

def split_files(edf_files, seed=42, train_ratio=0.70, val_ratio=0.10):
    """
    Split list of filenames into train/val/test by file (prevents leakage).
    """
    edf_files = list(edf_files)
    random.Random(seed).shuffle(edf_files)

    n = len(edf_files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_files = edf_files[:n_train]
    val_files = edf_files[n_train:n_train+n_val]
    test_files = edf_files[n_train+n_val:]

    assert len(train_files) + len(val_files) + len(test_files) == n
    return train_files, val_files, test_files
