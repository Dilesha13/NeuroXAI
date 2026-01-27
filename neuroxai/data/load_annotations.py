import numpy as np
from scipy.io import loadmat

def load_annotat_new(mat_path):
    """
    Loads the Helsinki annotation cell array from annotations_2017.mat.
    Returns the object stored under key 'annotat_new'.
    """
    mat = loadmat(mat_path)
    if "annotat_new" not in mat:
        raise KeyError("Expected key 'annotat_new' not found in MAT file.")
    return mat["annotat_new"]

def get_baby_seconds_matrix(annotat_new, baby_id):
    """
    baby_id: 1..79
    Returns A with shape (3, N_seconds) where rows=experts and columns=seconds.
    """
    i = baby_id - 1  # MATLAB->Python index
    # Common MATLAB cell import: shape (79,1) or (1,79)
    if annotat_new.shape[0] == 79:
        cell = annotat_new[i, 0]
    else:
        cell = annotat_new[0, i]

    A = np.array(cell)

    # Ensure A is (3, N)
    if A.ndim != 2:
        raise ValueError(f"Unexpected annotation element shape: {A.shape}")
    if A.shape[0] != 3 and A.shape[1] == 3:
        A = A.T
    if A.shape[0] != 3:
        raise ValueError(f"Expected 3 experts, got shape {A.shape}")

    # Convert to 0/1 integers safely
    A = (A > 0).astype(np.uint8)
    return A

def consensus_seconds(annotat_new, baby_id, mode="strict"):
    """
    Returns consensus 1D array of length N_seconds.
    mode:
      - 'strict': seizure second if ALL 3 experts say seizure
      - 'majority': seizure second if >=2 experts say seizure
    """
    A = get_baby_seconds_matrix(annotat_new, baby_id)  # (3, N)
    s = A.sum(axis=0)
    if mode == "strict":
        return (s == 3).astype(np.uint8)
    elif mode == "majority":
        return (s >= 2).astype(np.uint8)
    else:
        raise ValueError("mode must be 'strict' or 'majority'")
