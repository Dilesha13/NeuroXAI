BIPOLAR_PAIRS = [
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
BIPOLAR_NAMES = [f'{a}-{b}' for a, b in BIPOLAR_PAIRS]
CANON_ELECTRODES = sorted(list(set([x for pair in BIPOLAR_PAIRS for x in pair])))

MODEL_MANIFEST = {
    'model_name': 'mst_gat',
    'checkpoint_name': 'neuroxai_best_auc.pt',
    'threshold': 0.21,
    'sfreq_target': 100,
    'bandpass': [0.5, 30.0],
    'win_sec': 10,
    'step_sec_seizure': 5,
    'step_sec_nonseizure': 10,
    'bipolar_pairs': BIPOLAR_PAIRS,
    'bipolar_names': BIPOLAR_NAMES,
}
