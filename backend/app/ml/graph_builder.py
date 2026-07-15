import torch
from app.core.model_manifest import BIPOLAR_PAIRS, BIPOLAR_NAMES

def build_edge_index() -> torch.Tensor:
    edges = set()
    for i, (a1, b1) in enumerate(BIPOLAR_PAIRS):
        for j, (a2, b2) in enumerate(BIPOLAR_PAIRS):
            if i != j and len({a1, b1}.intersection({a2, b2})) > 0:
                edges.add((i, j))
                edges.add((j, i))

    idx_fzcz = BIPOLAR_NAMES.index('Fz-Cz')
    idx_czpz = BIPOLAR_NAMES.index('Cz-Pz')
    for nm in ['F4-C4', 'F3-C3', 'C4-P4', 'C3-P3']:
        k = BIPOLAR_NAMES.index(nm)
        edges.add((idx_fzcz, k))
        edges.add((k, idx_fzcz))
        edges.add((idx_czpz, k))
        edges.add((k, idx_czpz))

    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()
