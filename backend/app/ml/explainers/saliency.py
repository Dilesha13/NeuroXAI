from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from app.core.model_manifest import BIPOLAR_NAMES


def generate_saliency(
    model,
    x_window: torch.Tensor,
    edge_index: torch.Tensor,
    save_path: Path,
    downsample: int = 2,   # NEW: reduce computation cost
):
    """
    Optimized saliency:
    - optional temporal downsampling (huge speed gain)
    - uses autograd.grad instead of backward()
    - avoids unnecessary graph retention
    """

    # ------------------------------------------------------------------
    # 1. Optional downsampling (VERY IMPORTANT for speed)
    # ------------------------------------------------------------------
    if downsample > 1:
        x_window = x_window[:, :, ::downsample]

    x = x_window.clone().detach().requires_grad_(True)

    # ------------------------------------------------------------------
    # 2. Forward pass
    # ------------------------------------------------------------------
    logits = model(x, edge_index)
    score = logits.squeeze()

    # ------------------------------------------------------------------
    # 3. Compute gradients (faster than backward)
    # ------------------------------------------------------------------
    grads = torch.autograd.grad(
        outputs=score,
        inputs=x,
        retain_graph=False,
        create_graph=False,
        allow_unused=False
    )[0]

    # ------------------------------------------------------------------
    # 4. Convert to numpy
    # ------------------------------------------------------------------
    sal = grads[0].detach().cpu().numpy()
    sal = np.abs(sal)

    # normalize
    sal = sal / (sal.max() + 1e-8)

    # ------------------------------------------------------------------
    # 5. Save heatmap (same as before)
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.imshow(sal, aspect='auto')
    plt.colorbar()
    plt.yticks(range(len(BIPOLAR_NAMES)), BIPOLAR_NAMES, fontsize=7)
    plt.xlabel('Time samples')
    plt.ylabel('Bipolar EEG channels')
    plt.title('Gradient Saliency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # 6. Channel importance
    # ------------------------------------------------------------------
    channel_scores = sal.mean(axis=1)
    top_idx = np.argsort(channel_scores)[::-1][:5]

    return [BIPOLAR_NAMES[int(i)] for i in top_idx]