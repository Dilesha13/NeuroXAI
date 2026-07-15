from pathlib import Path
import json
import numpy as np
from app.core.model_manifest import BIPOLAR_NAMES


def save_temporal_attention(model, save_path: Path):
    payload = {}

    attn = getattr(model, "last_temporal_attn", None)
    tokens = getattr(model, "last_temporal_tokens", None)

    if attn is not None and tokens is not None:
        attn_arr = attn.detach().cpu().numpy()   # [B, H, T, T]
        tok_arr = tokens.detach().cpu().numpy()  # [B, T, D]

        if attn_arr.ndim == 4 and tok_arr.ndim == 3 and attn_arr.shape[0] > 0 and tok_arr.shape[0] > 0:
            sample_attn = attn_arr[0]   # [H, T, T]
            sample_tok = tok_arr[0]     # [T, D]

            # Mean over heads -> [T, T]
            mean_heads = sample_attn.mean(axis=0)

            # Token importance from attention received by each token
            received_attention = mean_heads.mean(axis=0)  # [T]

            # Token feature magnitude
            token_strength = np.linalg.norm(sample_tok, axis=1)  # [T]

            # Combine both signals
            scores = received_attention * token_strength

            # Safe normalization for frontend
            scores = np.asarray(scores, dtype=np.float32)
            if scores.size > 0:
                min_val = float(scores.min())
                max_val = float(scores.max())
                if max_val - min_val > 1e-8:
                    scores_norm = (scores - min_val) / (max_val - min_val)
                else:
                    scores_norm = np.zeros_like(scores, dtype=np.float32)
            else:
                scores_norm = scores

            top_idx = np.argsort(scores_norm)[::-1][:5]

            payload = {
                "shape": list(attn_arr.shape),
                "num_tokens": int(scores_norm.shape[0]),
                "token_importance": scores_norm.tolist(),
                "top_segments": [
                    {
                        "segment_index": int(i),
                        "importance": float(scores_norm[i]),
                    }
                    for i in top_idx
                ],
            }

    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_gat_attention(model, save_path: Path):
    payload = {}

    gat2 = getattr(model, "gat2", None)
    alpha = getattr(gat2, "last_alpha", None) if gat2 is not None else None

    if alpha is not None:
        arr = alpha.detach().cpu().numpy()

        # Expected [B, H, E], but handle safely
        if arr.ndim == 3:
            mean_edge_attention = arr.mean(axis=(0, 1))   # [E]
        elif arr.ndim == 2:
            mean_edge_attention = arr.mean(axis=0)        # [E]
        elif arr.ndim == 1:
            mean_edge_attention = arr
        else:
            mean_edge_attention = np.array([], dtype=np.float32)

        mean_edge_attention = np.asarray(mean_edge_attention, dtype=np.float32)

        if mean_edge_attention.size > 0:
            min_val = float(mean_edge_attention.min())
            max_val = float(mean_edge_attention.max())
            if max_val - min_val > 1e-8:
                mean_edge_attention_norm = (mean_edge_attention - min_val) / (max_val - min_val)
            else:
                mean_edge_attention_norm = np.zeros_like(mean_edge_attention, dtype=np.float32)
        else:
            mean_edge_attention_norm = mean_edge_attention

        payload = {
            "shape": list(arr.shape),
            "mean_edge_attention": mean_edge_attention_norm.tolist(),
        }

    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_top_channels(top_channels):
    if not top_channels:
        return "No dominant channels could be extracted."

    ch_txt = ", ".join(top_channels[:3])
    return (
        f"Highest model relevance was observed in channels such as {ch_txt}. "
        f"These should be interpreted as supportive explainability cues rather than clinical proof."
    )