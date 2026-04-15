from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels: List[int], dropout: float):
        super().__init__()
        layers = []
        cur = in_ch
        for k in kernels:
            layers += [
                nn.Conv1d(cur, out_ch, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            cur = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GATLayer(nn.Module):
    def __init__(self, fin: int, fout: int, heads: int, dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.fin = fin
        self.fout = fout
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W = nn.Linear(fin, heads * fout, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, fout))
        self.a_dst = nn.Parameter(torch.empty(heads, fout))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.last_alpha = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        E = edge_index.shape[1]

        src = edge_index[0].to(x.device)
        dst = edge_index[1].to(x.device)

        h = self.W(x).view(B, N, self.heads, self.fout)
        h_src = h[:, src, :, :]
        h_dst = h[:, dst, :, :]

        e = ((h_src * self.a_src[None, None, :, :]).sum(-1) + (h_dst * self.a_dst[None, None, :, :]).sum(-1))
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        alpha = torch.zeros_like(e)
        for n in range(N):
            mask = (dst == n)
            if mask.sum() == 0:
                continue
            logits = e[:, mask, :]
            logits = logits - logits.max(dim=1, keepdim=True).values
            attn = torch.exp(logits)
            attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-9)
            alpha[:, mask, :] = attn

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        self.last_alpha = alpha.permute(0, 2, 1).detach()

        out = torch.zeros(B, N, self.heads, self.fout, device=x.device, dtype=x.dtype)
        for ei in range(E):
            out[:, dst[ei], :, :] += alpha[:, ei, :].unsqueeze(-1) * h_src[:, ei, :, :]
        return out.reshape(B, N, self.heads * self.fout)

class MSTGAT(nn.Module):
    def __init__(self, embed_dim: int = 128, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads

        self.short = TemporalBlock(1, 32, [3, 5, 7], 0.2)
        self.medium = TemporalBlock(1, 32, [7, 15, 31], 0.2)
        self.long = nn.Sequential(
            nn.Conv1d(1, 32, 63, padding=31, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AvgPool1d(4, 4),
            nn.Conv1d(32, 32, 31, padding=15, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(25)
        )

        self.node_proj = nn.Linear(32 * 3, embed_dim)
        self.node_norm = nn.LayerNorm(embed_dim)
        self.node_dropout = nn.Dropout(dropout)

        self.gat1 = GATLayer(embed_dim, embed_dim // heads, heads, dropout=0.1)
        self.gat2 = GATLayer(embed_dim, embed_dim // heads, heads, dropout=0.1)
        self.gat_norm = nn.LayerNorm(embed_dim)

        self.temporal_tok_proj = nn.Linear(32, embed_dim)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.temporal_norm1 = nn.LayerNorm(embed_dim)
        self.temporal_norm2 = nn.LayerNorm(embed_dim)
        self.temporal_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.1),
        )
        self.last_temporal_attn = None

        fused_dim = (32 + 32 + 32) + embed_dim + embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        xn = x.reshape(B * N, 1, T)

        short_f = self.short(xn)
        med_f   = self.medium(xn)
        long_f  = self.long(xn)

        short_vec = F.adaptive_avg_pool1d(short_f, 1).squeeze(-1)
        med_vec   = F.adaptive_avg_pool1d(med_f, 1).squeeze(-1)
        long_vec  = F.adaptive_avg_pool1d(long_f, 1).squeeze(-1)

        node_feat = torch.cat([short_vec, med_vec, long_vec], dim=1)
        node_emb = self.node_proj(node_feat).view(B, N, self.embed_dim)
        node_emb = self.node_dropout(self.node_norm(node_emb))

        g = F.gelu(self.gat1(node_emb, edge_index))
        g = self.gat2(g, edge_index)
        g = self.gat_norm(g + node_emb)
        spatial_summary = g.mean(dim=1)

        med_f_ = med_f.view(B, N, 32, T).mean(dim=1)
        L = 50
        tokens = F.adaptive_avg_pool1d(med_f_, L).transpose(1, 2)
        tok = F.gelu(self.temporal_tok_proj(tokens))

        attn_out, attn_w = self.temporal_attn(tok, tok, tok, need_weights=True, average_attn_weights=False)
        self.last_temporal_attn = attn_w.detach()

        tok2 = self.temporal_norm1(tok + attn_out)
        tok3 = self.temporal_norm2(tok2 + self.temporal_ff(tok2))
        self.last_temporal_tokens = tok3.detach()
        temporal_summary = tok3.mean(dim=1)

        short_global = short_vec.view(B, N, 32).mean(dim=1)
        med_global   = med_vec.view(B, N, 32).mean(dim=1)
        long_global  = long_vec.view(B, N, 32).mean(dim=1)

        fused = torch.cat([short_global, med_global, long_global, spatial_summary, temporal_summary], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits
