"""Phase 3 action policy pi_phi on shared VGGT substrate.

Closes the "make it a VLA" axis of the project. Maps:
    (context tokens [B, T, P, D], text embedding [B, Dt]) -> actions [B, T, A]

Architecture: small Transformer encoder over the time × patch axis (T*P=64 patches × 8 frames = 512 tokens),
plus text embedding broadcast, with a learnable per-frame action query that reads out the action vector.

Trained in two stages:
  Stage 1 (BC):     L = MSE(pi(s,l), a_demo)
  Stage 2 (couple): L_bc + w_world * MSE(G(s,l,pi(s,l)), D_psi(s,pi(s,l)))

See docs/VLA_FOLLOWUP_PLAN.md for the full motivation.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ActionPolicyConfig:
    token_dim: int = 2048
    token_grid: int = 8
    seq_len: int = 8
    cond_text_dim: int = 768
    hidden_dim: int = 1280
    n_layers: int = 12
    n_heads: int = 16
    dropout: float = 0.1
    use_checkpoint: bool = False
    action_dim: int = 14
    discretize_bins: int = 0  # 0 = continuous regression head; >0 = per-slot bin classification


class ActionPolicy(nn.Module):
    """pi_phi: a small Transformer that proposes actions from VGGT-token context + text.

    Input shapes:
        ctx_tokens : [B, T, P, D]   T frames × P patches × D = 2048
        cond_text  : [B, Dt]        sentence embedding from frozen CLIP
    Output:
        actions    : [B, T, A]      per-step actions (continuous) OR
                     [B, T, A, K]   per-slot bin logits if discretize_bins > 0
    """

    def __init__(self, cfg: ActionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.token_grid ** 2
        self.P = P

        self.token_proj = nn.Linear(cfg.token_dim, cfg.hidden_dim)
        self.cond_proj = nn.Linear(cfg.cond_text_dim, cfg.hidden_dim)
        # Positional encoding over T*P tokens + T action queries.
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len * P + cfg.seq_len, cfg.hidden_dim))
        nn.init.normal_(self.pos, std=0.02)
        # One learnable query per future frame.
        self.action_query = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.hidden_dim))
        nn.init.normal_(self.action_query, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

        # Output head.
        if cfg.discretize_bins > 0:
            self.out = nn.Linear(cfg.hidden_dim, cfg.action_dim * cfg.discretize_bins)
        else:
            self.out = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(
        self,
        ctx_tokens: torch.Tensor,    # [B, T, P, D]
        cond_text: torch.Tensor,     # [B, Dt]
    ) -> torch.Tensor:
        B, T, P, D = ctx_tokens.shape
        x = self.token_proj(ctx_tokens.reshape(B, T * P, D))    # [B, T*P, H]
        x = x + self.cond_proj(cond_text).unsqueeze(1)
        q = self.action_query.expand(B, -1, -1)                  # [B, T, H]
        h = torch.cat([x, q], dim=1)                              # [B, T*P + T, H]
        h = h + self.pos[:, : h.size(1)]
        if self.cfg.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in self.backbone.layers:
                h = checkpoint(layer, h, use_reentrant=False)
            if self.backbone.norm is not None:
                h = self.backbone.norm(h)
        else:
            h = self.backbone(h)
        h = self.norm(h[:, T * P:])                              # [B, T, H]
        a = self.out(h)
        if self.cfg.discretize_bins > 0:
            return a.reshape(B, T, self.cfg.action_dim, self.cfg.discretize_bins)
        return a                                                  # [B, T, A]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
