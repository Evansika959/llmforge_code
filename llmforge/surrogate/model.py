from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FIELD_COUNT as _DEFAULT_FIELD_COUNT


class ArchTransformerRanker(nn.Module):
    def __init__(
        self,
        max_layers: int = 40,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        field_count: Optional[int] = None,
    ):
        super().__init__()
        self.max_layers = max_layers
        self.field_count = _DEFAULT_FIELD_COUNT if field_count is None else field_count
        # one linear projection per field (numeric input normalized 0..1)
        self.field_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.field_count)])
        self.layer_pos = nn.Embedding(max_layers, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, F] of normalized floats in [0,1]
        padding_mask: [B, L] bool, True = padding (ignore this position)
        returns scores: [B]
        """
        b, l, f = x.shape
        assert f == self.field_count, f"Expected {self.field_count} fields, got {f}"
        assert l <= self.max_layers, f"Input has {l} layers but max_layers={self.max_layers}"

        emb = 0
        for i, proj in enumerate(self.field_proj):
            emb = emb + proj(x[:, :, i:i+1])
        pos = self.layer_pos(torch.arange(l, device=x.device))  # [L, d_model]
        token = emb + pos  # [B, L, d_model]
        token = token.transpose(0, 1)  # [L, B, d_model]

        enc = self.encoder(token, src_key_padding_mask=padding_mask)  # [L, B, d_model]
        enc = enc.transpose(0, 1)  # [B, L, d_model]

        # Masked mean pooling: only pool over non-padding positions
        if padding_mask is not None:
            # padding_mask: [B, L] True = ignore
            real_mask = ~padding_mask  # [B, L] True = real
            real_mask_f = real_mask.unsqueeze(-1).float()  # [B, L, 1]
            enc = enc * real_mask_f  # zero out padding
            pooled = enc.sum(dim=1) / real_mask_f.sum(dim=1).clamp(min=1.0)  # [B, d_model]
        else:
            pooled = enc.mean(dim=1)  # [B, d_model]

        score = self.head(pooled).squeeze(-1)
        return score

    @staticmethod
    def pairwise_loss(score_a: torch.Tensor, score_b: torch.Tensor, label: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        score_a, score_b: [B]
        label: 1 if a better than b (lower loss), else 0
        weight: optional per-sample weight (0 to drop ties)
        """
        logits = score_b - score_a  # higher when a better (label=1)
        loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
        if weight is not None:
            loss = loss * weight
        return loss.mean()

    @staticmethod
    def absolute_loss(score: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """
        score: [B] predicted scalar (lower is better)
        target: [B] ground-truth scalar
        weight: optional per-sample weight
        """
        loss = torch.abs(score - target)
        if weight is not None:
            loss = loss * weight
        return loss.mean()
