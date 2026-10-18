"""Modelo manual base: convolucoes temporais causais e dilatadas, saida por quadro.

Lookahead: 0 quadros. A saida no quadro t so depende dos quadros <= t, entao o
modelo roda em fluxo sem esperar o futuro. Campo receptivo:
1 + (KERNEL - 1) * (2**layers - 1) quadros (31 com as 4 camadas padrao).
Ver docs/adr/0008-treino-base.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from maosabertas_training.features import FEATURE_DIM

KERNEL = 3


class CausalTCN(nn.Module):
    def __init__(self, classes: int, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.inp = nn.Conv1d(FEATURE_DIM, hidden, 1)
        self.convs = nn.ModuleList(
            nn.Conv1d(hidden, hidden, KERNEL, dilation=2**i) for i in range(layers)
        )
        self.out = nn.Conv1d(hidden, classes, 1)
        self.receptive_field = 1 + (KERNEL - 1) * (2**layers - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, FEATURE_DIM] -> logits [B, T, C]."""
        h = self.inp(x.transpose(1, 2))
        for conv in self.convs:
            # Padding so a esquerda: o quadro t nunca ve o futuro. Replicando o primeiro quadro,
            # e nao com zeros: a borda de zeros virava um atalho para o CTC emitir a glosa no
            # quadro 0, que so tem a pose de repouso (experimento 001).
            pad = (KERNEL - 1) * conv.dilation[0]
            h = h + torch.relu(conv(F.pad(h, (pad, 0), mode="replicate")))
        return self.out(h).transpose(1, 2)
