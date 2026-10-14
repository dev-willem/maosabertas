"""O modelo base e causal: lookahead 0 e campo receptivo conhecido."""

import torch

from maosabertas_training.features import FEATURE_DIM
from maosabertas_training.model import CausalTCN


def _model() -> CausalTCN:
    torch.manual_seed(0)
    return CausalTCN(classes=5, hidden=16, layers=3).eval()


@torch.no_grad()
def test_saida_por_quadro_e_futuro_nao_muda_o_passado():
    m = _model()
    x = torch.randn(2, 20, FEATURE_DIM)
    y = m(x)
    assert y.shape == (2, 20, 5)
    later = x.clone()
    later[:, 10:] += 1.0
    torch.testing.assert_close(m(later)[:, :10], y[:, :10])


@torch.no_grad()
def test_campo_receptivo():
    m = _model()
    rf = m.receptive_field
    assert rf == 15
    x = torch.randn(1, 40, FEATURE_DIM)
    moved = x.clone()
    moved[:, 0] += 1.0
    d = (m(moved) - m(x)).abs().amax(-1)[0]
    assert d[rf - 1] > 1e-4
    assert d[rf:].max() < 1e-6
