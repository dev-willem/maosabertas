"""A ponte Python -> Rust entrega as features do nucleo. Dados sinteticos."""

import numpy as np
import pytest

from maosabertas_training import features as ft
from maosabertas_training.landmarks import SHAPES


def _vazio(t: int) -> dict[str, np.ndarray]:
    data = {k: np.full((t, *shape), np.nan, np.float32) for k, shape in SHAPES.items()}
    data["t"] = np.arange(t, dtype=np.float64)
    data["size"] = np.array([640, 480], np.int32)
    return data


def test_quadro_com_ombros_e_quadro_sem_pose():
    data = _vazio(2)
    # Quadro 0: ombros em x = 0,25 e 0,75 com imagem 4:3, mao esquerda presente.
    data["pose"][0] = 0.5
    data["pose"][0, 11, 0] = 0.75
    data["pose"][0, 12, 0] = 0.25
    data["left_hand"][0] = [0.75, 0.5, 0.0]

    out = ft.features(data)
    assert out.shape == (2, ft.FEATURE_DIM)
    assert out.dtype == np.float32
    # Mao no ombro esquerdo: x = +0,5 larguras de ombro do centro.
    np.testing.assert_allclose(out[0, :3], [0.5, 0.0, 0.0], atol=1e-6)
    assert out[0, 63] == 1.0  # presenca da mao esquerda
    assert out[0, 127] == 0.0  # mao direita ausente
    assert np.isnan(out[1]).all()  # sem ombros, sem referencia


def test_sem_quadros():
    assert ft.features(_vazio(0)).shape == (0, ft.FEATURE_DIM)


def test_formato_errado_falha_no_rust():
    data = _vazio(1)
    data["face"] = data["face"][:, :100]
    with pytest.raises(ValueError, match="face"):
        ft.features(data)
