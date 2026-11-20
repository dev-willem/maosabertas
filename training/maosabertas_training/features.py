"""Features por quadro, calculadas pelo nucleo em Rust: o mesmo codigo do tempo real.

O modulo `maosabertas` vem de `crates/maosabertas-py`:

    uv pip install ../crates/maosabertas-py
"""

from __future__ import annotations

from collections.abc import Mapping

import maosabertas
import numpy as np

FEATURE_DIM: int = maosabertas.FEATURE_DIM
HANDSHAPE_DIM: int = maosabertas.HANDSHAPE_DIM
PARTS = ("left_hand", "right_hand", "face", "blendshapes", "pose")
HANDS = ("left_hand", "right_hand")


def _bytes(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, "<f4").tobytes()


def _aspect(data: Mapping[str, np.ndarray]) -> float:
    width, height = (float(v) for v in data["size"])
    return width / height


def handshape(data: Mapping[str, np.ndarray], hand: str) -> np.ndarray:
    """Descritor de configuracao de mao do nucleo (ADR-009), (T, HANDSHAPE_DIM) float32.

    `hand` e "left_hand" ou "right_hand". Quadro sem a mao sai inteiro como NaN.
    """
    if hand not in HANDS:
        raise ValueError(f"mao desconhecida: {hand!r}; use {HANDS}")
    raw = maosabertas.handshape(_bytes(data[hand]), _aspect(data))
    return np.frombuffer(raw, "<f4").reshape(-1, HANDSHAPE_DIM).copy()


def holds(data: Mapping[str, np.ndarray], fps: float) -> np.ndarray:
    """Quadros de "segurar" do nucleo, (T, 2) bool: mao (esquerda, direita) parada e fora
    do repouso desde o quadro anterior. Sem ombros ou sem a mao, False.
    """
    raw = maosabertas.holds(*(_bytes(data[k]) for k in (*HANDS, "pose")), _aspect(data), float(fps))
    return np.frombuffer(raw, np.uint8).reshape(-1, 2).astype(bool)


def features(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """`data`: arrays do .npz da extracao. Devolve (T, FEATURE_DIM) float32.

    Quadro sem ombros detectados sai inteiro como NaN; `fill_gaps` aplica a
    regra do tempo real a ele.
    """
    if len(data["t"]) == 0:
        return np.empty((0, FEATURE_DIM), np.float32)
    width, height = (float(v) for v in data["size"])
    raw = maosabertas.features(
        *(np.ascontiguousarray(data[k], "<f4").tobytes() for k in PARTS),
        width / height,
    )
    return np.frombuffer(raw, "<f4").reshape(-1, FEATURE_DIM).copy()


def fill_gaps(feats: np.ndarray, max_gap: int) -> np.ndarray:
    """Quadros sem referencia (linhas NaN de `features`) pela regra do nucleo.

    Os primeiros `max_gap` quadros de uma lacuna repetem o ultimo quadro valido;
    os seguintes, e os anteriores ao primeiro quadro valido, viram zeros. Devolve
    (T, FEATURE_DIM) float32 sem NaN. Ver `crates/maosabertas-core/src/gaps.rs`.
    """
    if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
        raise ValueError(f"esperado (T, FEATURE_DIM={FEATURE_DIM}), recebido {feats.shape}")
    raw = maosabertas.fill_gaps(np.ascontiguousarray(feats, "<f4").tobytes(), max_gap)
    return np.frombuffer(raw, "<f4").reshape(-1, FEATURE_DIM).copy()
