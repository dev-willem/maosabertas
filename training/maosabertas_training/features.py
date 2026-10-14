"""Features por quadro, calculadas pelo nucleo em Rust: o mesmo codigo do tempo real.

O modulo `maosabertas` vem de `crates/maosabertas-py`:

    uv pip install ../crates/maosabertas-py
"""

from __future__ import annotations

from collections.abc import Mapping

import maosabertas
import numpy as np

FEATURE_DIM: int = maosabertas.FEATURE_DIM
PARTS = ("left_hand", "right_hand", "face", "blendshapes", "pose")


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
