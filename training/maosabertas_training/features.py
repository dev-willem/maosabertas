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

    Quadro sem ombros detectados sai inteiro como NaN; o que fazer com ele e
    decisao do treino.
    """
    if len(data["t"]) == 0:
        return np.empty((0, FEATURE_DIM), np.float32)
    width, height = (float(v) for v in data["size"])
    raw = maosabertas.features(
        *(np.ascontiguousarray(data[k], "<f4").tobytes() for k in PARTS),
        width / height,
    )
    return np.frombuffer(raw, "<f4").reshape(-1, FEATURE_DIM).copy()
