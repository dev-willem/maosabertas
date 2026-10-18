"""Converte o .npz da extracao em JSON Lines, um quadro por linha.

Uso:
    python -m maosabertas_training.landmarks_jsonl ENTRADA.npz SAIDA.jsonl

E a entrada do exemplo em Rust `crates/maosabertas-onnx/examples/transcrever.rs`.
Cada linha: {"t", "size", "left_hand", "right_hand", "face", "blendshapes", "pose"},
com `t` e `size` como no .npz e cada parte achatada em lista. Parte nao
detectada (tudo NaN) vira `null`; NaN isolado vira `null` no valor.

A saida e dado pessoal, como o .npz: o script recusa caminho dentro do repositorio.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from maosabertas_training.landmarks import SHAPES, check_output


def _part(a: np.ndarray) -> list[float | None] | None:
    if np.isnan(a).all():
        return None
    return [None if math.isnan(v) else v for v in a.reshape(-1).tolist()]


def write_jsonl(data, out: Path) -> int:
    """`data`: arrays do .npz da extracao. Devolve o numero de quadros gravados."""
    size = [int(v) for v in data["size"]]
    parts = {k: data[k] for k in SHAPES}
    with out.open("w", encoding="utf-8") as f:
        for i, t in enumerate(data["t"].tolist()):
            row = {"t": t, "size": size} | {k: _part(a[i]) for k, a in parts.items()}
            f.write(json.dumps(row, allow_nan=False) + "\n")
    return len(data["t"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("entrada", type=Path, help=".npz da extracao")
    parser.add_argument("saida", type=Path, help="arquivo .jsonl fora do repositorio")
    args = parser.parse_args(argv)
    try:
        out = check_output(args.saida)
    except ValueError as e:
        parser.error(str(e))
    with np.load(args.entrada) as data:
        n = write_jsonl(data, out)
    print(f"{n} quadros -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
