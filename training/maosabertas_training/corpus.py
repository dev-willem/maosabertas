"""Manifesto do corpus, vocabulario e divisao por sinalizante.

Manifesto: CSV UTF-8 com cabecalho e as colunas
    clip     caminho do .npz da extracao, absoluto ou relativo ao manifesto
    signer   id do sinalizante (pseudonimo, nunca o nome)
    glosses  glosas do clipe separadas por espaco, na ordem sinalizada

Vocabulario: texto UTF-8, uma glosa por linha; linhas vazias sao ignoradas.
A classe 0 e o branco do CTC e a glosa da linha i (contando so as nao vazias)
vira a classe i.

Manifesto, vocabulario e clipes ficam fora do repositorio, junto dos dados.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import NamedTuple

BLANK = "<branco>"
COLUMNS = ("clip", "signer", "glosses")


class Clip(NamedTuple):
    path: Path
    signer: str
    labels: tuple[int, ...]


def read_vocab(path: Path) -> list[str]:
    """Devolve as classes do modelo: `[BLANK, glosa 1, glosa 2, ...]`."""
    vocab = [BLANK]
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        gloss = line.strip()
        if not gloss:
            continue
        if len(gloss.split()) > 1:
            raise ValueError(f"{path}:{n}: glosa com espaco: {gloss!r}")
        if gloss in vocab:
            raise ValueError(f"{path}:{n}: glosa repetida ou reservada: {gloss!r}")
        vocab.append(gloss)
    if len(vocab) == 1:
        raise ValueError(f"{path}: vocabulario vazio")
    return vocab


def read_manifest(path: Path, vocab: list[str]) -> list[Clip]:
    """Le o manifesto. Glosa fora do vocabulario e erro, com linha e glosa."""
    path = Path(path)
    index = {g: i for i, g in enumerate(vocab) if i}
    clips = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        missing = set(COLUMNS) - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: faltam as colunas {sorted(missing)}")
        for row in rows:
            clip, signer, glosses = ((row[k] or "").strip() for k in COLUMNS)
            if not (clip and signer and glosses):
                raise ValueError(f"{path}:{rows.line_num}: clip, signer e glosses sao obrigatorios")
            unknown = [g for g in glosses.split() if g not in index]
            if unknown:
                raise ValueError(
                    f"{path}:{rows.line_num}: glosa fora do vocabulario: {', '.join(unknown)}"
                )
            labels = tuple(index[g] for g in glosses.split())
            clips.append(Clip(path.parent / clip, signer, labels))
    if not clips:
        raise ValueError(f"{path}: manifesto vazio")
    return clips


def split_by_signer(
    clips: list[Clip], test_fraction: float, seed: int
) -> tuple[list[Clip], list[Clip]]:
    """Divide em (treino, teste) por sinalizante, nunca por clipe. Deterministica por `seed`.

    O teste recebe `round(test_fraction * sinalizantes)` sinalizantes, no minimo 1, e o
    treino nunca fica vazio.
    """
    signers = sorted({c.signer for c in clips})
    if len(signers) < 2:
        raise ValueError("divisao por sinalizante precisa de pelo menos 2 sinalizantes")
    random.Random(seed).shuffle(signers)
    test = set(signers[: min(max(1, round(test_fraction * len(signers))), len(signers) - 1)])
    return [c for c in clips if c.signer not in test], [c for c in clips if c.signer in test]
