"""Decodificacao CTC gulosa e taxa de erro de glosa por sinalizante e por glosa."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

KINDS = ("hits", "substitutions", "omissions", "insertions")


def greedy_decode(probs: np.ndarray, blank: int = 0) -> list[int]:
    """`probs`: (T, C). Argmax por quadro, junta repeticoes, descarta o branco.

    shortcut: copia em Python do `greedy_decode` de crates/maosabertas-core/src/ctc.rs, que e a
    referencia (empate vence a primeira classe nos dois); trocar pelo binding quando o crate
    maosabertas-py expuser a decodificacao.
    """
    best = probs.argmax(-1)
    return [int(c) for i, c in enumerate(best) if c != blank and (i == 0 or c != best[i - 1])]


def _align(ref: Sequence[int], hyp: Sequence[int]) -> list[tuple[int | None, int | None]]:
    """Alinhamento de Levenshtein: pares (ref, hyp); None do lado que falta."""
    d = [list(range(len(hyp) + 1))] + [[i] + [0] * len(hyp) for i in range(1, len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            sub = d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    pairs: list[tuple[int | None, int | None]] = []
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    return pairs[::-1]


def evaluate(samples: Iterable[tuple[str, Sequence[int], Sequence[int]]], vocab: list[str]) -> dict:
    """`samples`: (sinalizante, referencia, hipotese), em indices de classe.

    Taxa de erro de glosa (`ger`) = distancia de edicao / tamanho da referencia, no agregado e
    por sinalizante. Por glosa: acertos, substituicoes e omissoes contam na glosa da
    referencia; insercoes, na glosa inserida.
    """
    signers: dict[str, dict] = {}
    glosses = {g: dict.fromkeys(KINDS, 0) for g in vocab[1:]}
    for signer, ref, hyp in samples:
        s = signers.setdefault(signer, {"clips": 0, "ref_len": 0, "edits": 0})
        s["clips"] += 1
        s["ref_len"] += len(ref)
        for r, h in _align(ref, hyp):
            if r == h:
                kind = "hits"
            elif r is None:
                kind = "insertions"
            elif h is None:
                kind = "omissions"
            else:
                kind = "substitutions"
            glosses[vocab[h if r is None else r]][kind] += 1
            s["edits"] += kind != "hits"
    for s in signers.values():
        s["ger"] = s["edits"] / s["ref_len"]
    edits = sum(s["edits"] for s in signers.values())
    ref_len = sum(s["ref_len"] for s in signers.values())
    return {
        "ger": edits / ref_len,
        "edits": edits,
        "ref_len": ref_len,
        "by_signer": signers,
        "by_gloss": glosses,
    }
