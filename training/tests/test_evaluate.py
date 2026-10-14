"""Decodificacao gulosa e taxa de erro de glosa. Nomes sinteticos, nao sao glosas de Libras."""

import numpy as np

from maosabertas_training.corpus import BLANK
from maosabertas_training.evaluate import evaluate, greedy_decode

VOCAB = [BLANK, "A", "B", "C"]


def _probs(classes: list[int]) -> np.ndarray:
    return np.eye(len(VOCAB), dtype=np.float32)[classes]


def test_decodificacao_gulosa_junta_repeticoes_e_tira_o_branco():
    assert greedy_decode(_probs([0, 1, 1, 0, 1, 2, 2, 0])) == [1, 1, 2]
    assert greedy_decode(_probs([0, 0])) == []


def test_taxa_de_erro_por_sinalizante_e_contagens_por_glosa():
    m = evaluate([("s1", [1, 2, 3], [1, 3]), ("s2", [1], [2, 1]), ("s3", [3], [2])], VOCAB)
    assert m["by_signer"] == {
        "s1": {"clips": 1, "ref_len": 3, "edits": 1, "ger": 1 / 3},
        "s2": {"clips": 1, "ref_len": 1, "edits": 1, "ger": 1.0},
        "s3": {"clips": 1, "ref_len": 1, "edits": 1, "ger": 1.0},
    }
    assert (m["edits"], m["ref_len"], m["ger"]) == (3, 5, 3 / 5)
    assert m["by_gloss"] == {
        "A": {"hits": 2, "substitutions": 0, "omissions": 0, "insertions": 0},
        "B": {"hits": 0, "substitutions": 0, "omissions": 1, "insertions": 1},
        "C": {"hits": 1, "substitutions": 1, "omissions": 0, "insertions": 0},
    }
