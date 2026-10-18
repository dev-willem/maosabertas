"""Testes com dados sinteticos: nenhuma imagem de pessoa."""

import json

import numpy as np
import pytest

from maosabertas_training import landmarks as lm
from maosabertas_training import landmarks_jsonl as lj


def _npz(path, t=2):
    data = {k: np.full((t, *s), 0.25, np.float32) for k, s in lm.SHAPES.items()}
    data["left_hand"][0] = np.nan
    data["blendshapes"][1, 3] = np.nan
    data["t"] = np.arange(t) / 30.0
    data["size"] = np.asarray([640, 480], np.int32)
    np.savez_compressed(path, **data)
    return path


def test_quadro_por_linha_com_null_no_que_falta(tmp_path):
    out = tmp_path / "x.jsonl"
    assert lj.main([str(_npz(tmp_path / "x.npz")), str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["t"] for r in rows] == [0.0, 1 / 30]
    assert rows[0]["size"] == [640, 480]
    assert rows[0]["left_hand"] is None
    assert rows[1]["left_hand"] == [0.25] * 63
    assert len(rows[0]["face"]) == 478 * 3 and len(rows[0]["pose"]) == 33 * 4
    # NaN isolado vira null no valor, nao na parte.
    assert rows[1]["blendshapes"][3] is None and rows[1]["blendshapes"][2] == 0.25


def test_recusa_saida_dentro_do_repositorio(tmp_path):
    with pytest.raises(SystemExit):
        lj.main([str(_npz(tmp_path / "x.npz")), str(lm.REPO / "x.jsonl")])
    assert not (lm.REPO / "x.jsonl").exists()
