"""Treino ponta a ponta com dados sinteticos: um padrao por classe, nenhuma pessoa."""

import json
from pathlib import Path

import numpy as np
import pytest

from maosabertas_training.corpus import Clip
from maosabertas_training.export import ATOL, check_contract
from maosabertas_training.features import FEATURE_DIM
from maosabertas_training.landmarks import REPO
from maosabertas_training.train import ClipDataset, main

GLOSSES = ["A", "B", "C"]


def _clip(rng: np.random.Generator, labels) -> np.ndarray:
    """Cada classe acende um bloco de features por 6 quadros; entre sinais, so ruido."""
    parts = []
    for k in labels:
        parts.append(rng.normal(0, 0.1, (3, FEATURE_DIM)))
        sign = rng.normal(0, 0.1, (6, FEATURE_DIM))
        sign[:, 10 * k : 10 * (k + 1)] += 1.0
        parts.append(sign)
    parts.append(rng.normal(0, 0.1, (3, FEATURE_DIM)))
    return np.concatenate(parts).astype(np.float32)


def _corpus(tmp_path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    data, rows = {}, ["clip,signer,glosses"]
    for s in range(4):
        for i in range(6):
            labels = rng.integers(1, len(GLOSSES) + 1, rng.integers(1, 4))
            name = f"s{s}_{i}.npz"
            data[name] = _clip(rng, labels)
            rows.append(f"{name},s{s},{' '.join(GLOSSES[k - 1] for k in labels)}")
    (tmp_path / "m.csv").write_text("\n".join(rows), encoding="utf-8")
    (tmp_path / "v.txt").write_text("\n".join(GLOSSES), encoding="utf-8")
    return data


def _args(tmp_path: Path, out: Path) -> list[str]:
    return [
        "--manifest", str(tmp_path / "m.csv"),
        "--vocab", str(tmp_path / "v.txt"),
        "--out", str(out),
        "--epochs", "15",
        "--hidden", "32",
        "--layers", "3",
        "--test-fraction", "0.25",
    ]  # fmt: skip


def test_treino_ponta_a_ponta(tmp_path):
    data = _corpus(tmp_path)
    out = tmp_path / "saida"
    assert main(_args(tmp_path, out), load=lambda p: data[p.name]) == 0

    rec = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert rec["train_loss"][-1] < rec["train_loss"][0]
    assert check_contract(out / "model.onnx") == len(GLOSSES) + 1
    assert rec["onnx"]["max_abs_diff"] <= ATOL
    assert set(rec["train_signers"]).isdisjoint(rec["test_signers"])
    assert set(rec["metrics"]["by_signer"]) == set(rec["test_signers"])
    assert set(rec["metrics"]["by_gloss"]) == set(GLOSSES)


def test_saida_no_repositorio_e_recusada(tmp_path):
    data = _corpus(tmp_path)
    with pytest.raises(SystemExit):
        main(_args(tmp_path, REPO / "saida"), load=lambda p: data[p.name])


@pytest.mark.parametrize(
    ("x", "labels"),
    [
        (np.full((4, FEATURE_DIM), np.nan, np.float32), (1,)),
        (np.zeros((4, 10), np.float32), (1,)),
        (np.zeros((2, FEATURE_DIM), np.float32), (1, 1)),  # CTC precisa de 3 quadros
    ],
)
def test_dataset_recusa_nan_formato_errado_e_clipe_curto(x, labels):
    with pytest.raises(ValueError):
        ClipDataset([Clip(Path("c.npz"), "s", labels)], lambda p: x)
    ClipDataset([Clip(Path("c.npz"), "s", (1, 1))], lambda p: np.zeros((3, FEATURE_DIM)))


def test_loader_real_le_npz_de_landmarks(tmp_path):
    from maosabertas_training.landmarks import SHAPES
    from maosabertas_training.train import _features_loader

    # Landmarks sinteticos: ombros nos 2 primeiros quadros, nada nos 2 ultimos.
    data = {k: np.full((4, *shape), np.nan, np.float32) for k, shape in SHAPES.items()}
    data["pose"][:2] = 0.5
    data["pose"][:2, 11, 0] = 0.75
    data["pose"][:2, 12, 0] = 0.25
    data["t"] = np.arange(4, dtype=np.float64)
    data["size"] = np.array([640, 480], np.int32)
    path = tmp_path / "clip.npz"
    np.savez(path, **data)

    feats = _features_loader(max_gap=1)(path)
    assert feats.shape == (4, FEATURE_DIM)
    assert not np.isnan(feats).any()
    np.testing.assert_array_equal(feats[2], feats[1])  # lacuna curta repete
    assert (feats[3] == 0).all()  # alem de max_gap, zera
