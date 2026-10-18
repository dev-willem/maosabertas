"""Testes com dados sinteticos: nenhuma imagem de pessoa."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from maosabertas_training import landmarks as lm


def _pts(n, v=0.5):
    return [SimpleNamespace(x=v, y=v, z=v, visibility=v) for _ in range(n)]


def test_recusa_saida_dentro_do_repositorio(tmp_path):
    with pytest.raises(ValueError, match="dado pessoal"):
        lm.check_output(lm.REPO / "saida.npz")
    with pytest.raises(ValueError):
        lm.check_output(lm.REPO / "training" / ".." / "x.npz")
    assert lm.check_output(tmp_path / "x.npz") == (tmp_path / "x.npz").resolve()


def test_quadro_parcial_vira_nan_no_que_falta():
    result = SimpleNamespace(
        left_hand_landmarks=_pts(21),
        right_hand_landmarks=[],
        face_landmarks=[],
        pose_landmarks=_pts(33, 0.25),
        face_blendshapes=[SimpleNamespace(index=3, score=0.75, category_name="x")],
    )
    a = lm.frame_arrays(result)
    assert {k: v.shape for k, v in a.items()} == lm.SHAPES
    assert (a["left_hand"] == 0.5).all()
    assert np.isnan(a["right_hand"]).all() and np.isnan(a["face"]).all()
    assert (a["pose"] == 0.25).all()
    assert a["blendshapes"][3] == 0.75
    assert np.isnan(np.delete(a["blendshapes"], 3)).all()


def test_numero_de_pontos_errado_falha():
    result = SimpleNamespace(
        left_hand_landmarks=_pts(20),
        right_hand_landmarks=[],
        face_landmarks=[],
        pose_landmarks=[],
        face_blendshapes=None,
    )
    with pytest.raises(ValueError):
        lm.frame_arrays(result)


@pytest.fixture(scope="module")
def model():
    try:
        return lm.model_path()
    except OSError as e:
        pytest.skip(f"modelo indisponivel: {e}")


def test_sem_quadros_gera_arrays_vazios(model):
    data = lm.extract([], model)
    assert data["t"].shape == (0,)
    assert data["face"].shape == (0, 478, 3)


def test_recorte_em_fracoes_do_quadro():
    quadro = np.zeros((48, 64, 3), np.uint8)
    [(t, rgb)] = lm.crop([(0.5, quadro)], lm.parse_crop("0.5,0,1,0.5"))
    assert t == 0.5 and rgb.shape == (24, 32, 3)


@pytest.mark.parametrize(
    "texto",
    ["0,0,1", "a,0,1,1", "0.5,0,0.5,1", "0,.6,1,.4", "-.1,0,1,1", "0,0,1,2", "nan,0,1,1"],
)
def test_recorte_invalido_falha(texto):
    with pytest.raises(argparse.ArgumentTypeError):
        lm.parse_crop(texto)


def test_recorte_sem_pixel_falha():
    with pytest.raises(ValueError, match="vazio"):
        list(lm.crop([(0.0, np.zeros((48, 64, 3), np.uint8))], (0.5, 0.0, 0.501, 1.0)))


@pytest.mark.parametrize(
    ("recorte", "size"), [([], [64, 48]), (["--recorte", "0.5,0,1,0.5"], [32, 24])]
)
def test_video_sintetico_ponta_a_ponta(tmp_path, model, recorte, size):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "vazio.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()

    out = tmp_path / "saida.npz"
    assert lm.main([str(video), str(out), *recorte]) == 0
    data = np.load(out)
    np.testing.assert_allclose(data["t"], [0.0, 0.1, 0.2, 0.3, 0.4])
    assert data["left_hand"].shape == (5, 21, 3)
    assert np.isnan(data["face"]).all()  # quadro preto: nada detectado
    assert data["blendshape_names"].shape == (0,)
    assert data["size"].tolist() == size  # com recorte, o tamanho dele
