import numpy as np

from maosabertas_training.camera import FPS, Live, framing
from maosabertas_training.features import FEATURE_DIM

# Vocabulario sintetico: os nomes nao sao glosas de Libras.
VOCAB = ["<branco>", "A", "B"]
NAN = np.full(FEATURE_DIM, np.nan, np.float32)


def fake_run(x: np.ndarray) -> np.ndarray:
    """Modelo falso: vence, com certeza, a classe escrita na primeira feature do quadro."""
    probs = np.zeros((len(x), len(VOCAB)), np.float32)
    probs[np.arange(len(x)), x[:, 0].astype(int)] = 1
    return probs


def row(c: int) -> np.ndarray:
    r = np.zeros(FEATURE_DIM, np.float32)
    r[0] = c
    return r


def push_all(live: Live, rows: list[np.ndarray], hand: bool = True) -> list[str]:
    """Um quadro de camera meio quadro depois de cada tique."""
    return [g for i, r in enumerate(rows) for g in live.push((i + 0.5) / FPS, r, hand)]


def test_junta_repeticoes_descarta_branco_e_segue_o_relogio_do_treino():
    live = Live(fake_run, VOCAB, max_gap=2, receptive_field=3)
    out = push_all(live, [row(1), row(1), row(0), row(1)])
    # Camera lenta: o quadro de B cobre dois tiques; o trecho so fecha no branco seguinte.
    out += live.push(5.5 / FPS, row(2), True)
    assert out == ["A", "A"]
    out += live.push(6.5 / FPS, row(0), True)
    assert out == ["A", "A", "B"]
    assert live.ticks == 7


def test_quadro_sem_ombros_segue_a_regra_do_nucleo():
    live = Live(fake_run, VOCAB, max_gap=2, receptive_field=3)
    # Lacuna de 2 quadros repete o A; a de 3 vira zeros (branco) no terceiro.
    seq = [row(1), NAN, NAN, row(1), NAN, NAN, NAN, row(1), row(0)]
    assert push_all(live, seq) == ["A", "A"]


def test_camera_adiantada_nao_gera_quadro():
    live = Live(fake_run, VOCAB, max_gap=2, receptive_field=3)
    assert live.push(0.1 / FPS, row(1), True) == []
    assert live.push(0.2 / FPS, row(2), True) == []
    assert live.ticks == 1


def test_trecho_sem_mao_ou_com_pouca_confianca_nao_vira_legenda():
    live = Live(fake_run, VOCAB, max_gap=2, receptive_field=3)
    assert push_all(live, [row(1), row(0)], hand=False) == []

    def weak_run(x: np.ndarray) -> np.ndarray:
        probs = np.tile(np.float32([1, 0, 0]), (len(x), 1))
        probs[x[:, 0] == 1] = [0.4, 0.6, 0]
        return probs

    weak = Live(weak_run, VOCAB, max_gap=2, receptive_field=3)
    assert push_all(weak, [row(1), row(0)]) == []


def test_dica_de_enquadramento():
    pose = np.full((33, 4), np.nan, np.float32)
    assert framing(pose, 16 / 9) == "fique com os ombros no quadro"
    # Ombros na mediana do corpus de treino: nada a dizer.
    pose[11, :2], pose[12, :2] = (0.5 + 0.0886, 0.45), (0.5 - 0.0886, 0.45)
    assert framing(pose, 16 / 9) is None
    # Como na camera do usuario: ombros largos e baixos.
    pose[11, :2], pose[12, :2] = (0.62, 0.92), (0.28, 0.92)
    assert framing(pose, 16 / 9) == "afaste-se da camera"
