"""Conversor ELAN -> manifesto. Nomes sinteticos, nao sao glosas de Libras."""

import numpy as np
import pytest

from maosabertas_training.corpus import read_manifest, read_vocab
from maosabertas_training.elan import converter, main
from maosabertas_training.landmarks import REPO, SHAPES

# Trilha so com tempo herdado (REF_ANNOTATION), como uma de traducao.
REF = (
    '<TIER TIER_ID="Ref" LINGUISTIC_TYPE_REF="ref" PARENT_REF="G"><ANNOTATION>'
    '<REF_ANNOTATION ANNOTATION_ID="r1" ANNOTATION_REF="a2"><ANNOTATION_VALUE>x'
    "</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION></TIER>"
)


def _eaf(path, trilhas, origem=None, extra=""):
    """.eaf minimo. `trilhas`: nome -> [(ms inicial, ms final, valor)]; None e slot sem tempo."""
    slots, tiers = [], []
    for nome, anotacoes in trilhas.items():
        corpo = ""
        for a, b, valor in anotacoes:
            ids = []
            for ms in (a, b):
                ids.append(f"ts{len(slots) + 1}")
                tempo = "" if ms is None else f' TIME_VALUE="{ms}"'
                slots.append(f'<TIME_SLOT TIME_SLOT_ID="{ids[-1]}"{tempo}/>')
            corpo += (
                f'<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a{len(slots)}" '
                f'TIME_SLOT_REF1="{ids[0]}" TIME_SLOT_REF2="{ids[1]}">'
                f"<ANNOTATION_VALUE>{valor}</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>"
            )
        tiers.append(f'<TIER TIER_ID="{nome}" LINGUISTIC_TYPE_REF="lt">{corpo}</TIER>')
    attr = "" if origem is None else f' TIME_ORIGIN="{origem}"'
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<ANNOTATION_DOCUMENT FORMAT="3.0" VERSION="3.0">'
        '<HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">'
        f'<MEDIA_DESCRIPTOR MEDIA_URL="file:///video.mp4" MIME_TYPE="video/mp4"{attr}/></HEADER>'
        f"<TIME_ORDER>{''.join(slots)}</TIME_ORDER>{''.join(tiers)}{extra}</ANNOTATION_DOCUMENT>",
        encoding="utf-8",
    )
    return path


def _npz(path, quadros):
    """Video sintetico a 10 fps; left_hand[:, 0, 0] guarda o `t` original de cada quadro."""
    data = {k: np.full((quadros, *shape), np.nan, np.float32) for k, shape in SHAPES.items()}
    data["t"] = np.arange(quadros) / 10
    data["left_hand"][:, 0, 0] = data["t"]
    data["blendshape_names"] = np.asarray([f"b{i}" for i in range(52)])
    data["size"] = np.array([640, 480], np.int32)
    np.savez(path, **data)
    return path


def _ler(saida):
    vocab = read_vocab(saida / "vocab.txt")
    return vocab, read_manifest(saida / "corpus.csv", vocab)


def test_cada_glosa_vira_um_clipe(tmp_path, capsys):
    # A terceira glosa cai depois do fim do video: sem quadro, e pulada.
    eaf = _eaf(tmp_path / "v.eaf", {"G": [(500, 800, " B "), (0, 300, "A"), (9000, 9500, "C")]})
    saida = tmp_path / "saida"
    converter(eaf, _npz(tmp_path / "v.npz", 20), saida, "s01", "G")

    vocab, clips = _ler(saida)
    assert vocab == ["<branco>", "A", "B"]
    assert [(c.path.name, c.signer, c.labels) for c in clips] == [
        ("0001.npz", "s01", (1,)),
        ("0002.npz", "s01", (2,)),
    ]
    assert "clipe 3" in capsys.readouterr().err
    with np.load(clips[1].path) as d:
        np.testing.assert_allclose(d["left_hand"][:, 0, 0], [0.5, 0.6, 0.7])
        np.testing.assert_allclose(d["t"], [0.0, 0.1, 0.2])
        assert {k: d[k].shape[1:] for k in SHAPES} == SHAPES
        assert d["blendshape_names"][3] == "b3" and list(d["size"]) == [640, 480]


def test_enunciado_junta_as_glosas_em_ordem_de_tempo(tmp_path):
    # TIME_ORIGIN 500: o zero do ELAN e 0,5 s no video.
    eaf = _eaf(
        tmp_path / "v.eaf",
        {
            "G": [(600, 900, "B"), (100, 400, "A"), (1200, 1500, "C")],
            "E": [(0, 1000, ""), (1100, 1600, "")],
        },
        origem=500,
    )
    saida = tmp_path / "saida"
    converter(eaf, _npz(tmp_path / "v.npz", 30), saida, "s01", "G", "E")

    vocab, clips = _ler(saida)
    assert [[vocab[i] for i in c.labels] for c in clips] == [["A", "B"], ["C"]]
    with np.load(clips[0].path) as d:
        np.testing.assert_allclose(d["left_hand"][[0, -1], 0, 0], [0.5, 1.4])
    with np.load(clips[1].path) as d:
        np.testing.assert_allclose(d["left_hand"][[0, -1], 0, 0], [1.6, 2.0])


def test_trilha_inexistente_ou_sem_tempo_proprio_e_erro(tmp_path):
    eaf = _eaf(tmp_path / "v.eaf", {"G": [(0, 300, "A")]}, extra=REF)
    npz = _npz(tmp_path / "v.npz", 5)
    with pytest.raises(ValueError, match=r"'Glosa' nao existe; trilhas do arquivo: G, Ref"):
        converter(eaf, npz, tmp_path / "saida", "s01", "Glosa")
    with pytest.raises(ValueError, match="'Ref' usa REF_ANNOTATION"):
        converter(eaf, npz, tmp_path / "saida", "s01", "G", "Ref")


def test_slot_sem_tempo_e_erro(tmp_path):
    eaf = _eaf(tmp_path / "v.eaf", {"G": [(0, None, "A")]})
    with pytest.raises(ValueError, match="sem TIME_VALUE"):
        converter(eaf, _npz(tmp_path / "v.npz", 5), tmp_path / "saida", "s01", "G")


def test_glosa_com_espaco_e_erro(tmp_path):
    eaf = _eaf(tmp_path / "v.eaf", {"G": [(0, 300, "A B")]})
    with pytest.raises(ValueError, match="'A B'"):
        converter(eaf, _npz(tmp_path / "v.npz", 5), tmp_path / "saida", "s01", "G")


def test_saida_dentro_do_repositorio_e_recusada(tmp_path):
    eaf = _eaf(tmp_path / "v.eaf", {"G": [(0, 300, "A")]})
    npz = _npz(tmp_path / "v.npz", 5)
    with pytest.raises(SystemExit):
        main([str(eaf), str(npz), str(REPO / "corpus-elan"), "--sinalizante", "s01",
              "--trilha-glosa", "G"])  # fmt: skip
