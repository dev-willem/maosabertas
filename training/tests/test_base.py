"""Base unica em SQLite: importacao, selecao para o treino e garantias. Dados sinteticos."""

import csv
import io
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from maosabertas_training import base
from maosabertas_training.corpus import BLANK
from maosabertas_training.landmarks import SHAPES

REPO = Path(__file__).resolve().parents[2]


def _npz(path: Path, quadros: int, valor: float = 0.0) -> None:
    """`.npz` no formato da extracao, com todas as partes valendo `valor`."""
    partes = {k: np.full((quadros, *s), valor, np.float32) for k, s in SHAPES.items()}
    np.savez_compressed(
        path, t=np.arange(quadros, dtype=np.float64), size=np.array([640, 480]), **partes
    )


def _manifesto(pasta: Path, linhas: list[tuple[str, str, str]]) -> Path:
    (pasta / "s").mkdir(parents=True, exist_ok=True)
    for i, (clipe, _, _) in enumerate(linhas):
        _npz(pasta / clipe, quadros=3 + i, valor=i)
    m = pasta / "corpus.csv"
    with m.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("clip", "signer", "glosses"))
        w.writerows(linhas)
    return m


@pytest.fixture
def con(tmp_path):
    c = base.abrir(tmp_path / "base.sqlite", criar=True)
    yield c
    c.close()


def test_importa_e_entrega_ao_treino(con, tmp_path):
    m = _manifesto(tmp_path / "a", [("s/1.npz", "p1", "A B"), ("s/2.npz", "p2", "B")])
    base.registrar_fonte(con, "fa", "CC BY 4.0", comercial=True)
    assert base.importar_manifesto(con, "fa", m) == (2, 0)
    # Reimportar o mesmo conteudo nao duplica.
    assert base.importar_manifesto(con, "fa", m) == (0, 2)

    vocab = [BLANK, *base.vocabulario(con)]
    assert vocab == [BLANK, "A", "B"]
    clips = base.clipes(con, vocab)
    assert [(c.signer, c.labels) for c in clips] == [("p1", (1, 2)), ("p2", (2,))]
    dados = base.carregar(con, clips[1].path)
    assert dados["t"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert (dados["pose"] == 1).all()

    r = base.resumo(con)
    assert r[0]["amostras"] == 2 and r[0]["sinalizantes"] == 2 and r[0]["quadros"] == 7
    assert r[0]["trilhas"]["glosa"] == {"rotulos": 3, "valores": 2}


def test_selecao_por_fonte_e_formato(con, tmp_path):
    base.registrar_fonte(con, "fa", "CC BY 4.0", comercial=True)
    base.registrar_fonte(con, "fb", "CC BY-NC 4.0", comercial=False)
    base.importar_manifesto(con, "fa", _manifesto(tmp_path / "a", [("s/1.npz", "p1", "A")]))
    base.importar_manifesto(con, "fb", _manifesto(tmp_path / "b", [("s/1.npz", "q1", "C")]))
    # Outro sensor e amostra sem pessoa identificada nao entram no treino.
    buf = io.BytesIO()
    np.savez_compressed(buf, t=np.arange(2.0))
    rotulo = base.Rotulo("movimento-fonte", "classe-3")
    base.inserir(con, "fb", "outro/1", None, buf.getvalue(), [rotulo], formato="trajetoria")

    assert base.vocabulario(con, ["fb"]) == ["C"]
    assert [c.signer for c in base.clipes(con, [BLANK, "A", "C"])] == ["p1", "q1"]
    assert [f["comercial"] for f in base.fontes(con)] == [True, False]
    with pytest.raises(ValueError, match="fora do vocabulario"):
        base.clipes(con, [BLANK, "A"])
    with pytest.raises(ValueError, match="nenhuma amostra"):
        base.clipes(con, [BLANK, "A"], ["nao-existe"])


def test_recusa_conflitos(con, tmp_path):
    base.registrar_fonte(con, "fa", "CC BY 4.0", comercial=True)
    with pytest.raises(ValueError, match="ja registrada"):
        base.registrar_fonte(con, "fa", "CC BY-NC 4.0", comercial=False)

    m = _manifesto(tmp_path / "a", [("s/1.npz", "p1", "A")])
    base.importar_manifesto(con, "fa", m)
    _npz(tmp_path / "a" / "s" / "1.npz", quadros=9)
    with pytest.raises(ValueError, match="outro conteudo"):
        base.importar_manifesto(con, "fa", m)

    # O mesmo pseudonimo em duas fontes seria a mesma pessoa na divisao: recusa.
    base.registrar_fonte(con, "fb", "CC BY 4.0", comercial=True)
    with pytest.raises(ValueError, match="ja pertence"):
        base.importar_manifesto(con, "fb", _manifesto(tmp_path / "b", [("s/1.npz", "p1", "A")]))
    assert base.resumo(con)[1]["amostras"] == 0  # a transacao falhou inteira

    with pytest.raises(KeyError):
        base.carregar(con, Path("fa", "s", "nao-existe.npz"))


@pytest.mark.parametrize("nome", ["a/b", "a\\b", "C:", ".", "..", "", "com espaco"])
def test_nome_de_fonte_precisa_ser_simples(con, nome):
    # O nome e o primeiro componente da referencia do treino; nao pode se partir.
    with pytest.raises(ValueError, match="nome simples"):
        base.registrar_fonte(con, nome, "CC BY 4.0", comercial=True)


@pytest.mark.parametrize("chave", ["../x.npz", "/abs.npz", "a\\b.npz", "C:/x.npz", ""])
def test_chave_precisa_ser_caminho_relativo(con, tmp_path, chave):
    base.registrar_fonte(con, "fa", "CC BY 4.0", comercial=True)
    _npz(tmp_path / "x.npz", 2)
    with pytest.raises(ValueError, match="chave"):
        base.inserir(con, "fa", chave, "p1", (tmp_path / "x.npz").read_bytes(), [])


def test_npz_holistico_incompleto_e_recusado(con):
    base.registrar_fonte(con, "fa", "CC BY 4.0", comercial=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, t=np.arange(2.0))
    with pytest.raises(ValueError, match="sem"):
        base.inserir(con, "fa", "x.npz", "p1", buf.getvalue(), [])


def test_base_fica_fora_do_repositorio_e_versao_e_conferida(tmp_path):
    with pytest.raises(ValueError, match="repositorio"):
        base.abrir(REPO / "base.sqlite", criar=True)
    caminho = tmp_path / "base.sqlite"
    # Caminho digitado errado nao vira base vazia.
    with pytest.raises(ValueError, match="inexistente"):
        base.abrir(caminho)
    assert not caminho.exists()
    base.abrir(caminho, criar=True).close()
    base.abrir(caminho).close()
    with sqlite3.connect(caminho) as c:
        c.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="versao 99"):
        base.abrir(caminho)


def test_linha_de_comando(tmp_path, capsys):
    m = _manifesto(tmp_path / "a", [("s/1.npz", "p1", "A")])
    args = ["importar", str(tmp_path / "b.sqlite"), "--fonte", "fa", "--manifesto", str(m)]
    assert base.main([*args, "--licenca", "CC BY-NC 4.0", "--nao-comercial"]) == 0
    saida = capsys.readouterr().out
    assert "fa: 1 amostras novas" in saida and "NAO comercial" in saida
    assert base.main(["resumo", str(tmp_path / "b.sqlite")]) == 0
    assert "glosa" in capsys.readouterr().out
    # Erros viram mensagem da linha de comando, nao traceback.
    with pytest.raises(SystemExit):
        base.main([*args, "--licenca", "CC BY 4.0"])  # licenca diferente da registrada
    with pytest.raises(SystemExit):
        base.main(["resumo", str(tmp_path / "nao-existe.sqlite")])
