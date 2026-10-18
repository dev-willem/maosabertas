"""Manifesto, vocabulario e divisao por sinalizante. Nomes sinteticos, nao sao glosas de Libras."""

from pathlib import Path

import pytest

from maosabertas_training import corpus as cp


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_vocabulario_reserva_o_branco(tmp_path):
    assert cp.read_vocab(_write(tmp_path / "v.txt", "A\nB\n\n")) == [cp.BLANK, "A", "B"]


@pytest.mark.parametrize("text", ["A\nA\n", "A B\n", "\n"])
def test_vocabulario_invalido(tmp_path, text):
    with pytest.raises(ValueError):
        cp.read_vocab(_write(tmp_path / "v.txt", text))


def test_manifesto(tmp_path):
    other = tmp_path / "outro" / "c2.npz"
    m = _write(tmp_path / "m.csv", f"clip,signer,glosses\nc1.npz,s1,A B A\n{other}, s2 ,B\n")
    assert cp.read_manifest(m, [cp.BLANK, "A", "B"]) == [
        cp.Clip(tmp_path / "c1.npz", "s1", (1, 2, 1)),
        cp.Clip(other, "s2", (2,)),
    ]


def test_glosa_fora_do_vocabulario_e_erro_claro(tmp_path):
    m = _write(tmp_path / "m.csv", "clip,signer,glosses\nc1.npz,s1,A\nc2.npz,s1,A C\n")
    with pytest.raises(ValueError, match=r"m\.csv:3: glosa fora do vocabulario: C"):
        cp.read_manifest(m, [cp.BLANK, "A"])


@pytest.mark.parametrize(
    "text",
    [
        "clip,signer\nc1.npz,s1\n",
        "clip,signer,glosses\nc1.npz,s1,\n",
        "clip,signer,glosses\nc1.npz,,A\n",
        "clip,signer,glosses\n",
    ],
)
def test_manifesto_invalido(tmp_path, text):
    with pytest.raises(ValueError):
        cp.read_manifest(_write(tmp_path / "m.csv", text), [cp.BLANK, "A"])


def _clips(signers: int, per: int = 3) -> list[cp.Clip]:
    return [
        cp.Clip(Path(f"{s}_{i}.npz"), f"s{s}", (1,)) for s in range(signers) for i in range(per)
    ]


def test_divisao_por_sinalizante_sem_vazamento():
    clips = _clips(10)
    train, test = cp.split_by_signer(clips, 0.2, seed=7)
    assert {c.signer for c in train}.isdisjoint({c.signer for c in test})
    assert len({c.signer for c in test}) == 2
    assert sorted(train + test) == sorted(clips)
    assert cp.split_by_signer(clips, 0.2, seed=7) == (train, test)


def test_seed_muda_a_divisao():
    clips = _clips(10)
    tests = {frozenset(c.signer for c in cp.split_by_signer(clips, 0.2, s)[1]) for s in range(5)}
    assert len(tests) > 1


def test_sinalizante_so_de_treino_nunca_vai_para_o_teste():
    clips = _clips(5) + [cp.Clip(Path("n_0.npz"), "n01", (2,))]
    for seed in range(10):
        train, test = cp.split_by_signer(clips, 0.5, seed, train_only={"n01"})
        assert "n01" in {c.signer for c in train}
        assert "n01" not in {c.signer for c in test}
        assert len({c.signer for c in test}) == 2  # round(0,5 * 5) do resto
    assert cp.split_by_signer(clips, 0.5, 3, train_only={"n01"}) == cp.split_by_signer(
        clips, 0.5, 3, train_only={"n01"}
    )


def test_juntar_manifestos_une_vocabularios(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ma = _write(tmp_path / "a" / "m.csv", "clip,signer,glosses\nx.npz,s1,AMARELO\ny.npz,s2,BALA\n")
    va = _write(tmp_path / "a" / "v.txt", "AMARELO\nBALA\n")
    mb = _write(tmp_path / "b" / "m.csv", "clip,signer,glosses\nz.npz,n01,ÁGUA AMARELO\n")
    vb = _write(tmp_path / "b" / "v.txt", "ÁGUA\nAMARELO\n")
    manifest, vocab_path = cp.merge_manifests([(ma, va), (mb, vb)], tmp_path / "junto")
    vocab = cp.read_vocab(vocab_path)
    assert vocab == [cp.BLANK, "AMARELO", "BALA", "ÁGUA"]
    assert cp.read_manifest(manifest, vocab) == [
        cp.Clip((tmp_path / "a" / "x.npz").resolve(), "s1", (1,)),
        cp.Clip((tmp_path / "a" / "y.npz").resolve(), "s2", (2,)),
        cp.Clip((tmp_path / "b" / "z.npz").resolve(), "n01", (3, 1)),
    ]


def test_divisao_precisa_de_dois_sinalizantes():
    with pytest.raises(ValueError, match="2 sinalizantes"):
        cp.split_by_signer(_clips(1), 0.5, seed=0)
    train, test = cp.split_by_signer(_clips(2), 0.9, seed=0)
    assert train and test  # treino nunca fica vazio
