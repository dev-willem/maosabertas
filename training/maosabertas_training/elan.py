"""Converte uma anotacao do ELAN (.eaf) em clipes, manifesto e vocabulario de treino.

Uso:
    python -m maosabertas_training.elan ANOTACAO.eaf VIDEO.npz SAIDA --sinalizante s01
        --trilha-glosa TRILHA [--trilha-enunciado TRILHA]

VIDEO.npz e a extracao do video inteiro (`landmarks`) que o .eaf anota. Com
--trilha-enunciado, cada anotacao dessa trilha vira um clipe com as glosas cujo ponto
medio cai dentro dela, em ordem de tempo; sem ela, cada glosa vira um clipe (sinal
isolado). Nenhum nome de trilha e fixo: quem define e o guia de anotacao.

As glosas sao o texto das anotacoes, so sem os espacos das pontas; nada e corrigido.
Glosa vazia ou com espaco no meio e erro, porque o manifesto separa glosas por espaco.

Tempos: TIME_VALUE (ms) mais o TIME_ORIGIN do MEDIA_DESCRIPTOR, que e onde a linha do
tempo do ELAN comeca no video. Anotacao presa a TIME_SLOT sem TIME_VALUE (subdivisao nao
alinhada) e erro, e nao interpolacao: seria inventar fronteira de sinal.

Cada clipe leva os quadros com inicio <= t < fim, com `t` reiniciado no primeiro quadro,
como sairia de um video cortado (passo 3 do docs/GUIA.md). Clipe sem quadro e pulado com
aviso. Saem SAIDA/NNNN.npz (NNNN e a ordem da anotacao no tempo), SAIDA/corpus.csv e
SAIDA/vocab.txt (em ordem alfabetica). Uma pasta por .eaf; `corpus.merge_manifests`
junta varias.
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from maosabertas_training.corpus import COLUMNS
from maosabertas_training.landmarks import SHAPES, check_output

Anotacao = tuple[float, float, str]  # inicio e fim em segundos do video, valor
POR_QUADRO = (*SHAPES, "t")  # arrays com um item por quadro; os outros vao inteiros


def _avisar(texto: str) -> None:
    print(f"aviso: {texto}", file=sys.stderr)


def _origem(raiz: ET.Element) -> int:
    """TIME_ORIGIN em ms; 0 sem ele. Valores diferentes entre midias e erro."""
    origens = {int(m.get("TIME_ORIGIN", 0)) for m in raiz.iter("MEDIA_DESCRIPTOR")}
    if len(origens) > 1:
        raise ValueError(
            f"MEDIA_DESCRIPTOR com TIME_ORIGIN diferentes {sorted(origens)}: "
            "nao da para saber de qual video e o .npz"
        )
    return origens.pop() if origens else 0


def ler_trilha(raiz: ET.Element, nome: str) -> list[Anotacao]:
    """Anotacoes da trilha `nome`, em ordem de tempo, em segundos do video."""
    trilhas = {t.get("TIER_ID", ""): t for t in raiz.iter("TIER")}
    ref = {n for n, t in trilhas.items() if t.find(".//REF_ANNOTATION") is not None}
    if nome not in trilhas or nome in ref:
        lista = ", ".join(n + " (REF_ANNOTATION)" * (n in ref) for n in sorted(trilhas))
        motivo = "usa REF_ANNOTATION, sem tempo proprio" if nome in ref else "nao existe"
        raise ValueError(f"trilha {nome!r} {motivo}; trilhas do arquivo: {lista}")
    slots = {s.get("TIME_SLOT_ID"): s.get("TIME_VALUE") for s in raiz.iter("TIME_SLOT")}
    origem = _origem(raiz)
    saida = []
    for a in trilhas[nome].iter("ALIGNABLE_ANNOTATION"):
        ms = [slots.get(a.get(c)) for c in ("TIME_SLOT_REF1", "TIME_SLOT_REF2")]
        if None in ms:
            raise ValueError(
                f"trilha {nome!r}, anotacao {a.get('ANNOTATION_ID')}: TIME_SLOT sem "
                "TIME_VALUE; alinhe a anotacao no ELAN"
            )
        valor = (a.findtext("ANNOTATION_VALUE") or "").strip()
        saida.append(((int(ms[0]) + origem) / 1000, (int(ms[1]) + origem) / 1000, valor))
    return sorted(saida)


def clipes(
    eaf: Path, trilha_glosa: str, trilha_enunciado: str | None = None
) -> list[tuple[float, float, list[str]]]:
    """(inicio, fim, glosas) de cada clipe, em segundos do video e em ordem de tempo."""
    raiz = ET.parse(eaf).getroot()
    for header in raiz.iter("HEADER"):
        if header.get("TIME_UNITS", "milliseconds") != "milliseconds":
            raise ValueError(f"{eaf}: TIME_UNITS {header.get('TIME_UNITS')!r} nao suportado")
    glosas = ler_trilha(raiz, trilha_glosa)
    for inicio, _, g in glosas:
        if len(g.split()) != 1:
            raise ValueError(f"{eaf}: glosa vazia ou com espaco em {inicio:.3f} s: {g!r}")
    if trilha_enunciado is None:
        return [(a, b, [g]) for a, b, g in glosas]
    saida, usadas = [], 0
    for a, b, _ in ler_trilha(raiz, trilha_enunciado):
        dentro = [g for ga, gb, g in glosas if a <= (ga + gb) / 2 < b]
        usadas += len(dentro)
        saida.append((a, b, dentro))
    if usadas < len(glosas):
        _avisar(f"{len(glosas) - usadas} glosas fora de qualquer enunciado ficaram de fora")
    return saida


def converter(
    eaf: Path,
    npz: Path,
    saida: Path,
    sinalizante: str,
    trilha_glosa: str,
    trilha_enunciado: str | None = None,
) -> tuple[Path, Path]:
    """Grava SAIDA/NNNN.npz, SAIDA/corpus.csv e SAIDA/vocab.txt; ver o topo."""
    saida = check_output(saida)
    with np.load(npz) as d:
        dados = {k: d[k] for k in d.files}
    t = dados["t"]
    linhas = []
    for k, (inicio, fim, glosas) in enumerate(clipes(eaf, trilha_glosa, trilha_enunciado), 1):
        trecho = f"clipe {k} [{inicio:.3f}, {fim:.3f}) s"
        dentro = (t >= inicio) & (t < fim)
        if not glosas or not dentro.any():
            _avisar(f"{trecho} pulado: {'sem glosa' if not glosas else 'sem quadro no .npz'}")
            continue
        clipe = {n: v[dentro] if n in POR_QUADRO else v for n, v in dados.items()}
        clipe["t"] = clipe["t"] - clipe["t"][0]
        nome = f"{k:04d}.npz"
        saida.mkdir(parents=True, exist_ok=True)
        parcial = saida / f"{k:04d}.parcial"
        with parcial.open("wb") as f:
            np.savez_compressed(f, **clipe)
        parcial.replace(saida / nome)  # .npz so aparece completo
        linhas.append((nome, sinalizante, glosas))
    if not linhas:
        raise ValueError(f"{eaf}: nenhum clipe com glosa e quadros")
    vocab = saida / "vocab.txt"
    todas = sorted({g for _, _, glosas in linhas for g in glosas})
    vocab.write_text("".join(f"{g}\n" for g in todas), encoding="utf-8")
    manifesto = saida / "corpus.csv"
    with manifesto.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows((nome, s, " ".join(glosas)) for nome, s, glosas in linhas)
    return manifesto, vocab


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("eaf", type=Path, help="anotacao do ELAN")
    parser.add_argument("npz", type=Path, help="landmarks do video inteiro (landmarks.py)")
    parser.add_argument("saida", type=Path, help="pasta fora do repositorio")
    parser.add_argument("--sinalizante", required=True, help="pseudonimo, nunca o nome")
    parser.add_argument("--trilha-glosa", required=True, help="TIER_ID da trilha de glosas")
    parser.add_argument("--trilha-enunciado", help="TIER_ID da trilha de enunciados")
    args = parser.parse_args(argv)
    try:
        manifesto, vocab = converter(
            args.eaf,
            args.npz,
            args.saida,
            args.sinalizante,
            args.trilha_glosa,
            args.trilha_enunciado,
        )
    except ValueError as e:
        parser.error(str(e))
    print(f"manifesto: {manifesto}\nvocabulario: {vocab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
