"""Base unica de dados: todas as fontes num arquivo SQLite (ADR-013).

Uso:
    python -m maosabertas_training.base importar BASE.sqlite --fonte NOME \\
        --manifesto corpus.csv --licenca "CC BY 4.0" [--nao-comercial] [--credito TEXTO] \\
        [--origem anotacao]
    python -m maosabertas_training.base resumo BASE.sqlite

Uma amostra e uma gravacao ja extraida (os arrays do `.npz`, guardados como vieram,
comprimidos) com os rotulos dela em trilhas, como no ELAN:

    glosa       sequencia de glosas, na ordem sinalizada (o que o treino usa)
    traducao    traducao em portugues dada pela fonte
    ...-fonte   rotulo no vocabulario da propria fonte, nao validado pelo projeto (regra 3)

`origem` diz de onde veio o rotulo (anotacao humana, fala do video, texto na imagem):
rotulo fraco fica marcado, nao misturado. A fonte guarda licenca e se permite uso
comercial; o treino registra isso no experimento.

A base tem landmarks de pessoas reais (dado biometrico, LGPD): fica fora do
repositorio, e o modulo recusa caminho dentro dele. Video nunca entra.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np

from maosabertas_training.corpus import Clip
from maosabertas_training.landmarks import SHAPES, check_output

VERSAO = 1
# Formato do `.npz` de `landmarks.py` (HolisticLandmarker). Outros sensores usam outro nome.
HOLISTIC = "holistic"
HOLISTIC_ARRAYS = {"t", "size", *SHAPES}
NOME = re.compile(r"[A-Za-z0-9._-]+")

ESQUEMA = """
CREATE TABLE fonte (
    nome        TEXT PRIMARY KEY,
    licenca     TEXT NOT NULL,
    comercial   INTEGER NOT NULL CHECK (comercial IN (0, 1)),
    credito     TEXT NOT NULL DEFAULT '',
    importada   TEXT NOT NULL
);
-- Pseudonimo, nunca o nome. Um sinalizante pertence a uma fonte so: o mesmo id em duas
-- fontes seria tratado como a mesma pessoa na divisao por sinalizante.
CREATE TABLE sinalizante (
    id          TEXT PRIMARY KEY,
    fonte       TEXT NOT NULL REFERENCES fonte(nome)
);
CREATE TABLE amostra (
    id          INTEGER PRIMARY KEY,
    fonte       TEXT NOT NULL REFERENCES fonte(nome),
    chave       TEXT NOT NULL,
    sinalizante TEXT REFERENCES sinalizante(id),  -- NULL: a fonte nao identifica a pessoa
    formato     TEXT NOT NULL,
    quadros     INTEGER NOT NULL CHECK (quadros >= 0),
    dados       BLOB NOT NULL,
    sha256      TEXT NOT NULL,
    UNIQUE (fonte, chave)
);
CREATE TABLE rotulo (
    amostra     INTEGER NOT NULL REFERENCES amostra(id) ON DELETE CASCADE,
    trilha      TEXT NOT NULL,
    ordem       INTEGER NOT NULL,
    valor       TEXT NOT NULL,
    inicio      REAL,  -- segundos na amostra; NULL = amostra inteira
    fim         REAL,
    origem      TEXT NOT NULL,
    PRIMARY KEY (amostra, trilha, ordem)
);
CREATE INDEX rotulo_por_valor ON rotulo (trilha, valor);
"""


class Rotulo(NamedTuple):
    trilha: str
    valor: str
    inicio: float | None = None
    fim: float | None = None
    origem: str = "anotacao"


def abrir(caminho: Path, criar: bool = False) -> sqlite3.Connection:
    """Abre a base; com `criar`, cria o arquivo e o esquema se faltarem.

    Sem `criar`, base inexistente e erro: caminho digitado errado nao vira base vazia.
    Recusa caminho dentro do repositorio.
    """
    caminho = check_output(Path(caminho))
    if not (criar or caminho.is_file()):
        raise ValueError(f"{caminho}: base inexistente")
    con = sqlite3.connect(caminho)
    con.execute("PRAGMA foreign_keys = ON")
    versao = con.execute("PRAGMA user_version").fetchone()[0]
    tabelas = con.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
    if versao == 0 and tabelas == 0:
        # Esquema e versao na mesma transacao: queda no meio nao deixa base pela metade.
        con.executescript(f"BEGIN;\n{ESQUEMA}\nPRAGMA user_version = {VERSAO};\nCOMMIT;")
    elif versao != VERSAO:
        con.close()
        raise ValueError(f"{caminho}: esquema versao {versao}, este codigo le {VERSAO}")
    return con


def registrar_fonte(
    con: sqlite3.Connection, nome: str, licenca: str, comercial: bool, credito: str = ""
) -> None:
    """Cria a fonte ou confere que a ja registrada tem a mesma licenca.

    O nome vira o primeiro componente da referencia `fonte/chave` do treino: so
    letras, digitos, `-`, `_` e `.`, para a licenca nunca se perder no caminho.
    """
    if not (NOME.fullmatch(nome) and nome not in {".", ".."} and licenca.strip()):
        raise ValueError(f"fonte precisa de nome simples e licenca: {nome!r}")
    linha = con.execute("SELECT licenca, comercial FROM fonte WHERE nome = ?", (nome,)).fetchone()
    if linha is None:
        agora = datetime.now(UTC).isoformat(timespec="seconds")
        with con:
            con.execute(
                "INSERT INTO fonte VALUES (?, ?, ?, ?, ?)",
                (nome, licenca, int(comercial), credito, agora),
            )
    elif tuple(linha) != (licenca, int(comercial)):
        raise ValueError(f"fonte {nome!r} ja registrada com licenca {linha[0]!r}")


def inserir(
    con: sqlite3.Connection,
    fonte: str,
    chave: str,
    sinalizante: str | None,
    dados: bytes,
    rotulos: Iterable[Rotulo],
    formato: str = HOLISTIC,
) -> int | None:
    """Insere uma amostra (bytes de um `.npz`) e seus rotulos, numa transacao.

    Devolve o id, ou None se a chave ja existe na fonte com o mesmo conteudo
    (reimportar e seguro). Conteudo diferente na mesma chave e erro.
    """
    partes = PurePosixPath(chave).parts
    if not partes or "\\" in chave or chave.startswith("/") or ".." in partes or ":" in chave:
        raise ValueError(f"{fonte}: chave precisa ser caminho relativo com '/': {chave!r}")
    sha = hashlib.sha256(dados).hexdigest()
    antes = con.execute(
        "SELECT sha256 FROM amostra WHERE fonte = ? AND chave = ?", (fonte, chave)
    ).fetchone()
    if antes is not None:
        if antes[0] != sha:
            raise ValueError(f"{fonte}/{chave}: ja importada com outro conteudo")
        return None
    with np.load(io.BytesIO(dados)) as npz:
        faltam = HOLISTIC_ARRAYS - set(npz.files) if formato == HOLISTIC else set()
        if faltam:
            raise ValueError(f"{fonte}/{chave}: .npz sem {sorted(faltam)}")
        quadros = len(npz["t"]) if "t" in npz.files else 0
    with con:
        if sinalizante is not None:
            con.execute("INSERT OR IGNORE INTO sinalizante VALUES (?, ?)", (sinalizante, fonte))
            (dono,) = con.execute(
                "SELECT fonte FROM sinalizante WHERE id = ?", (sinalizante,)
            ).fetchone()
            if dono != fonte:
                raise ValueError(f"sinalizante {sinalizante!r} ja pertence a fonte {dono!r}")
        cur = con.execute(
            "INSERT INTO amostra (fonte, chave, sinalizante, formato, quadros, dados, sha256)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fonte, chave, sinalizante, formato, quadros, dados, sha),
        )
        ordem: dict[str, int] = {}
        for r in rotulos:
            n = ordem[r.trilha] = ordem.get(r.trilha, -1) + 1
            con.execute(
                "INSERT INTO rotulo VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cur.lastrowid, r.trilha, n, r.valor, r.inicio, r.fim, r.origem),
            )
    return cur.lastrowid


def importar_manifesto(
    con: sqlite3.Connection, fonte: str, manifesto: Path, origem: str = "anotacao"
) -> tuple[int, int]:
    """Importa um manifesto do treino (`clip,signer,glosses`, ver `corpus.py`).

    A chave de cada amostra e o caminho do clipe relativo ao manifesto. Devolve
    (novas, ja existentes).
    """
    manifesto = Path(manifesto)
    novas = existentes = 0
    with manifesto.open(encoding="utf-8-sig", newline="") as f:
        for linha in csv.DictReader(f):
            clipe, sinalizante, glosas = (
                (linha.get(k) or "").strip() for k in ("clip", "signer", "glosses")
            )
            if not (clipe and sinalizante and glosas):
                raise ValueError(f"{manifesto}: linha sem clip, signer ou glosses: {linha}")
            rotulos = [Rotulo("glosa", g, origem=origem) for g in glosas.split()]
            dados = (manifesto.parent / clipe).read_bytes()
            chave = Path(clipe).as_posix()
            if inserir(con, fonte, chave, sinalizante, dados, rotulos) is None:
                existentes += 1
            else:
                novas += 1
    return novas, existentes


def fontes(con: sqlite3.Connection) -> list[dict]:
    """Fontes registradas, com licenca e se permitem uso comercial."""
    linhas = con.execute("SELECT nome, licenca, comercial FROM fonte ORDER BY nome")
    return [{"nome": n, "licenca": lic, "comercial": bool(c)} for n, lic, c in linhas]


def _glosas(con: sqlite3.Connection, selecao: Sequence[str] | None):
    """(id, fonte, chave, sinalizante, glosa) das amostras de treino, em ordem."""
    filtro = f" AND a.fonte IN ({', '.join('?' * len(selecao))})" if selecao else ""
    return con.execute(
        "SELECT a.id, a.fonte, a.chave, a.sinalizante, r.valor"
        " FROM amostra a JOIN rotulo r ON r.amostra = a.id AND r.trilha = 'glosa'"
        f" WHERE a.formato = ? AND a.sinalizante IS NOT NULL{filtro}"
        " ORDER BY a.id, r.ordem",
        [HOLISTIC, *(selecao or [])],
    )


def vocabulario(con: sqlite3.Connection, selecao: Sequence[str] | None = None) -> list[str]:
    """Glosas das amostras de treino (holisticas, com sinalizante), na ordem em que aparecem."""
    return list(dict.fromkeys(linha[4] for linha in _glosas(con, selecao)))


def clipes(
    con: sqlite3.Connection, vocab: list[str], selecao: Sequence[str] | None = None
) -> list[Clip]:
    """Amostras de treino no formato de `corpus.read_manifest`.

    `Clip.path` e a referencia `fonte/chave`, lida por `carregar`. So entram amostras
    holisticas com sinalizante e glosa. Glosa fora de `vocab` e erro, como no manifesto.
    """
    indice = {g: i for i, g in enumerate(vocab) if i}
    out = []
    for (_, fonte, chave, sinalizante), linhas in groupby(
        _glosas(con, selecao), key=lambda linha: linha[:4]
    ):
        glosas = [linha[4] for linha in linhas]
        fora = [g for g in glosas if g not in indice]
        if fora:
            raise ValueError(f"{fonte}/{chave}: glosa fora do vocabulario: {', '.join(fora)}")
        ref = Path(fonte, *PurePosixPath(chave).parts)
        out.append(Clip(ref, sinalizante, tuple(indice[g] for g in glosas)))
    if not out:
        raise ValueError("nenhuma amostra de treino na selecao")
    return out


def carregar(con: sqlite3.Connection, ref: Path) -> dict[str, np.ndarray]:
    """Arrays da amostra `fonte/chave` (o `Clip.path` de `clipes`)."""
    fonte, *partes = Path(ref).parts
    linha = con.execute(
        "SELECT dados FROM amostra WHERE fonte = ? AND chave = ?",
        (fonte, "/".join(partes)),
    ).fetchone()
    if linha is None:
        raise KeyError(f"amostra inexistente: {ref}")
    with np.load(io.BytesIO(linha[0])) as npz:
        return {k: npz[k] for k in npz.files}


def resumo(con: sqlite3.Connection) -> list[dict]:
    """Por fonte: licenca, amostras, sinalizantes, quadros e rotulos por trilha."""
    out = []
    for f in fontes(con):
        amostras, pessoas, quadros = con.execute(
            "SELECT count(*), count(DISTINCT sinalizante), coalesce(sum(quadros), 0)"
            " FROM amostra WHERE fonte = ?",
            (f["nome"],),
        ).fetchone()
        trilhas = con.execute(
            "SELECT r.trilha, count(*), count(DISTINCT r.valor) FROM rotulo r"
            " JOIN amostra a ON a.id = r.amostra WHERE a.fonte = ? GROUP BY r.trilha",
            (f["nome"],),
        ).fetchall()
        out.append(
            f
            | {
                "amostras": amostras,
                "sinalizantes": pessoas,
                "quadros": quadros,
                "trilhas": {t: {"rotulos": n, "valores": v} for t, n, v in trilhas},
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="comando", required=True)
    imp = sub.add_parser("importar", help="importa um manifesto do treino")
    imp.add_argument("base", type=Path)
    imp.add_argument("--fonte", required=True)
    imp.add_argument("--manifesto", type=Path, required=True)
    imp.add_argument("--licenca", required=True)
    imp.add_argument("--nao-comercial", action="store_true", help="a licenca proibe uso comercial")
    imp.add_argument("--credito", default="")
    imp.add_argument("--origem", default="anotacao", help="de onde vieram os rotulos")
    res = sub.add_parser("resumo", help="mostra o que a base tem")
    res.add_argument("base", type=Path)
    args = parser.parse_args(argv)

    try:
        con = abrir(args.base, criar=args.comando == "importar")
    except ValueError as e:
        parser.error(str(e))
    try:
        if args.comando == "importar":
            registrar_fonte(con, args.fonte, args.licenca, not args.nao_comercial, args.credito)
            novas, existentes = importar_manifesto(con, args.fonte, args.manifesto, args.origem)
            print(f"{args.fonte}: {novas} amostras novas, {existentes} ja existentes")
    except ValueError as e:
        # Cada amostra e uma transacao: as anteriores ficam, e reimportar continua delas.
        con.close()
        parser.error(str(e))
    try:
        for f in resumo(con):
            uso = "comercial" if f["comercial"] else "NAO comercial"
            print(
                f"{f['nome']:<16} {f['licenca']:<16} {uso:<13} {f['amostras']:>6} amostras"
                f" {f['sinalizantes']:>3} sinalizantes {f['quadros']:>8} quadros"
            )
            for t, n in sorted(f["trilhas"].items()):
                print(f"{'':16} {t:<20} {n['rotulos']:>7} rotulos, {n['valores']} valores")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
