"""Treino base do modelo manual com CTC: manifesto -> modelo -> ONNX -> avaliacao.

Uso:
    python -m maosabertas_training.train --manifest corpus.csv --vocab vocab.txt --out DIR

Divide por sinalizante, treina, exporta DIR/model.onnx no contrato do runtime,
valida o ONNX contra o PyTorch, avalia o arquivo exportado nos sinalizantes de
teste e registra config e metricas em DIR/experiment.json (e no MLflow, se
instalado). DIR precisa ficar fora do repositorio.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from maosabertas_training.corpus import Clip, read_manifest, read_vocab, split_by_signer
from maosabertas_training.evaluate import evaluate, greedy_decode
from maosabertas_training.export import (
    ATOL,
    OPSET,
    check_contract,
    export_onnx,
    run_onnx,
    validate_onnx,
)
from maosabertas_training.features import FEATURE_DIM
from maosabertas_training.landmarks import check_output
from maosabertas_training.model import CausalTCN

Loader = Callable[[Path], np.ndarray]


class ClipDataset(Dataset):
    """Clipes do manifesto com features (T, FEATURE_DIM) sem NaN vindas de `load`."""

    def __init__(self, clips: list[Clip], load: Loader):
        self.clips = clips
        # shortcut: carrega tudo na memoria; trocar por carga sob demanda quando o corpus nao couber.
        self.feats = [_checked(load(c.path), c) for c in clips]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.feats[i]), torch.tensor(self.clips[i].labels)


def _checked(x: np.ndarray, clip: Clip) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"{clip.path}: features precisam ser (T, {FEATURE_DIM}), e {x.shape}")
    if np.isnan(x).any():
        raise ValueError(f"{clip.path}: NaN nas features; o loader precisa preencher as lacunas")
    # O CTC precisa de um quadro por glosa e de um branco entre glosas repetidas.
    labels = clip.labels
    need = len(labels) + sum(a == b for a, b in pairwise(labels))
    if len(x) < need:
        raise ValueError(f"{clip.path}: {len(x)} quadros nao comportam {len(labels)} glosas")
    return x


def collate(batch):
    xs, ys = zip(*batch, strict=True)
    return (
        pad_sequence(xs, batch_first=True),
        torch.tensor([len(x) for x in xs]),
        torch.cat(ys),
        torch.tensor([len(y) for y in ys]),
    )


def fit(
    model: nn.Module,
    data: ClipDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    schedule: str = "constante",
    score: Callable[[nn.Module], float] | None = None,
) -> tuple[list[float], list[float]]:
    """Treina com CTC (branco = 0). Devolve a loss media e o `score` de cada epoca.

    `schedule`: "constante" ou "cosseno" (lr decai ate 0 na ultima epoca). Com `score`
    (erro na validacao), o modelo termina com os pesos da primeira epoca de menor erro.
    O padding a direita do lote nao contamina os quadros validos porque o modelo e causal.
    """
    loader = DataLoader(
        data,
        batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    # Com o branco dominando, parte do gradiente do CTC vira subnormal e o backward da
    # convolucao fica de 5 a 15 vezes mais lento na CPU. Zerar subnormais nao muda a loss.
    torch.set_flush_denormal(True)
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if schedule == "cosseno" else None
    )
    ctc = nn.CTCLoss(blank=0)
    losses, scores = [], []
    best = None
    model.train()
    for _ in range(epochs):
        total = 0.0
        for x, x_len, y, y_len in loader:
            log_probs = model(x).log_softmax(-1).transpose(0, 1)  # CTC quer [T, B, C]
            loss = ctc(log_probs, y, x_len, y_len)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x_len)
        losses.append(total / len(data))
        if sched:
            sched.step()
        if score:
            scores.append(score(model))
            model.train()
            if scores[-1] < min(scores[:-1], default=float("inf")):
                best = copy.deepcopy(model.state_dict())
    if best is not None:
        model.load_state_dict(best)
    return losses, scores


def log_experiment(out: Path, record: dict) -> None:
    """Config e metricas em out/experiment.json; tambem no MLflow, se instalado."""
    text = json.dumps(record, indent=2, ensure_ascii=False, default=str)
    (out / "experiment.json").write_text(text, encoding="utf-8")
    try:
        import mlflow
    except ImportError:
        return
    if "MLFLOW_TRACKING_URI" not in os.environ:
        # Sem destino configurado, o registro fica junto da saida, fora do repositorio.
        # shortcut: o MLflow 3.x recusa o backend em pasta sem este opt-out; trocar por sqlite
        # (com artifact_location em `out`) quando ele deixar de existir.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri((out / "mlruns").as_uri())
    with mlflow.start_run():
        mlflow.log_params(record["config"])
        for step, loss in enumerate(record["train_loss"]):
            mlflow.log_metric("train_loss", loss, step=step)
        mlflow.log_metric("test_ger", record["metrics"]["ger"])
        mlflow.log_artifact(str(out / "experiment.json"))


def _features_loader(max_gap: int) -> Loader:
    def load(path: Path) -> np.ndarray:
        # Import tardio: so o loader real depende de fill_gaps.
        from maosabertas_training.features import features, fill_gaps

        with np.load(path) as data:
            return fill_gaps(features(data), max_gap)

    return load


def main(argv: list[str] | None = None, load: Loader | None = None) -> int:
    """`load` substitui o loader real (features + fill_gaps) nos testes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True, help="CSV clip,signer,glosses")
    parser.add_argument("--vocab", type=Path, required=True, help="uma glosa por linha")
    parser.add_argument("--out", type=Path, required=True, help="diretorio fora do repositorio")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4, help="campo receptivo 1 + 2(2^L - 1)")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="dos sinalizantes")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="dos sinalizantes de treino; se > 0, guarda a epoca de menor erro na validacao",
    )
    parser.add_argument("--schedule", choices=["constante", "cosseno"], default="constante")
    parser.add_argument(
        "--max-gap", type=int, default=5, help="fill_gaps; use o mesmo valor no tempo real"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-only", nargs="*", default=[], help="sinalizantes sempre no treino, fora do sorteio"
    )
    args = parser.parse_args(argv)
    try:
        out = check_output(args.out)
    except ValueError as e:
        parser.error(str(e))

    vocab = read_vocab(args.vocab)
    train_clips, test_clips = split_by_signer(
        read_manifest(args.manifest, vocab), args.test_fraction, args.seed, args.train_only
    )
    val_clips: list[Clip] = []
    if args.val_fraction > 0:
        # A validacao sai do treino, nunca do teste: o teste fica o mesmo com ou sem ela.
        train_clips, val_clips = split_by_signer(
            train_clips, args.val_fraction, args.seed, args.train_only
        )
    load = load or _features_loader(args.max_gap)
    train_data, test_data = ClipDataset(train_clips, load), ClipDataset(test_clips, load)
    val_data = ClipDataset(val_clips, load)

    def val_ger(m: nn.Module) -> float:
        m.eval()
        with torch.no_grad():
            hyps = [greedy_decode(m(torch.from_numpy(x)[None])[0].numpy()) for x in val_data.feats]
        triples = ((c.signer, c.labels, h) for c, h in zip(val_clips, hyps, strict=True))
        return evaluate(triples, vocab)["ger"]

    torch.manual_seed(args.seed)
    model = CausalTCN(len(vocab), args.hidden, args.layers)
    losses, val_gers = fit(
        model,
        train_data,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        args.schedule,
        val_ger if val_clips else None,
    )

    path = export_onnx(model, out / "model.onnx")
    if check_contract(path) != len(vocab):
        raise ValueError(f"{path}: numero de classes difere do vocabulario")
    diff = validate_onnx(model, path)
    # Avalia o arquivo exportado, que e o que o runtime vai rodar.
    hyps = [greedy_decode(p) for p in run_onnx(path, test_data.feats)]
    metrics = evaluate(((c.signer, c.labels, h) for c, h in zip(test_clips, hyps)), vocab)

    log_experiment(
        out,
        {
            "config": vars(args),
            "model": {
                "name": "CausalTCN",
                "lookahead_frames": 0,
                "receptive_field_frames": model.receptive_field,
                "classes": len(vocab),
            },
            "vocab": vocab,
            "train_signers": sorted({c.signer for c in train_clips}),
            "val_signers": sorted({c.signer for c in val_clips}),
            "test_signers": sorted({c.signer for c in test_clips}),
            "train_loss": losses,
            "val_ger": val_gers,
            "best_epoch": int(np.argmin(val_gers)) + 1 if val_gers else None,
            "onnx": {"path": path, "opset": OPSET, "max_abs_diff": diff, "atol": ATOL},
            "metrics": metrics,
        },
    )
    print(f"modelo: {path}\nregistro: {out / 'experiment.json'}")
    print(f"taxa de erro de glosa no teste: {metrics['ger']:.3f}")
    for signer, s in metrics["by_signer"].items():
        print(f"  {signer}: {s['ger']:.3f} ({s['clips']} clipes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
