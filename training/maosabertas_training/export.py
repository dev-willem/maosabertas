"""Exportacao para ONNX no contrato do runtime e validacao contra o PyTorch.

Contrato (docs/adr/0005 e crates/maosabertas-onnx): uma entrada float32
[T, FEATURE_DIM], T dinamico, sem batch; uma saida float32 [T, C], C fixo, ja
em probabilidades (softmax). Classe 0 e o branco do CTC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from maosabertas_training.features import FEATURE_DIM
from maosabertas_training.landmarks import check_output

OPSET = 17
# Maior diferenca absoluta aceita entre probabilidades do ONNX Runtime e do PyTorch.
ATOL = 1e-5


class _Probs(nn.Module):
    """Adapta o modelo ao contrato: [T, FEATURE_DIM] -> probabilidades [T, C]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x[None])[0], -1)


def export_onnx(model: nn.Module, path: Path) -> Path:
    """Grava o .onnx fora do repositorio (o modelo carrega o que aprendeu dos dados)."""
    path = check_output(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    # shortcut: exportador TorchScript (dynamo=False), deprecado; o novo exige onnxscript,
    # dependencia nao aprovada. Trocar quando o legado sair do torch ou onnxscript for aprovado.
    torch.onnx.export(
        _Probs(model).eval(),
        torch.zeros(16, FEATURE_DIM),
        path,
        input_names=["features"],
        output_names=["probs"],
        dynamic_axes={"features": {0: "T"}, "probs": {0: "T"}},
        opset_version=OPSET,
        dynamo=False,
    )
    return path


def check_contract(path: Path) -> int:
    """Confere o contrato no arquivo exportado e devolve C. Fora dele, ValueError."""
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError(f"{path}: precisa de uma entrada e uma saida")

    def dims(value: onnx.ValueInfoProto) -> list[int | str]:
        t = value.type.tensor_type
        if t.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"{path}: {value.name} precisa ser float32")
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in t.shape.dim]

    i, o = dims(graph.input[0]), dims(graph.output[0])
    if len(i) != 2 or not isinstance(i[0], str) or i[1] != FEATURE_DIM:
        raise ValueError(f"{path}: entrada precisa ser [T, {FEATURE_DIM}] com T dinamico, e {i}")
    if len(o) != 2 or not isinstance(o[0], str) or not isinstance(o[1], int) or o[1] < 1:
        raise ValueError(f"{path}: saida precisa ser [T, C] com T dinamico e C fixo, e {o}")
    return o[1]


def run_onnx(path: Path, xs: list[np.ndarray]) -> list[np.ndarray]:
    """Probabilidades (T, C) do arquivo exportado, uma por entrada (T, FEATURE_DIM)."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return [session.run(None, {name: x})[0] for x in xs]


def validate_onnx(model: nn.Module, path: Path, lengths=(1, 17, 100), atol: float = ATOL) -> float:
    """Compara ONNX Runtime e PyTorch em entradas aleatorias de varios T.

    Devolve a maior diferenca absoluta; acima de `atol`, ValueError.
    """
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((t, FEATURE_DIM), dtype=np.float32) for t in lengths]
    wrapped = _Probs(model).eval()
    worst = 0.0
    with torch.no_grad():
        for x, got in zip(xs, run_onnx(path, xs), strict=True):
            ref = wrapped(torch.from_numpy(x)).numpy()
            if got.shape != ref.shape:
                raise ValueError(
                    f"ONNX diverge do PyTorch: forma {got.shape}, esperado {ref.shape}"
                )
            worst = max(worst, float(np.abs(got - ref).max()))
    if worst > atol:
        raise ValueError(f"ONNX diverge do PyTorch: {worst:.2e} > {atol:.0e}")
    return worst
