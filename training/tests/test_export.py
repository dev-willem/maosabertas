"""Exportacao ONNX no contrato do runtime e validacao contra o PyTorch."""

import pytest
import torch

from maosabertas_training import export as ex
from maosabertas_training.features import FEATURE_DIM
from maosabertas_training.landmarks import REPO
from maosabertas_training.model import CausalTCN


def _model() -> CausalTCN:
    torch.manual_seed(0)
    return CausalTCN(classes=3, hidden=8, layers=1)


def test_exportacao_recusa_caminho_no_repositorio():
    with pytest.raises(ValueError, match="repositorio"):
        ex.export_onnx(_model(), REPO / "modelo.onnx")


def test_exporta_no_contrato_e_validacao_pega_divergencia(tmp_path):
    m = _model()
    path = ex.export_onnx(m, tmp_path / "m.onnx")
    assert ex.check_contract(path) == 3
    assert ex.validate_onnx(m, path) <= ex.ATOL
    with torch.no_grad():
        m.out.bias[0] += 1.0
    with pytest.raises(ValueError, match="diverge"):
        ex.validate_onnx(m, path)


@pytest.mark.parametrize(
    ("module", "example", "dynamic"),
    [
        (_model(), torch.zeros(1, 4, FEATURE_DIM), {"x": {1: "T"}, "y": {1: "T"}}),  # com batch
        (ex._Probs(_model()), torch.zeros(4, FEATURE_DIM), None),  # T fixo
    ],
)
def test_contrato_recusa_modelo_fora_do_formato(tmp_path, module, example, dynamic):
    path = tmp_path / "fora.onnx"
    torch.onnx.export(
        module.eval(),
        example,
        path,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes=dynamic,
        dynamo=False,
    )
    with pytest.raises(ValueError, match="entrada"):
        ex.check_contract(path)
