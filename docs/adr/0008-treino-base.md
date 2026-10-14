# ADR-008: Treino base do modelo manual com TCN causal e CTC

- **Status:** aceita
- **Data:** 2026-10-14

## Contexto

O runtime (ADR-005) consome um ONNX `[T, FEATURE_DIM] -> [T, C]` e
decodifica com CTC guloso, quadro a quadro e em fluxo. O treino precisa
de um modelo de partida que caiba nesse contrato e no tempo real, antes de
haver dados. Nada foi medido ainda: a escolha é de forma, não de
desempenho.

## Decisão

1. Modelo `CausalTCN` (`training/maosabertas_training/model.py`):
   convoluções temporais causais e dilatadas (kernel 3, dilatação
   1, 2, 4, ...), com residual, saída por quadro.
2. **Lookahead de 0 quadros.** A saída no quadro `t` só depende dos
   quadros `<= t`. Campo receptivo de `1 + 2 (2^L - 1)` quadros com `L`
   camadas: 31 com as 4 camadas padrão.
3. Treino com CTC, branco na classe 0; vocabulário e manifesto (CSV) vêm
   de arquivos do usuário, fora do repositório.
4. Exportação com o exportador TorchScript do PyTorch (`dynamo=False`),
   opset 17, com softmax e sem dimensão de batch. O arquivo é conferido
   contra o contrato e comparado com o PyTorch no ONNX Runtime (diferença
   absoluta máxima de `1e-5` nas probabilidades).
5. A avaliação roda o arquivo ONNX exportado, não o modelo PyTorch.

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| LSTM/Transformer bidirecional | Lookahead do enunciado inteiro; não serve para fluxo |
| GRU unidirecional | Causal, mas o estado não cabe no contrato de uma entrada e uma saída; a janela teria de ser recalculada do início |
| Convolução com lookahead > 0 (padding assimétrico) | Atrasa a saída; adiado até haver medição que justifique |
| Exportador novo (`dynamo=True`) | Exige `onnxscript`, dependência não aprovada |

## Consequências

- Para reproduzir a saída do treino, o runtime precisa dar ao modelo pelo
  menos `campo receptivo - 1` quadros de contexto antes do quadro que vai
  decodificar.
- Com lookahead 0, a glosa só pode sair depois de o modelo ver o sinal; o
  atraso real só se conhece medindo com dado real.
- O exportador TorchScript está deprecado no PyTorch; trocar quando ele
  sair ou quando `onnxscript` for aprovado.
- Arquitetura, tamanho e hiperparâmetros padrão são ponto de partida, sem
  medição.
