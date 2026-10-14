# ADR-006: Regra única para quadros sem referência

- **Status:** aceita
- **Data:** 2026-10-14

## Contexto

Sem ombros detectados, `frame_features` não tem centro nem escala e
devolve `None`; no treino o quadro vira linha NaN. O `maosabertas-onnx`
recusa NaN (ADR-005). Treino e tempo real precisam tratar esse quadro do
mesmo jeito, senão o modelo recebe no produto algo que não viu no treino
(o mesmo motivo do ADR-004).

## Decisão

1. A regra vive no núcleo (`maosabertas-core/src/gaps.rs`), com uma única
   implementação: `GapFiller` em fluxo, quadro a quadro
   (`Option<&[f32]>`, como sai de `frame_features`), e `fill_gaps` em lote,
   que só alimenta o `GapFiller`. O treino usa `fill_gaps` via
   `maosabertas-py`.
2. Os primeiros `max_gap` quadros de uma lacuna repetem o último quadro
   válido inteiro, presenças inclusive. Do quadro `max_gap + 1` em
   diante, e antes de qualquer quadro válido, sai a linha toda zero
   (presenças 0, como mão e rosto ausentes).
3. `max_gap` não tem default: é parâmetro do treino e acompanha o modelo.
4. No lote, linha com algum NaN conta como quadro sem referência.

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| Zerar a lacuna inteira quando passa de `max_gap` | Em fluxo só se sabe o tamanho da lacuna quando ela acaba: custaria até `max_gap` quadros de atraso |
| Interpolar entre os quadros válidos vizinhos | Precisa do quadro seguinte: atraso no tempo real |
| Descartar o quadro | Quebra o alinhamento quadro a quadro com o tempo, de que dependem a decodificação e o escopo |
| Regra só no treino, em Python | Duas implementações, divergência silenciosa (ADR-004) |

## Consequências

- O tempo real não ganha atraso: cada quadro sai assim que entra.
- Uma lacuna longa aparece para o modelo como "congela e depois some".
- Mudar a regra ou o `max_gap` muda a entrada do modelo e exige retreino.
- O valor de `max_gap` e o efeito da regra no desempenho ainda não foram
  medidos.
