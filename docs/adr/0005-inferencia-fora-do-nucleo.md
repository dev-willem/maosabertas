# ADR-005: Inferência ONNX em crate separado do núcleo

- **Status:** aceita
- **Data:** 2026-10-14

## Contexto

O ADR-001 prevê o mesmo núcleo compilando para nativo e para
`wasm32-unknown-unknown`. O ONNX Runtime (crate `ort`) é biblioteca nativa
e não compila para wasm32. Se entrar no `maosabertas-core`, o núcleo deixa
de ir para o navegador.

## Decisão

1. A inferência fica em `maosabertas-onnx`, que depende do núcleo; o
   núcleo não depende dele.
2. Contrato do modelo exportado pelo treino: uma entrada float32
   `[T, FEATURE_DIM]` com `T` dinâmico e uma saída float32 `[T, C]` com `C`
   fixo, já em probabilidades (softmax no modelo manual, sigmoide por rótulo
   no não manual). O crate recusa modelo fora do contrato ao carregar.
3. Entrada com NaN é recusada. Quadro sem referência precisa ser tratado
   antes, pela mesma regra do treino; essa regra ainda não existe.

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| `ort` no núcleo atrás de feature | Funciona, mas mistura dependência nativa no crate que precisa ir para wasm |
| `tract` (Rust puro, compila para wasm) | Cobertura de operadores menor; reavaliar quando houver modelo real e alvo wasm |

## Consequências

- No navegador, a inferência vai precisar de outro caminho (`tract`,
  ONNX Runtime Web pela casca, ou outro). Decisão adiada para a fase da web.
- `ort` 2.0 ainda é release candidate e exige Rust 1.92 neste crate.
- O build baixa os binários do ONNX Runtime (licença MIT) na primeira
  compilação; precisa de rede.
