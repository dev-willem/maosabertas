# ADR-004: Features calculadas só no núcleo, usadas pelo treino via pyo3

- **Status:** aceita
- **Data:** 2026-10-14

## Contexto

O modelo precisa receber no tempo real exatamente as features que viu no
treino. O ADR-001 põe as features no núcleo em Rust, mas o treino é em
Python. Duas implementações do mesmo cálculo divergem com o tempo, e a
divergência é silenciosa: o modelo só piora.

## Decisão

1. O cálculo de features vive em `maosabertas-core` (`features.rs`).
2. O crate `maosabertas-py` (pyo3, build com maturin) expõe esse cálculo
   como o módulo Python `maosabertas`. O treino nunca reimplementa.
3. Os arrays atravessam a fronteira como bytes float32 little-endian, sem
   o crate `numpy`, para manter uma dependência a menos.
4. A entrada é o `.npz` do `HolisticLandmarker` do MediaPipe, com NaN onde
   nada foi detectado (formato em `training/maosabertas_training/landmarks.py`).

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| Reimplementar em Python com teste cruzado | Duas fontes da verdade; o teste cobre só os casos lembrados |
| Crate `numpy` (rust-numpy) | Mais uma dependência para ganhar uma cópia a menos, fora do caminho de tempo real |
| CLI em Rust que converte `.npz` | Exige crate de leitura de `.npy` e um passo a mais no treino |

## Consequências

- Mudar o layout das features é mudança do núcleo e invalida modelos
  treinados com o layout anterior. `FEATURE_DIM` precisa acompanhar a
  versão do modelo.
- O workspace passa a exigir um interpretador Python para compilar.
  O crate `-py` exige Rust 1.83 por causa do pyo3 0.29; o resto segue em 1.75.
- A escolha das features (centro e escala pelos ombros, pontos do braço e
  da cabeça, blendshapes) é ponto de partida, ainda não medida.
