# Treino

Python, fora do produto. O resultado é um arquivo ONNX consumido pelo
núcleo em Rust.

## Ambiente

```bash
uv venv && uv pip install -e ".[dev]"
```

Só para extrair landmarks e calcular features, sem PyTorch:

```bash
uv venv && uv pip install mediapipe opencv-python numpy pytest ruff ../crates/maosabertas-py
```

O último pacote compila o núcleo em Rust (precisa de `cargo`). No Windows
com pyenv, o pyo3 não acha o interpretador pelos shims: exporte
`PYO3_PYTHON` com o caminho do `python.exe` do venv antes de rodar `cargo`
no workspace.

## Extração de landmarks

```bash
python -m maosabertas_training.landmarks video.mp4 ~/dados/saida.npz
python -m maosabertas_training.landmarks 0 ~/dados/saida.npz   # câmera; Ctrl+C encerra
```

Usa o `HolisticLandmarker` do MediaPipe (mãos, rosto com blendshapes e
pose numa passada). O modelo é baixado uma vez para `~/.cache/maosabertas`
(ou `MAOSABERTAS_MODELS`). A saída é recusada se cair dentro do
repositório. O formato dos arrays está no docstring de
`maosabertas_training/landmarks.py`.

## Features

`maosabertas_training.features.features(np.load("saida.npz"))` devolve
`(T, 203)`: coordenadas centradas nos ombros e escaladas pela largura
deles, mais blendshapes. O cálculo é o do núcleo em Rust
(`crates/maosabertas-core/src/features.rs`), o mesmo do tempo real.
Quadro sem ombros sai como NaN.

Antes de entrar no modelo, `fill_gaps(feats, max_gap)` tira os NaN com a
regra do núcleo (`crates/maosabertas-core/src/gaps.rs`, ADR-006): os
primeiros `max_gap` quadros de uma lacuna repetem o último quadro válido;
os seguintes, e os anteriores ao primeiro quadro válido, viram zeros. O
tempo real usa a mesma regra; o `max_gap` do treino precisa acompanhar o
modelo exportado.

## Treino

```bash
python -m maosabertas_training.train --manifest ~/dados/corpus.csv \
    --vocab ~/dados/vocab.txt --out ~/experimentos/base-01
```

Divide os clipes por sinalizante (`--test-fraction` dos sinalizantes vai
para o teste, sorteio fixado por `--seed`), treina o modelo manual com CTC,
exporta `model.onnx` no contrato do runtime (ADR-005), confere o contrato,
compara ONNX Runtime e PyTorch (diferença máxima `1e-5`) e avalia o arquivo
exportado nos sinalizantes de teste. Config, sinalizantes de cada lado,
loss por época, vocabulário e métricas vão para `experiment.json` no
diretório de saída, e para o MLflow se estiver instalado (sem
`MLFLOW_TRACKING_URI`, em `mlruns/` dentro do diretório de saída). A saída
é recusada se cair dentro do repositório. Outros hiperparâmetros:
`--epochs`, `--batch-size`, `--lr`, `--hidden`, `--layers` e `--max-gap`
(lacuna preenchida por `fill_gaps`; use o mesmo valor no tempo real).

Métricas: taxa de erro de glosa (distância de edição / tamanho da
referência) no agregado e por sinalizante; por glosa, acertos,
substituições e omissões (contados na glosa da referência) e inserções
(na glosa inserida).

O modelo (`model.py`) é causal: lookahead de 0 quadros, campo receptivo de
`1 + 2 (2^L - 1)` quadros com `L` camadas. Ver
`docs/adr/0008-treino-base.md`.

**Manifesto:** CSV UTF-8 com cabeçalho `clip,signer,glosses`. `clip` é o
`.npz` da extração (absoluto ou relativo ao manifesto), `signer` é o id do
sinalizante (pseudônimo, nunca o nome) e `glosses` são as glosas do clipe
separadas por espaço, na ordem sinalizada.

```csv
clip,signer,glosses
s01/0001.npz,s01,GLOSA1 GLOSA2
```

**Vocabulário:** texto UTF-8, uma glosa por linha. A classe 0 é o branco
do CTC; a glosa da linha `i` (sem contar linhas vazias) é a classe `i`.
Glosa do manifesto fora do vocabulário é erro, com linha e glosa.

## Regras

- **Nenhum dado entra no repositório.** Caminhos de dados vêm de variável
  de ambiente ou arquivo de configuração local, nunca versionados.
- Divisão de treino e teste **por sinalizante**, nunca por clipe.
- Todo experimento registrado (MLflow ou Weights & Biases).
- Após exportar para ONNX, validar numericamente contra o PyTorch antes
  de considerar o modelo pronto.
- Métricas por sinalizante e por classe, nunca só a média.

## Ordem sugerida

1. Extração de landmarks e blendshapes com MediaPipe.
2. Baseline de sinais isolados, numa base pública.
3. Classificador facial, primeiro por regras, depois treinado.
4. Reconhecimento contínuo com CTC.

Atenção: as bases públicas têm termos próprios e algumas restringem uso
comercial. Verifique antes de publicar qualquer modelo treinado nelas.
