# Treino

Python, fora do produto. O resultado é um arquivo ONNX consumido pelo
núcleo em Rust.

## Ambiente

```bash
uv venv && uv pip install -e ".[dev]"
```

Só para extrair landmarks, sem PyTorch:

```bash
uv venv && uv pip install mediapipe opencv-python numpy pytest ruff
```

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
2. Baseline de sinais isolados (MINDS-Libras ou V-LIBRASIL).
3. Classificador facial, primeiro por regras, depois treinado.
4. Reconhecimento contínuo com CTC.

Atenção: as bases públicas têm termos próprios e algumas restringem uso
comercial. Verifique antes de publicar qualquer modelo treinado nelas.
