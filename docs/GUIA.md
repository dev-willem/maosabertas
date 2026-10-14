# Guia: como rodar, como treinar e o que falta

> **Estado:** pesquisa em andamento. Ainda não existe modelo treinado, então
> não há reconhecimento de Libras funcionando. Este guia mostra o que já roda
> e como chegar ao primeiro modelo.

Comandos em PowerShell, a partir da raiz do repositório.

---

## 1. Preparar o ambiente

Pré-requisitos:

| Ferramenta | Para quê | Instalação no Windows |
| --- | --- | --- |
| Rust (rustup) | núcleo, bindings, inferência | `winget install Rustlang.Rustup` |
| Visual Studio Build Tools (C++) | linker do Rust | instalador da Microsoft |
| Python 3.12 + `uv` | extração, treino | `winget install astral-sh.uv` |
| git | versionamento | `winget install Git.Git` |

Ambiente Python (uma vez):

```powershell
uv venv -p 3.12 training\.venv
uv pip install -p training\.venv mediapipe opencv-python numpy pytest ruff jsonschema onnx onnxruntime
uv pip install -p training\.venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p training\.venv .\crates\maosabertas-py
```

O último pacote compila o núcleo em Rust e o expõe ao Python (ADR-004).
Reinstale-o (`--reinstall`) sempre que mudar código em `crates/`.

O crate `maosabertas-py` precisa achar um Python para compilar. Com pyenv no
Windows, aponte para o do venv antes de rodar `cargo`:

```powershell
$env:PYO3_PYTHON = "$PWD\training\.venv\Scripts\python.exe"
```

---

## 2. Rodar os testes

```powershell
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all                                     # núcleo, fluxo, features, ONNX
training\.venv\Scripts\python.exe tools\validate_spec.py   # exemplos e JSON do núcleo contra o schema

cd training
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest -q               # extração, ponte Rust, treino sintético, ONNX
cd ..
```

A primeira compilação baixa o ONNX Runtime e precisa de rede.

---

## 3. Testar a extração com a câmera

É a única parte que hoje processa imagem real.

```powershell
cd training
.venv\Scripts\python.exe -m maosabertas_training.landmarks 0 $HOME\teste_landmarks.npz
```

- Não abre janela. Fique com ombros, rosto e mãos no quadro e levante só a
  mão **esquerda** por uns 10 segundos. Encerre com Ctrl+C.
- No resumo, `left_hand` deve aparecer em muitos quadros e `right_hand` em
  quase nenhum. Se for o contrário, os lados estão trocados: abra uma issue
  antes de treinar qualquer coisa.
- O arquivo é dado biométrico seu (LGPD). Fica fora do repositório; apague
  quando terminar.

Ver as features que o modelo receberia:

```powershell
.venv\Scripts\python.exe -c "import numpy as np; from maosabertas_training.features import features, fill_gaps; f = features(np.load(r'$HOME\teste_landmarks.npz')); ok = ~np.isnan(f).any(axis=1); print('quadros:', len(f), '| com ombros:', int(ok.sum())); print('NaN apos fill_gaps:', int(np.isnan(fill_gaps(f, 5)).sum()))"
```

---

## 4. Como treinar, passo a passo

O modelo aprende de pares **(trecho de vídeo, sequência de glosas)**. Vídeo
sem anotação não treina nada. Os passos 0 a 2 não são código e vêm antes de
tudo.

### Passo 0 — Permissão

- **Direito autoral:** vídeo público no YouTube não autoriza uso, e os termos
  do YouTube proíbem baixar sem permissão. Peça autorização por escrito a
  quem produziu.
- **LGPD:** extrair landmarks de rosto e mãos é tratamento de dado biométrico
  (dado sensível). Precisa de consentimento específico de cada pessoa que
  aparece sinalizando.
- **Bases públicas** (MINDS-Libras, V-LIBRASIL) já têm consentimento, mas
  têm termos próprios; algumas restringem uso comercial, e o modelo herda a
  restrição.
- Nada disso substitui revisão jurídica (pendência no ADR-003).

### Passo 1 — Escolher os vídeos

- Tronco, rosto e mãos inteiros no quadro; 720p ou mais; 25 fps ou mais;
  pouca compressão. Janela pequena de intérprete no canto costuma não ter
  resolução para os dedos.
- **Vários sinalizantes.** A avaliação é por sinalizante; com um ou dois não
  há como medir se o modelo generaliza.
- Se interpretação simultânea do português serve como dado, e em que
  proporção frente à sinalização espontânea de pessoas surdas, é decisão da
  equipe linguística.

### Passo 2 — Anotar

- No [ELAN](https://archive.mpi.nl/tla/elan), por pessoas surdas e
  linguistas, seguindo um guia de anotação (ainda não existe; abra uma issue
  do tipo "linguística").
- Uma trilha de glosas com início e fim de cada enunciado; se possível,
  trilhas de marcadores não manuais.
- Vocabulário de **domínio fechado**, 150 a 300 sinais na v1, definido pela
  equipe. Trechos com sinais fora dele ficam de fora.
- Legenda em português **não é glosa**. Guarde para a camada de tradução.

### Passo 3 — Cortar os clipes

Hoje é manual: um arquivo por enunciado anotado. Com
[ffmpeg](https://ffmpeg.org), usando os tempos do ELAN:

```powershell
ffmpeg -ss 00:01:02.500 -to 00:01:05.000 -i D:\videos\aula01.mp4 -an D:\clipes\aula01\0001.mp4
```

Vídeos e clipes ficam **sempre fora do repositório**.

### Passo 4 — Extrair os landmarks

```powershell
cd training
Get-ChildItem D:\clipes -Recurse -Filter *.mp4 | ForEach-Object {
    $saida = Join-Path D:\corpus ($_.FullName.Substring(10) -replace '\.mp4$', '.npz')
    New-Item -ItemType Directory -Force (Split-Path $saida) | Out-Null
    .venv\Scripts\python.exe -m maosabertas_training.landmarks $_.FullName $saida
}
```

(`Substring(10)` tira o prefixo `D:\clipes\`; ajuste ao seu caminho.)

### Passo 5 — Manifesto e vocabulário

`D:\corpus\corpus.csv` — um clipe por linha; caminho relativo ao próprio
manifesto; sinalizante por **pseudônimo**, nunca pelo nome:

```csv
clip,signer,glosses
aula01/0001.npz,s01,GLOSA1 GLOSA2 GLOSA3
aula01/0002.npz,s01,GLOSA4
interp03/0001.npz,s02,GLOSA2 GLOSA5
```

`D:\corpus\vocab.txt` — uma glosa por linha, na ordem definida pela equipe.
A classe 0 é reservada ao branco do CTC; a linha 1 vira a classe 1, e assim
por diante. Glosa do manifesto fora do vocabulário é erro.

### Passo 6 — Treinar

```powershell
.venv\Scripts\python.exe -m maosabertas_training.train `
    --manifest D:\corpus\corpus.csv `
    --vocab D:\corpus\vocab.txt `
    --out D:\experimentos\base-01
```

Parâmetros: `--epochs` (30), `--batch-size` (16), `--lr` (1e-3), `--hidden`
(128), `--layers` (4), `--test-fraction` (0,2 dos sinalizantes), `--max-gap`
(5), `--seed` (0). Nenhum desses valores foi calibrado.

O que acontece:

1. Divide os clipes **por sinalizante** (nenhum sinalizante nos dois lados).
2. Calcula as features pelo núcleo em Rust e preenche os quadros sem ombros
   com `fill_gaps` (ADR-006).
3. Treina um modelo causal (lookahead 0, ADR-008) com CTC.
4. Exporta `model.onnx` no contrato do runtime (ADR-005), confere o contrato
   e compara ONNX Runtime com PyTorch (diferença máxima 1e-5).
5. Avalia o arquivo exportado nos sinalizantes de teste.

### Passo 7 — Ler o resultado

`D:\experimentos\base-01\experiment.json` traz config, sinalizantes de cada
lado, loss por época e métricas: taxa de erro de glosa no agregado, **por
sinalizante** e, por glosa, acertos, substituições, omissões e inserções.
Com MLflow instalado, o experimento também vai para lá.

Antes de publicar qualquer número, ele precisa estar num experimento
registrado (regra 4 do `CLAUDE.md`).

### Passo 8 — Usar o modelo no tempo real (Rust)

O caminho por quadro, já implementado:

1. `frame_features` — landmarks → features (`maosabertas-core`);
2. `GapFiller` — preenche quadros sem ombros, **com o mesmo `max_gap` do treino**;
3. `Model::run` — features → probabilidades (`maosabertas-onnx`);
4. `Transcriber::push` — probabilidades → `Utterance` no formato do protocolo,
   fechando enunciados por pausa (ADR-007).

Falta um programa que ligue esses passos à câmera (ver abaixo).

---

## 5. O que falta fazer

### Pode ser feito agora (código)

- [ ] **Conversor ELAN (`.eaf`) → manifesto**, fatiando o `.npz` do vídeo
      inteiro pelos tempos das anotações (elimina o passo 3 manual).
- [ ] **Recorte da região do sinalizante** na extração (vídeos com
      intérprete no canto).
- [ ] **Programa de demonstração** (CLI em Rust): vídeo ou `.npz` → features →
      modelo → enunciados em JSON.
- [ ] **Transcritor sem modelo não manual**: hoje `Transcriber` exige os dois
      modelos; sem classificador facial treinado, não há como rodá-lo.
- [ ] **Treino do modelo não manual** (sigmoide por rótulo) no mesmo
      pipeline, quando houver anotação de marcadores.
- [ ] **Exportador ONNX novo do PyTorch** (o atual está descontinuado; o
      novo exige aprovar `onnxscript`).
- [ ] **Rodar o CI** pela primeira vez no GitHub e corrigir o que falhar.

### Depende de pessoas, dados ou decisão

- [ ] Confirmar com a câmera que esquerda e direita não estão trocadas.
- [ ] Revisão do schema e dos rótulos não manuais por linguistas e
      sinalizantes surdos (`spec/README.md`).
- [ ] Guia de anotação e vocabulário da v1 (150 a 300 sinais).
- [ ] Corpus com consentimento e permissão de uso.
- [ ] Calibrar com dados: `pause`, `max_duration`, `max_gap`, histerese e
      hiperparâmetros do treino.
- [ ] Camada de tradução glosa → português.
- [ ] Casca web (câmera no navegador) — exige aprovar dependências
      (TypeScript, Vite, MediaPipe Tasks, wasm-bindgen) e decidir a inferência
      no navegador (ADR-005).

### Jurídico e projeto (ADR-003)

- [ ] Reservar a organização `maosabertas` no GitHub.
- [ ] Pesquisar o nome no INPI e verificar domínios.
- [ ] Revisão por advogado de software livre e proteção de dados.
- [ ] Auditar a licença de cada base de dados.
- [ ] Adicionar o texto da CC BY 4.0 em `spec/`.
