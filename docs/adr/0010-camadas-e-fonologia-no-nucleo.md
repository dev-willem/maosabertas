# ADR-010: Três camadas e medição fonológica no núcleo

- **Status:** aceita
- **Data:** 2026-11-20

## Contexto

O objetivo do projeto é que a câmera produza um **protocolo** com os cinco parâmetros
da Libras — configuração de mão (CM), ponto de articulação (PA), movimento (M),
orientação da palma (O) e expressões não manuais (ENM) — e que um modelo de IA
**só leia esse protocolo** para traduzir ao português. Até aqui o protocolo carregava
glosa e marcadores não manuais, mas nenhum parâmetro manual. O ADR-009 propôs medir
parâmetros em vez de reconhecer classificadores inteiros e deixou como passo 3 portar
o descritor de CM e o detector de "segurar" para o núcleo.

## Decisão

### 1. Três camadas, um contrato no meio

```
câmera → landmarks → MEDIÇÃO     CM · PA · O · M por quadro, ENM por modelo
                     TRANSCRIÇÃO glosas, escopos, parâmetros por item  → protocolo (JSON)
                     TRADUÇÃO    lê só o protocolo                     → português
```

- **Medição** é geométrica e determinística quando dá (CM, PA, O, M) e aprendida
  quando não dá (ENM, glosa). Vive no núcleo em Rust, causal.
- **Transcrição** junta medição e modelos em itens do protocolo (`Transcriber`).
- **Tradução** recebe um `Utterance` e nada mais: nem vídeo, nem landmarks (ADR-012).
  Tudo o que ela precisa tem de estar no protocolo; o que não estiver se perde. Por
  isso o protocolo carrega incerteza (`conf`, `alt`) e o valor medido ao lado da
  categoria (ADR-011).

### 2. Medição fonológica no núcleo (`maosabertas-core::phonology`)

| O quê | Como |
| --- | --- |
| Referencial | O do corpo: origem no meio dos ombros, unidade = largura dos ombros, x para a direita **de quem sinaliza**, y para cima, z para a frente. Imagem sem espelhamento. É o referencial do protocolo |
| Mão | `HandPose`: 21 pontos nesse referencial; z relativo ao punho (o MediaPipe não dá profundidade da mão) |
| CM | `handshape`: o descritor de 34 valores do ADR-009 (15 flexões, 4 aberturas, 15 distâncias), invariante a rotação, escala, translação e espelho |
| PA | `palm_center`: centro da palma (punho e bases dos dedos) em x, y |
| O | `palm_normal` (para onde a palma olha, com a ordem do produto vetorial trocada entre as mãos) e `fingers` (punho → base do dedo médio) |
| Segurar | `is_hold`: mão no espaço de sinalização (punho até 1 largura de ombro abaixo dos ombros) e parada (menos de 1,5 palma/s) em relação ao **corpo** |

O binding Python expõe `handshape` e `holds` (ADR-004): pesquisa e treino usam o
mesmo cálculo do tempo real.

### 3. O que foi medido antes de decidir (experimento 006, corpus do modelo base, 12 sinalizantes)

- **Porte fiel:** o descritor em Rust difere do protótipo em no máximo 4,2e-4 em 5296
  quadros. O segurar concorda em 99,2% dos quadros; a diferença vem de medir a mão
  parada em relação aos ombros, não à imagem.
- **Profundidade de locação fica de fora:** o z do punho na pose tem cerca de 8 vezes o
  ruído de x e y em trechos de segurar e um deslocamento médio por sinalizante de 0,30
  largura de ombro (0,06 a 0,11 em x e y). Sem referência de profundidade real, não há
  como saber se ele acompanha a profundidade.
- **Orientação entra:** dentro de um segurar, a normal da palma treme 1,4° (mediana) e
  varia 33° entre trechos; a direção dos dedos, 1,1° e 41°.

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| Rede ponta a ponta vídeo → português | Caixa-preta sem protocolo; contradiz o objetivo e não deixa a comunidade surda auditar a saída |
| Medir só na imagem, sem referencial do corpo | Locação e movimento passariam a depender de onde a pessoa está no quadro |
| Profundidade pela pose já na v0.2 | Medida ruidosa e enviesada por sinalizante (item 3) |
| Descritor de CM dentro das 203 features | Muda `FEATURE_DIM` e invalida modelos sem medição de ganho (ADR-009) |

## Consequências

- O cálculo dos ombros passa a ser compartilhado entre `features.rs` e
  `phonology.rs`, com a mesma aritmética: as 203 features não mudam.
- Limiares de segurar e de repouso continuam sem calibração humana (`shortcut` no código).
- Categorias de CM, PA, O e M continuam com a equipe linguística (regra 3): o núcleo
  mede, não nomeia.
- Câmera espelhada (modo selfie) troca esquerda e direita; quem captura precisa
  desespelhar antes.
