# Especificação do protocolo — v0.2

> **Licença:** esta pasta é publicada sob **CC BY 4.0** (atribuição),
> diferente do código, que é Apache-2.0. O texto da licença está em
> [`LICENSE`](LICENSE).

Status: **rascunho**. O conjunto de rótulos ainda não passou por revisão
de linguistas e sinalizantes surdos.

## O que o protocolo descreve

Um **enunciado**: trilhas temporais alinhadas, não uma linha de texto.
A expressão facial tem escopo próprio e cobre vários sinais, por isso
`nonmanual` tem tempo independente de `manual`.

```
tempo    0,0s ──────── 0,6s ──────── 1,2s ──────── 1,8s
manual   [ QUEIJO ]    [ RATO ]      [ COMER ]
facial   ├────── interrogativa-sn (conf. 0,86) ──────┤
saída    "O rato comeu o queijo?"
```

A mesma sequência de glosas com face neutra é afirmação. É isso que a
camada de tradução recebe.

Desde a v0.2, cada item manual também traz a `fonologia` de cada mão que o
articula: configuração de mão (`cm`), ponto de articulação (`pa`),
orientação (`o`) e movimento (`m`). As expressões não manuais são a trilha
`nonmanual`. Com isso, os cinco parâmetros da Libras chegam à tradução, que
lê **só** o protocolo (ADR-010).

## Notação

| Marcação | Significado | Exemplo |
| --- | --- | --- |
| `MAIÚSCULAS` | Glosa de sinal lexical | `QUEIJO` |
| `PALAVRA-COMPOSTA` | Sinal composto | `CAVALO-MARINHO` |
| `d-a-n-i-e-l` | Soletração manual (`kind: fingerspelling`) | — |
| `CL-VEICULO` | Classificador (`kind: classifier`); forma, locação e trajetória em `fonologia` | `CL-VEICULO` + `"m": {"desloc": [0.08, -0.51]}` |
| `IX(esq)` | Apontação (`kind: pointing`) | — |

**Glosa não é tradução.** A ordem dos sinais em Libras não é fixa:
`QUEIJO RATO COMER` é tópico-comentário, uma entre várias ordens válidas.
Nenhuma implementação deve aplicar regra fixa de reordenação.

## Fonologia (v0.2)

Uma entrada por mão que articula o item (`mao`: `esquerda` ou `direita`, de
quem sinaliza). Cada parâmetro tem dois níveis:

- **Medida:** geometria calculada pelo núcleo, sem modelo e sem decisão
  linguística.
- **Categoria** (`cat` e `conf`): valor de um inventário publicado pela equipe
  linguística e por pessoas surdas. Enquanto o inventário não existe, `cat`
  fica ausente; `conf` sem `cat` é inválido.

**Referencial do corpo:** origem no meio dos ombros, unidade = largura dos
ombros, `x` para a direita de quem sinaliza, `y` para cima e `z` para a frente
(de quem sinaliza para a câmera).

| Campo | Medida | Como sai |
| --- | --- | --- |
| `visivel` | fração dos quadros do trecho com a mão no espaço de sinalização | mão abaixo de 0,5 não entra |
| `cm` | — | só `cat`; o descritor de forma não sai (privacidade) |
| `pa.pos` | centro da palma (x, y) nos quadros de "segurar" do trecho | mediana |
| `pa.prof` | reservado | não emitido: a profundidade monocular não foi validada |
| `o.palma` | vetor unitário para onde a palma olha | mediana nos quadros de segurar |
| `o.dedos` | vetor unitário do punho à base do dedo médio | idem |
| `m.desloc` | centro da palma do primeiro ao último quadro visível | (x, y) |
| `m.trajeto` | comprimento do caminho percorrido | larguras de ombro |

Valores com duas casas decimais. O protocolo **nunca** carrega landmarks,
trajetórias quadro a quadro nem o descritor de forma da mão: são dado
biométrico e não ajudam a tradução (ADR-011).

Itens manuais podem se sobrepor no tempo (por exemplo, um classificador em
cada mão).

## Rótulos não manuais (sem mudança desde a v0.1)

Conjunto fechado. Mudá-lo exige nova versão da especificação.

| `tier` | `label` | Efeito |
| --- | --- | --- |
| `gramatical` | `neutra` | declarativa |
| `gramatical` | `interrogativa-sn` | pergunta fechada |
| `gramatical` | `interrogativa-qu` | pergunta aberta |
| `gramatical` | `duvida` | depende do contexto |
| `cabeca` | `negacao`, `afirmacao` | nega ou enfatiza |
| `intensidade` | `baixa`, `media`, `alta` | grau lexical |
| `afetiva` | `alegria`, `tristeza`, `susto` | tom da tradução |
| `olhar` | (reservado) | — |

As pistas visuais de cada rótulo são **hipóteses de partida** vindas da
literatura geral, ainda não validadas. O rótulo descreve uma **expressão
facial sinalizada**, nunca o estado emocional da pessoa.

## Validação

```bash
pip install jsonschema
python3 tools/validate_spec.py
```

## Compatibilidade

`spec_version` é obrigatório e fixo por versão. Mudança incompatível sobe
a versão e entra no `CHANGELOG.md`. Um consumidor pode falar o protocolo
sem usar o núcleo: basta validar contra o schema.
