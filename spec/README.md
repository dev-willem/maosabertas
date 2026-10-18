# Especificação do protocolo — v0.1

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

## Notação

| Marcação | Significado | Exemplo |
| --- | --- | --- |
| `MAIÚSCULAS` | Glosa de sinal lexical | `QUEIJO` |
| `PALAVRA-COMPOSTA` | Sinal composto | `CAVALO-MARINHO` |
| `d-a-n-i-e-l` | Soletração manual (`kind: fingerspelling`) | — |
| `CL-VEICULO(...)` | Classificador (`kind: classifier`, detalhes em `param`) | `CL-VEICULO` + `{"movimento":"queda"}` |
| `IX(esq)` | Apontação (`kind: pointing`) | — |

**Glosa não é tradução.** A ordem dos sinais em Libras não é fixa:
`QUEIJO RATO COMER` é tópico-comentário, uma entre várias ordens válidas.
Nenhuma implementação deve aplicar regra fixa de reordenação.

## Rótulos não manuais da v0.1

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
