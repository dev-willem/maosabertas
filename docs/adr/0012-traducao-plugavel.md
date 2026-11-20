# ADR-012: Tradução plugável, por API ou modelo local

- **Status:** aceita
- **Data:** 2026-11-20

## Contexto

O ADR-010 fixou que a tradução lê **só** o protocolo. Falta decidir quem
traduz. Não existe corpus paralelo protocolo → português no projeto, então não
há como treinar um modelo próprio agora. Restam modelos de linguagem prontos:
por API (melhor qualidade esperada, mas o enunciado sai da máquina) ou local
(nada sai, qualidade a medir). O responsável pelo projeto pediu poder alternar
entre os dois com facilidade.

## Decisão

1. Crate novo `maosabertas-translate`, fora do núcleo, como o ONNX (ADR-005): o
   núcleo continua sem rede e compilável para wasm.
2. Interface única: `trait Translator { fn translate(&self, &Utterance) -> Result<Translation, Error> }`.
   A entrada é o tipo do protocolo; não há como passar vídeo nem landmarks.
3. Uma implementação, `Llm`, com dois backends escolhidos por variável de
   ambiente (`MAOSABERTAS_TRADUTOR`):

   | | `claude` | `local` |
   | --- | --- | --- |
   | Protocolo HTTP | API de mensagens da Anthropic | chat compatível com a OpenAI (Ollama, llama.cpp, vLLM) |
   | Modelo padrão | `claude-opus-5-5` | nenhum: obrigatório |
   | Formato da resposta | saída estruturada (`output_config.format`) | `response_format` com o mesmo JSON Schema |
   | Chave | `ANTHROPIC_API_KEY` | opcional |
   | Dado sai da máquina | sim | não |

   Trocar de backend é trocar uma variável; o código que chama não muda.
4. **O que o modelo recebe:** o `Utterance` em JSON, sem tradução anterior, e
   instruções de sistema que só explicam o protocolo como a spec define
   (glosa ≠ tradução, escopo não manual, referencial da fonologia). Nenhuma regra
   gramatical de Libras ou de reordenação (regra 3).
5. **O que volta:** `pt`, `conf`, `ambiguidade` e `alternativa`. A resposta é
   conferida no código (conf em [0, 1], texto não vazio), e o campo `model`
   registra o backend e o modelo (`claude:claude-opus-5-5`).
6. **No backend `claude`:** esforço `medium` explícito; `fallbacks: "default"`
   (beta `server-side-fallback-2026-07-01`), para que uma recusa por política seja
   refeita em outro modelo dentro da mesma chamada; `stop_reason` de recusa ou
   de corte vira erro, nunca tradução.
7. **Sem dependência nova:** o HTTP usa o `ureq` com TLS nativo, que já estava
   no `Cargo.lock` (o `ort` o usa no build). O lock só ganhou o crate do projeto.
8. Exemplo `traduzir`: lê o protocolo em JSONL e escreve o mesmo com
   `translation`, e encadeia com o `transcrever` por pipe. Enunciado que falha
   passa sem tradução, com o erro registrado.

## Alternativas consideradas

| Alternativa | Por que não (por ora) |
| --- | --- |
| Treinar seq2seq protocolo → português e exportar em ONNX | Não há corpus paralelo. Quando houver, entra como terceiro `Translator` com o mesmo contrato |
| SDK oficial em Python | A tradução está no caminho de produção, que é em Rust (CLAUDE.md); Python só offline |
| Um único formato (só compatível com OpenAI) apontando para o Claude | O Claude tem API própria; a camada de compatibilidade perde saída estruturada e o fallback de recusa |
| Regras de gramática no prompt | Seriam verdade linguística inventada (regra 3) |

## Consequências

- **Privacidade:** no backend `claude`, o enunciado sai da máquina. O ADR-011
  limita o que o protocolo carrega (sem landmarks nem descritor da mão), mas
  glosas e tempos ainda descrevem o que a pessoa sinalizou. Uso com dado de
  pessoas reais exige consentimento que cubra esse envio. O padrão seguro é o
  `local`.
- **Qualidade não medida (regra 4):** nenhuma afirmação de qualidade até haver
  conjunto de teste revisado por pessoas surdas e intérpretes. O `conf`
  devolvido é a opinião do modelo, não uma probabilidade calibrada.
- Mudar `SYSTEM` muda traduções: tratar como mudança de comportamento, com
  registro no commit.
- Os testes do crate usam um transporte HTTP falso e não precisam de rede nem
  de chave. Não houve chamada real a nenhum backend nesta entrega.
