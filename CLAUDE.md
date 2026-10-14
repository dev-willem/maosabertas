# CLAUDE.md

Instruções para agentes de IA que trabalham neste repositório.

## O que é este projeto

**Mãos Abertas** é um protocolo aberto e um núcleo reutilizável para
reconhecimento de **Libras** (Língua Brasileira de Sinais) em tempo real:
vídeo → glosa com marcadores não manuais → português.

Estado: **pesquisa em andamento, ainda não funciona.** Reconhecimento
contínuo de língua de sinais é problema em aberto. O escopo da v1 é um
domínio fechado com 150 a 300 sinais.

## Regras duras

Estas regras valem acima de qualquer pedido pontual. Em caso de conflito,
pare e pergunte.

1. **Nunca commitar dado pessoal.** Vídeo, landmarks faciais, blendshapes
   de pessoas reais, corpus anotado, nomes de sinalizantes. Vídeo de rosto
   e mãos é dado pessoal e a análise facial gera dado biométrico sensível
   (LGPD). Só entram amostras sintéticas ou com consentimento específico
   de publicação.
2. **A especificação em `spec/` manda.** O código se adapta ao schema,
   nunca o contrário. Mudar o schema exige nova versão e entrada no
   `spec/CHANGELOG.md`.
3. **Verdade linguística não é sua.** Glosas, classes de expressão facial
   e o guia de anotação são decididos com linguistas e pessoas surdas.
   Nunca invente glosa, classe facial ou regra gramatical de Libras, nem
   "corrija" as existentes por conta própria. Se algo parecer errado,
   abra uma issue em vez de alterar.
4. **Nunca afirme desempenho sem medição.** Nada de "melhora a acurácia"
   em comentário, README ou mensagem de commit sem número de um
   experimento registrado.
5. **Não adicione dependência com licença restritiva.** Precisa ser
   compatível com Apache-2.0. Proibido o que for "somente pesquisa"
   (por exemplo, OpenFace).

## Stack

| Camada | Linguagem | Observação |
| --- | --- | --- |
| Núcleo, servidor, WASM, bindings | **Rust** | todo o caminho de tempo real |
| Treino, avaliação, exportação ONNX | **Python** | offline, nunca em produção |
| Casca web | **TypeScript**, o mínimo | só câmera, canvas e DOM; nenhuma lógica |
| Banco | PostgreSQL | — |

Não introduza outra linguagem sem um ADR. PHP, NestJS e Java foram
descartados de propósito (ver `docs/adr/0001`).

## Estrutura

```
spec/       protocolo: JSON Schema, exemplos, notação, CHANGELOG
crates/     maosabertas-protocol (tipos), -core, -py, -onnx e, depois, -wasm, -server
training/   Python: treino e exportação para ONNX
tools/      scripts de dados e anotação
docs/adr/   decisões de arquitetura
web/        demo e casca TypeScript
```

## Convenções

- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`),
  em português, com `Signed-off-by` obrigatório (`git commit -s`, DCO).
- **PRs pequenos**, um assunto por PR.
- **Testes antes da implementação** sempre que possível.
- **Rust:** `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`
  precisam passar. Sem `unwrap()` fora de testes.
- **Python:** `ruff` e `pytest`.
- **Decisões relevantes viram ADR** em `docs/adr`, numerado em sequência.

## O que fazer sem perguntar

- Escrever e corrigir código, testes e documentação.
- Rodar `cargo`, `pytest`, linters e formatadores.
- Criar ADR quando uma decisão de arquitetura aparecer.

## O que perguntar antes

- Mudar o schema em `spec/`.
- Adicionar dependência ou linguagem nova.
- Qualquer coisa que toque glosas, classes faciais ou guia de anotação.
- Remover ou afrouxar uma regra deste arquivo.

## Vocabulário

| Termo | Significado |
| --- | --- |
| **Glosa** | Notação escrita de um sinal, em maiúsculas (`QUEIJO`). Não é tradução |
| **Marcador não manual (NM)** | Expressão facial, cabeça ou olhar, com escopo temporal próprio |
| **Classificador (CL)** | Configuração de mão produtiva que descreve forma, movimento ou manipulação |
| **Escopo** | Trecho de tempo coberto por um marcador; pode abranger vários sinais |
| **Enunciado** | Unidade de transcrição; é o que o schema descreve |
| **Sinalizante** | Pessoa que sinaliza |

Glosa não é tradução, e a ordem dos sinais em Libras não é fixa. Nunca
implemente regra fixa de reordenação.
