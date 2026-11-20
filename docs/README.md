# Documentação

- [`GUIA.md`](GUIA.md) — como rodar, testar, treinar e o que falta fazer.
- [`../spec/README.md`](../spec/README.md) — o protocolo: notação, fonologia e
  rótulos não manuais.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — como participar, programando ou não.
- `adr/` — decisões de arquitetura, numeradas e em ordem cronológica. Uma decisão
  relevante vira um ADR curto: contexto, decisão, alternativas, consequências.

A documentação técnica completa (camada facial, corpus, avaliação, ética) ainda
está em documento externo e será migrada para cá, onde fica versionada junto com
o código.

## Decisões (ADRs)

| ADR | Decisão | Status |
| --- | --- | --- |
| [001](adr/0001-nucleo-em-rust.md) | Núcleo em Rust, Python só no treino | aceita |
| [002](adr/0002-protocolo-separado.md) | Protocolo separado da implementação | aceita |
| [003](adr/0003-licenca-apache-2-0.md) | Licença Apache-2.0 para o código | aceita |
| [004](adr/0004-features-no-nucleo.md) | Features calculadas só no núcleo, usadas pelo treino via pyo3 | aceita |
| [005](adr/0005-inferencia-fora-do-nucleo.md) | Inferência ONNX em crate separado do núcleo | aceita |
| [006](adr/0006-quadros-sem-referencia.md) | Regra única para quadros sem referência | aceita |
| [007](adr/0007-fim-de-enunciado.md) | Fim de enunciado por pausa manual ou duração máxima | aceita |
| [008](adr/0008-treino-base.md) | Treino base do modelo manual com TCN causal e CTC | aceita |
| [009](adr/0009-classificadores-por-parametros.md) | Classificadores por parâmetros fonológicos | proposta (depende de decisões linguísticas) |
| [010](adr/0010-camadas-e-fonologia-no-nucleo.md) | Três camadas e medição fonológica no núcleo | aceita |
| [011](adr/0011-fonologia-no-protocolo.md) | Fonologia no protocolo (spec v0.2) | aceita |
| [012](adr/0012-traducao-plugavel.md) | Tradução plugável, por API ou modelo local | aceita |
| [013](adr/0013-base-unica.md) | Base única de dados em SQLite | aceita |
