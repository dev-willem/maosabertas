# Mãos Abertas

Protocolo e núcleo de código aberto para **reconhecimento de Libras**
(Língua Brasileira de Sinais) em tempo real, com transcrição em glosa,
marcadores não manuais (expressões faciais com função gramatical e afetiva)
e tradução para o português.

> **Status: pesquisa em andamento. Ainda não funciona.**
> Reconhecimento contínuo de língua de sinais é um problema de pesquisa em
> aberto no mundo todo. Este projeto começa em domínio fechado e cresce em
> camadas, junto com a comunidade surda. Não use como substituto de
> intérprete de Libras, nem em contextos médicos, jurídicos ou de
> emergência.

## Como usar

Como rodar, como treinar e o que falta: [docs/GUIA.md](docs/GUIA.md).

## Objetivo

Um **protocolo aberto** (formato de transcrição versionado) e um **núcleo
reutilizável** (biblioteca em Rust) que qualquer sistema possa integrar:
videochamadas, aplicativos, atendimento ao público. O aplicativo de
videochamada é a primeira prova de uso, não o produto final.

## Princípios

- **Pessoas surdas no centro:** gravando, anotando, avaliando e decidindo.
- **Privacidade por projeto:** o vídeo é processado no dispositivo; só texto
  estruturado atravessa a rede.
- **Especificação separada da implementação**, para que outros possam
  implementar reconhecedores compatíveis.
- **Honestidade:** toda saída traz nível de confiança; nada de promessa
  além do que foi medido.

## Estrutura planejada

```
spec/       protocolo (JSON Schema, exemplos, notação)
crates/     núcleo em Rust, WASM, servidor, bindings
training/   treino em Python, exportação para ONNX
tools/      scripts de dados e anotação
docs/       decisões (ADRs) e guias
```

## Licença

- **Código:** [Apache-2.0](LICENSE). Mantenha o aviso de copyright e o
  [NOTICE](NOTICE) ao redistribuir.
- **Nome e selo:** veja [TRADEMARKS.md](TRADEMARKS.md).
- **Especificação:** [CC BY 4.0](spec/LICENSE).
- **Modelos e dados:** termos próprios, definidos por modelo e por corpus.

Se você usa o Mãos Abertas em um produto, agradecemos o selo
**"Reconhecimento de Libras por Mãos Abertas"** em local visível.

## Contribuir

Leia [CONTRIBUTING.md](CONTRIBUTING.md). Todo commit exige `Signed-off-by`
(DCO). Nunca envie vídeo, landmarks faciais ou dados de pessoas.

## Decisões de projeto

Veja [docs/adr](docs/adr).
