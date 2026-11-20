# Mãos Abertas

Protocolo e núcleo de código aberto para **reconhecimento de Libras**
(Língua Brasileira de Sinais) em tempo real: da câmera a um protocolo com os
parâmetros da Libras e, dele, ao português.

> **Status: pesquisa em andamento.** O caminho completo — câmera → protocolo →
> português — já roda, mas o reconhecimento cobre só um vocabulário pequeno de
> sinais isolados, gravados em estúdio, e nada foi avaliado com pessoas surdas.
> Reconhecimento contínuo de língua de sinais é um problema de pesquisa em
> aberto no mundo todo. Não use como substituto de intérprete de Libras, nem em
> contextos médicos, jurídicos ou de emergência.

## Como funciona

```
câmera → landmarks → MEDIÇÃO     configuração de mão, locação, orientação, movimento
                     TRANSCRIÇÃO glosas, expressões não manuais, parâmetros por sinal → protocolo (JSON)
                     TRADUÇÃO    lê só o protocolo                                  → português
```

O **protocolo** (`spec/`) descreve um enunciado em trilhas alinhadas no tempo,
com os cinco parâmetros da Libras:

| Parâmetro | No protocolo |
| --- | --- |
| Configuração de mão (CM) | `fonologia[].cm` |
| Ponto de articulação (PA) | `fonologia[].pa` |
| Movimento (M) | `fonologia[].m` |
| Orientação da palma (O) | `fonologia[].o` |
| Expressões não manuais (ENM) | trilha `nonmanual`, com escopo de tempo próprio |

Cada parâmetro traz a **medida** feita pelo núcleo e, quando a equipe
linguística publicar os inventários, a **categoria**. A tradução recebe só o
protocolo — nunca vídeo nem landmarks — e pode usar um modelo de linguagem por
API ou um modelo local. Detalhes nos ADRs 010 a 013.

## Como usar

Como rodar, testar, treinar e o que falta: [docs/GUIA.md](docs/GUIA.md).

## Objetivo

Um **protocolo aberto** (formato de transcrição versionado) e um **núcleo
reutilizável** (biblioteca em Rust) que qualquer sistema possa integrar:
videochamadas, aplicativos, atendimento ao público. O aplicativo de
videochamada é a primeira prova de uso, não o produto final.

## Princípios

- **Pessoas surdas no centro:** gravando, anotando, avaliando e decidindo.
- **Privacidade por projeto:** o vídeo é processado no dispositivo; o protocolo
  não carrega landmarks nem medidas que identifiquem a pessoa.
- **Especificação separada da implementação**, para que outros possam
  implementar reconhecedores compatíveis.
- **Honestidade:** toda saída traz nível de confiança; nada de promessa
  além do que foi medido.

## Estrutura

```
spec/       protocolo: JSON Schema, exemplos, notação, CHANGELOG (v0.2)
crates/     maosabertas-protocol (tipos), -core (núcleo de tempo real),
            -onnx (inferência), -translate (tradução), -py (ponte para o treino)
training/   Python: extração, treino, exportação para ONNX, base única de dados
tools/      validação da especificação
docs/       guia e decisões (ADRs)
web/        casca web (ainda vazia)
```

## Licença

- **Código:** [Apache-2.0](LICENSE). Mantenha o aviso de copyright e o
  [NOTICE](NOTICE) ao redistribuir.
- **Nome e selo:** veja [TRADEMARKS.md](TRADEMARKS.md).
- **Especificação:** [CC BY 4.0](spec/LICENSE).
- **Modelos e dados:** termos próprios, definidos por modelo e por corpus. Um
  modelo treinado com dado de uso não comercial herda essa restrição.

Se você usa o Mãos Abertas em um produto, agradecemos o selo
**"Reconhecimento de Libras por Mãos Abertas"** em local visível.

## Contribuir

Programando ou não: pessoas surdas, intérpretes, linguistas e professores de
Libras são quem decide o que o sistema deve reconhecer e se ele está certo.
Veja como participar em [CONTRIBUTING.md](CONTRIBUTING.md). Todo commit exige
`Signed-off-by` (DCO). Nunca envie vídeo, landmarks ou dados de pessoas.

## Decisões de projeto

Veja [docs/adr](docs/adr) e o índice em [docs/README.md](docs/README.md).
