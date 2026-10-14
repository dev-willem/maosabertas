# ADR-002: Protocolo separado da implementação

- **Status:** aceita
- **Data:** 2026-10-14

## Contexto

A meta é que o reconhecimento de Libras possa ser integrado em qualquer
sistema. Se o formato de saída existir só dentro do código, ninguém
consegue implementar um reconhecedor compatível nem consumir a saída sem
usar o núcleo.

## Decisão

1. O formato de transcrição vive em `spec/`, como **JSON Schema
   versionado** com exemplos, independente do código.
2. `spec_version` é obrigatório em todo documento.
3. Os tipos Rust em `maosabertas-protocol` **seguem** o schema; o schema
   é a fonte da verdade.
4. O CI valida os exemplos contra o schema a cada PR.
5. Mudança no schema exige issue aprovada e entrada no `CHANGELOG.md`.

## Consequências

- Outra equipe pode escrever um reconhecedor compatível, em qualquer
  linguagem, sem usar o núcleo.
- Há custo de manter schema e tipos em sincronia. Mitigação possível:
  gerar os tipos a partir do schema, ou gerar o schema a partir dos tipos
  com `schemars`. A decisão foi adiada até o schema estabilizar.
- A especificação usa CC BY 4.0, diferente do código (ver ADR-003).
