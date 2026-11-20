# Changelog da especificação

Versionamento semântico. Mudança incompatível sobe a versão maior.

## [0.2] — não publicada

Os parâmetros manuais entram no protocolo (ADR-010 e ADR-011).

### Adicionado
- `fonologia` em cada item manual: uma entrada por mão (`esquerda`,
  `direita`) com `cm`, `pa`, `o` e `m`. Cada parâmetro tem a medida do núcleo
  e a categoria (`cat`, `conf`) do inventário linguístico, ausente até ele
  existir.
- Referencial do corpo para posições e vetores.
- `pa.prof` reservado para profundidade, ainda não emitida.
- Itens manuais podem se sobrepor no tempo.

### Removido (incompatível)
- `param` com chaves livres. Para migrar: os parâmetros vão para
  `fonologia`; os valores livres da v0.1 eram ilustrativos e não têm
  equivalente.

### Em aberto
- Inventários de CM, PA, O e M, e os nomes dos classificadores (equipe
  linguística).
- Validação da profundidade monocular.

## [0.1] — não publicada

Primeira versão, rascunho.

### Adicionado
- Enunciado com trilhas `manual` e `nonmanual` em tempos independentes.
- `kind` para sinal: lexical, classificador, soletração, apontação.
- Alternativas (`alt`) e confiança por item.
- Conjunto fechado de rótulos não manuais.
- Bloco `quality` com condições de captura.
- Bloco `translation`, opcional enquanto o enunciado não fechou.

### Em aberto
- Espaço de sinalização e referência anafórica (fora da v1).
- ~~Parâmetros de classificador: `param` ainda tem chaves livres.~~ Resolvido na 0.2.
- Camada `olhar`: reservada, sem rótulos.
- Revisão por linguistas e sinalizantes surdos.
