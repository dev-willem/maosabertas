# ADR-003: Licença Apache-2.0 para o código

- **Status:** aceita
- **Data:** 2026-10-14

## Contexto

O objetivo é que o núcleo e o protocolo possam ser integrados em qualquer
sistema, inclusive comerciais, e que o projeto seja reconhecido quando
usado. Pela definição de código aberto da OSI, uma licença não pode
distinguir uso lucrativo de não lucrativo: a exigência de crédito vale
para todos igualmente.

## Decisão

1. O **código** é licenciado sob **Apache-2.0**, com um arquivo `NOTICE`.
2. O **nome e o logotipo** ficam fora da licença e são regidos por
   `TRADEMARKS.md`.
3. A **especificação do protocolo** (`spec/`) usará **CC BY 4.0**, que
   exige atribuição e é adequada a texto e especificações.
4. **Modelos treinados** terão licença própria, com model card, definida
   junto com quem gravou o corpus.
5. **Corpus e landmarks** não são abertos por padrão: dependem do
   consentimento específico de cada sinalizante.
6. Contribuições exigem **DCO** (`Signed-off-by`) desde o primeiro PR.
7. O copyright pertence aos autores coletivamente ("The Mãos Abertas
   Authors"), não a uma pessoa.

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| MIT | Sem concessão de patentes; crédito mais fraco |
| MPL-2.0 | Protege melhor o núcleo, mas aumenta o atrito jurídico para empresas |
| CPAL-1.0 | Única aberta com atribuição visível obrigatória, porém rara e afasta adotantes |
| GPL/AGPL | Empresas evitam embutir; contraria a meta de protocolo universal |

## Consequências

- Empresas podem usar, modificar e vender produtos baseados no núcleo,
  mantendo o aviso de copyright e o `NOTICE`.
- O crédito visível na interface **não é obrigatório**; é pedido por meio
  do selo sugerido. A única licença aberta que o exigiria (CPAL) foi
  descartada por reduzir a adoção.
- Dependências com restrição de uso (por exemplo, OpenFace, somente
  pesquisa) não podem entrar no núcleo. Modelos treinados em bases com
  termos restritivos podem herdar a
  restrição e precisam de revisão antes de publicação.

## Pendências

- [ ] Reservar a organização `maosabertas` no GitHub
- [ ] Pesquisar o nome no INPI e verificar domínios
- [ ] Revisão por advogado de software livre e proteção de dados
- [ ] Auditar a licença de cada dependência e de cada base de dados
- [x] Adicionar o texto da CC BY 4.0 em `spec/` (`spec/LICENSE`, texto oficial em
      creativecommons.org/licenses/by/4.0/legalcode.txt)
