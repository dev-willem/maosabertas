# Como contribuir

Obrigado por querer ajudar. Antes de começar:

## Regras que não mudam

1. **Nenhum dado pessoal no repositório.** Nem vídeo, nem landmarks
   faciais, nem corpus anotado. Vídeo de rosto e mãos é dado pessoal, e a
   análise facial gera dado biométrico sensível (LGPD). Só entram
   amostras sintéticas ou com consentimento específico de publicação.
2. **A especificação em `spec/` manda.** O código se adapta ao schema; mudar
   o schema exige uma proposta (issue) e uma nova versão da especificação.
3. **Verdade linguística é da comunidade surda.** Glosas, classes de
   expressão facial e o guia de anotação são decididos com linguistas e
   pessoas surdas. Propostas de mudança nessas áreas são bem-vindas, mas
   passam por revisão deles, não só pela de quem programa.

## Como participar sem programar

Quem decide o que o sistema deve reconhecer, e se ele acerta, são pessoas
surdas e quem trabalha com Libras. O código só mede e organiza.

| Quem | Como ajuda |
| --- | --- |
| Pessoas surdas | Escolher em que situações o sistema deve servir primeiro (o domínio da v1). Decidir glosas, inventários de configuração de mão, locação, movimento, orientação e classificadores, e os rótulos não manuais. Avaliar o reconhecimento e as traduções. Gravar sinais, com consentimento |
| Linguistas e professores de Libras | Guia de anotação; inventários; critério entre sinal lexical e uso classificador (ADR-009 e ADR-011); revisão da especificação |
| Intérpretes de Libras | Traduções de referência para o conjunto de teste; avaliação da saída em português |
| Anotadoras e anotadores | Anotar gravações no ELAN seguindo o guia de anotação |

**Como começar:** abra uma issue com o modelo "Questão linguística". **Nunca
anexe vídeo de pessoa numa issue ou PR:** é dado pessoal (LGPD). Gravação para o
projeto só com termo de consentimento específico e combinada por canal privado.

**O que o projeto garante a quem participa:**

- consentimento por escrito, explicando para que o dado serve, e o direito de
  retirar as próprias gravações a qualquer momento;
- crédito pelo nome, se a pessoa quiser, ou anonimato;
- nenhuma decisão sobre glosa, categoria ou regra da Libras sem revisão de
  pessoas surdas (regra 3 acima);
- o sistema é complemento, não substituto de intérprete.

## Sign-off obrigatório (DCO)

Todo commit precisa terminar com uma linha de assinatura:

```
Signed-off-by: Seu Nome <seu@email.com>
```

Use `git commit -s`. Ela declara que você tem o direito de enviar a
contribuição sob a licença Apache-2.0 do projeto, nos termos do
[Developer Certificate of Origin](https://developercertificate.org/).
Commits sem assinatura não são aceitos.

## Pull requests

- Pequenos e focados: uma mudança por PR.
- Com testes. Para o núcleo em Rust: `cargo fmt`, `cargo clippy` e
  `cargo test` precisam passar. No treino em Python: `ruff check`,
  `ruff format --check` e `pytest`. Mudou `spec/`? Rode
  `tools/validate_spec.py` e registre no `spec/CHANGELOG.md`.
- Nenhuma afirmação de desempenho sem número de experimento registrado.
- Dependências novas precisam de licença compatível com Apache-2.0 e
  não podem ser restritas a "uso em pesquisa" (por exemplo, OpenFace).
