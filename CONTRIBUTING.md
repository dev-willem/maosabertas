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
  `cargo test` precisam passar.
- Dependências novas precisam de licença compatível com Apache-2.0 e
  não podem ser restritas a "uso em pesquisa" (por exemplo, OpenFace).
