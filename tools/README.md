# Ferramentas

- `validate_spec.py` — valida contra o JSON Schema os exemplos de
  `spec/examples/` e os enunciados de referência dos testes
  (`crates/*/tests/data/*.json`), que são gerados pelo próprio núcleo. Assim o
  CI pega o código saindo do schema. Roda a cada PR.

```bash
pip install jsonschema
python3 tools/validate_spec.py
```

Scripts de extração de features, preparação de corpus e priorização de
anotação entram aqui. Nenhum deles deve gravar dado de pessoa dentro do
repositório.
