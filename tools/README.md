# Ferramentas

- `validate_spec.py` — valida `spec/examples/*.json` contra o JSON Schema.
  Roda no CI a cada PR.

```bash
pip install jsonschema
python3 tools/validate_spec.py
```

Scripts de extração de features, preparação de corpus e priorização de
anotação entram aqui. Nenhum deles deve gravar dado de pessoa dentro do
repositório.
