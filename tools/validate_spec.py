#!/usr/bin/env python3
"""Valida spec/examples e os JSON de teste em crates/*/tests/data contra o schema."""
import json
import pathlib
import sys

from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "spec" / "utterance.schema.json"
EXAMPLES = ROOT / "spec" / "examples"


def main() -> int:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    files = sorted(EXAMPLES.glob("*.json"))
    if not files:
        print("nenhum exemplo encontrado", file=sys.stderr)
        return 1
    # Enunciados de referencia dos testes: o JSON que o codigo gera.
    files += sorted(ROOT.glob("crates/*/tests/data/*.json"))

    failed = 0
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        doc = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        if errors:
            failed += 1
            print(f"FALHOU  {name}")
            for err in errors:
                where = "/".join(str(p) for p in err.path) or "(raiz)"
                print(f"        {where}: {err.message}")
        else:
            print(f"ok      {name}")

    print(f"\n{len(files) - failed}/{len(files)} documentos validos")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
