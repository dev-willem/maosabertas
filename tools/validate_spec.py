#!/usr/bin/env python3
"""Valida os exemplos de spec/examples contra spec/utterance.schema.json."""
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

    failed = 0
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        if errors:
            failed += 1
            print(f"FALHOU  {path.name}")
            for err in errors:
                where = "/".join(str(p) for p in err.path) or "(raiz)"
                print(f"        {where}: {err.message}")
        else:
            print(f"ok      {path.name}")

    print(f"\n{len(files) - failed}/{len(files)} exemplos validos")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
