"""Reads the JSONL data files the eval scripts share and checks the tagger weights directory."""

import json
from pathlib import Path

SAMPLE_SIZE = 2000  # keeps the Presidio runs short; the synthetic file is already this size
SAMPLE_SEED = 7
MODEL_DIR_HELP = "weights directory from hf download elajeff/greek-pii-redactor-ner"


def load_records(data_dir: Path, file_name: str, command: str) -> list[dict]:
    path = data_dir / file_name
    if not path.exists():
        raise SystemExit(f"{path} not found. Create it with: {command}")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def require_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(
            f"{model_dir} not found. Download it with: "
            f"hf download elajeff/greek-pii-redactor-ner --local-dir {model_dir}"
        )
