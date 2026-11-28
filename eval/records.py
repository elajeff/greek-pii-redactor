"""Names, reads and samples the JSONL data files the eval and train scripts share, splits
WikiANN train into folds, and checks the tagger weights directories."""

import hashlib
import json
import random
from pathlib import Path

from greek_pii_redactor.rules import Span

SAMPLE_SIZE = 2000  # keeps the Presidio runs short; the synthetic file is already this size
SAMPLE_SEED = 7
MODEL_DIR_HELP = "weights directory from hf download elajeff/greek-pii-redactor-ner"
FOLDS = 5  # a model trained on the other four fifths still sees 80% of the data
FOLD_SEED = 0  # fixed, so a record lands in the same fold in every run
HASH_BYTES = 8  # 64 bits of the hash, far more than the modulo needs

# (file name, command that creates it)
DOWNLOAD_NER = "python eval/download_ner.py"
WIKIANN_TRAIN = ("wikiann_el_train.jsonl", DOWNLOAD_NER)
WIKIANN_VALIDATION = ("wikiann_el_validation.jsonl", DOWNLOAD_NER)
WIKIANN_TEST = ("wikiann_el_test.jsonl", DOWNLOAD_NER)
SYNTHETIC_TRAIN = (
    "synthetic_train.jsonl",
    "python eval/make_synthetic.py --seed 11 --split train --count 2000",
)
SYNTHETIC_TEST = (
    "synthetic_test.jsonl",
    "python eval/make_synthetic.py --seed 7 --split test --count 2000",
)


def load_records(data_dir: Path, file_name: str, command: str) -> list[dict]:
    path = data_dir / file_name
    if not path.exists():
        raise SystemExit(f"{path} not found. Create it with: {command}")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def person_spans(record: dict) -> set[Span]:
    return {Span(*span) for span in record["spans"] if span[2] == "PERSON"}


def fold_of(record: dict) -> int:
    digest = hashlib.sha256(f"{FOLD_SEED}:{record['text']}".encode()).digest()
    return int.from_bytes(digest[:HASH_BYTES]) % FOLDS


def training_records(
    data_dir: Path, *, seed: int, without_synthetic: bool, train_fraction: float,
    leave_out_fold: int | None,
) -> list[dict]:
    records = load_records(data_dir, *WIKIANN_TRAIN)
    if not without_synthetic:
        records.extend(load_records(data_dir, *SYNTHETIC_TRAIN))
    if leave_out_fold is not None:
        records = [record for record in records if fold_of(record) != leave_out_fold]
    if train_fraction < 1.0:
        records = random.Random(seed).sample(records, round(train_fraction * len(records)))
    return records


def fixed_sample(records: list[dict]) -> list[dict]:
    """SAMPLE_SIZE records drawn with SAMPLE_SEED, so every script scores the same WikiANN test
    sentences."""
    return random.Random(SAMPLE_SEED).sample(records, SAMPLE_SIZE)


def seeded_sample(records: list[dict], count: int, seed: int, path: Path) -> list[dict]:
    if count > len(records):
        raise SystemExit(f"--count {count} exceeds the {len(records)} records in {path}")
    return random.Random(seed).sample(records, count)


def require_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(
            f"{model_dir} not found. Download it with: "
            f"hf download elajeff/greek-pii-redactor-ner --local-dir {model_dir}"
        )


def require_epochs_file(model_dir: Path) -> None:
    """train/train_tagger.py writes epochs.json; the published weights have none."""
    if not (model_dir / "epochs.json").is_file():
        raise SystemExit(
            f"{model_dir / 'epochs.json'} not found. Train: python train/train_tagger.py"
        )
