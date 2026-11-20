"""Scores detectors on exact span matches (start, end, kind), per dataset and per kind.

Detectors: "rules only" (find_ids from rules.py) and "Presidio + spaCy el" (Presidio's built-in
recognizers on top of the Greek spaCy model). With --model-dir, also "XLM-R tagger + rules": the
rules for IDs plus the fine-tuned PERSON tagger.

Setup: pip install -e ".[eval]" and then
    python -m spacy download el_core_news_lg
The model is not on PyPI. Its licence is CC BY-NC-SA 3.0, non-commercial use only.
For the tagger: pip install -e ".[train]", train it (needs a CUDA GPU) with
    python train/train_tagger.py --out <weights dir outside the repo>
and pass the same directory as --model-dir.

Data (in --data-dir, default data/), created by:
    python eval/make_synthetic.py --seed 7 --split test --count 2000
    python eval/download_ner.py

The real datasets only label PERSON, so only PERSON is scored on them.
"""

import argparse
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from greek_pii_redactor.rules import Span, find_ids

SAMPLE_SIZE = 2000  # keeps the Presidio runs short; the synthetic file is already this size
SAMPLE_SEED = 7
LANGUAGE = "el"
SPACY_MODEL = "el_core_news_lg"
ID_KINDS = ("AFM", "AMKA", "IBAN", "PHONE", "EMAIL")
ALL_KINDS = ("PERSON", *ID_KINDS)
# Presidio entities mapped to our kinds. Anything else it returns (LOCATION, URL, DATE_TIME ...)
# is dropped rather than counted as a false positive, because we only score the kinds we target.
PRESIDIO_KINDS = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "IBAN_CODE": "IBAN",
    "PHONE_NUMBER": "PHONE",
}

# file name, kinds scored, command that creates the file
DATASETS = [
    (
        "synthetic_test.jsonl",
        ALL_KINDS,
        "python eval/make_synthetic.py --seed 7 --split test --count 2000",
    ),
    # greek_legal_ner is left out: many real names in it have no PERSON label.
    ("wikiann_el_test.jsonl", ("PERSON",), "python eval/download_ner.py"),
]

Detector = Callable[[str], list[Span]]


def build_presidio() -> Detector:
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": LANGUAGE, "model_name": SPACY_MODEL}],
        }
    ).create_engine()
    # The built-in recognizers are only registered for English unless asked for Greek.
    registry = RecognizerRegistry(supported_languages=[LANGUAGE])
    registry.load_predefined_recognizers(languages=[LANGUAGE], nlp_engine=nlp_engine)
    analyzer = AnalyzerEngine(
        registry=registry, nlp_engine=nlp_engine, supported_languages=[LANGUAGE]
    )

    def detect(text: str) -> list[Span]:
        return [
            Span(found.start, found.end, PRESIDIO_KINDS[found.entity_type])
            for found in analyzer.analyze(text, language=LANGUAGE)
            if found.entity_type in PRESIDIO_KINDS
        ]

    return detect


def load_records(data_dir: Path, file_name: str, command: str) -> list[dict]:
    path = data_dir / file_name
    if not path.exists():
        raise SystemExit(f"{path} not found. Create it with: {command}")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.3f}" if denominator else "n/a"


def print_scores(
    detector: str, records: list[dict], predictions: list[list[Span]], kinds: tuple[str, ...]
) -> None:
    for kind in kinds:
        true_pos = false_pos = false_neg = 0
        for record, predicted in zip(records, predictions, strict=True):
            expected = {Span(*span) for span in record["spans"] if span[2] == kind}
            found = {span for span in predicted if span.kind == kind}
            true_pos += len(expected & found)
            false_pos += len(found - expected)
            false_neg += len(expected - found)
        print(
            f"{detector:<21}{kind:<8}{true_pos:>7}{false_pos:>7}{false_neg:>7}"
            f"{ratio(true_pos, true_pos + false_pos):>11}{ratio(true_pos, true_pos + false_neg):>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, help="weights from train/train_tagger.py")
    args = parser.parse_args()
    if args.model_dir and not args.model_dir.is_dir():
        raise SystemExit(f"{args.model_dir} not found. Train it with: python train/train_tagger.py")

    datasets = [
        (file_name, kinds, load_records(args.data_dir, file_name, command))
        for file_name, kinds, command in DATASETS
    ]
    detectors = {"rules only": find_ids, "Presidio + spaCy el": build_presidio()}
    if args.model_dir:
        # Imported here so the other detectors run without torch and transformers installed.
        from greek_pii_redactor.tagger import load_tagger

        tag_persons = load_tagger(args.model_dir)
        detectors["XLM-R tagger + rules"] = lambda text: sorted(find_ids(text) + tag_persons(text))

    for file_name, kinds, records in datasets:
        heading = f"{file_name}: {len(records)} sentences"
        if len(records) > SAMPLE_SIZE:
            records = random.Random(SAMPLE_SEED).sample(records, SAMPLE_SIZE)
            heading = f"{file_name}: {SAMPLE_SIZE} sentences sampled with seed {SAMPLE_SEED}"
        print(f"\n{heading}")
        print(f"{'detector':<21}{'kind':<8}{'TP':>7}{'FP':>7}{'FN':>7}{'precision':>11}{'recall':>9}")
        seconds = {}
        for name, detect in detectors.items():
            started = time.perf_counter()
            predictions = [
                [span for span in detect(record["text"]) if span.kind in kinds]
                for record in records
            ]
            seconds[name] = time.perf_counter() - started
            print_scores(name, records, predictions, kinds)
        times = ", ".join(f"{name} {s:.1f}" for name, s in seconds.items())
        print(f"Wall time in seconds, single run on this machine, not a benchmark: {times}")


if __name__ == "__main__":
    main()
