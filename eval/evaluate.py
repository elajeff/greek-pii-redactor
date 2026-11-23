"""Scores detectors on exact span matches (start, end, kind), per dataset and per kind.

Detectors: "rules only" (find_ids from rules.py) and "Presidio + spaCy el" (Presidio's built-in
recognizers on top of the Greek spaCy model). With --model-dir, also "XLM-R tagger + rules": the
rules for IDs plus the fine-tuned PERSON tagger. With --ollama-model, also "LLM + rules": the rules
for IDs plus the names a local model lists when asked through Ollama; every occurrence of a listed
name in the text is a PERSON span, and listed names that are not in the text are counted.

Setup: pip install -e ".[eval]" and then
    python -m spacy download el_core_news_lg
The model is not on PyPI. Its licence is CC BY-NC-SA 3.0, non-commercial use only.
For the tagger: pip install -e ".[train]", train it (needs a CUDA GPU) with
    python train/train_tagger.py --out <weights dir outside the repo>
and pass the same directory as --model-dir.
For the LLM row: install Ollama, run
    ollama pull qwen3:4b-instruct
and pass --ollama-model qwen3:4b-instruct. On an RTX 5070 that row took about 72 minutes per
dataset.

Data (in --data-dir, default data/), created by:
    python eval/make_synthetic.py --seed 7 --split test --count 2000
    python eval/download_ner.py

The real datasets only label PERSON, so only PERSON is scored on them. The synthetic sentences
also get a PERSON table for the Greek ones and one for the English ones.
"""

import argparse
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from survival import ask_model

from greek_pii_redactor.rules import Span, find_ids

SAMPLE_SIZE = 2000  # keeps the Presidio runs short; the synthetic file is already this size
SAMPLE_SEED = 7
LANGUAGE = "el"
SPACY_MODEL = "el_core_news_lg"
ID_KINDS = ("AFM", "AMKA", "IBAN", "PHONE", "EMAIL")
ALL_KINDS = ("PERSON", *ID_KINDS)
LANGUAGE_NAMES = {"el": "Greek", "en": "English"}
TABLE_HEADER = (
    f"{'detector':<21}{'kind':<8}{'TP':>7}{'FP':>7}{'FN':>7}{'precision':>11}{'recall':>9}"
)
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

NAME_INSTRUCTION = (
    "List every person's name that appears in the user's text, copied exactly as it is written "
    "there, one per line. Only people, not places, organisations or works. "
    "If there is none, reply with NONE."
)
LLM_SEED = 7  # any fixed value; at temperature 0 reruns still found up to 8 names more or fewer

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


def persons_in_reply(reply: str, text: str) -> tuple[list[Span], int]:
    """Returns a PERSON span for every occurrence in text of each name line in the model's reply,
    and the number of lines that do not occur in text."""
    persons = []
    not_in_text = 0
    for line in reply.splitlines():
        name = line.strip().lstrip("-*0123456789. ").strip()
        if not name or name == "NONE":
            continue
        start = text.find(name)
        if start == -1:
            not_in_text += 1
        while start != -1:
            persons.append(Span(start, start + len(name), "PERSON"))
            start = text.find(name, start + len(name))
    return persons, not_in_text


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
    parser.add_argument("--ollama-model", help="adds the LLM + rules row, e.g. qwen3:4b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
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
    if args.ollama_model:

        def ask_for_persons(text: str) -> list[Span]:
            nonlocal llm_names_not_in_text
            reply = ask_model(args.ollama_url, args.ollama_model, LLM_SEED, NAME_INSTRUCTION, text)
            persons, not_in_text = persons_in_reply(reply, text)
            llm_names_not_in_text += not_in_text
            return sorted(find_ids(text) + persons)

        detectors["LLM + rules"] = ask_for_persons
        print(f"LLM + rules: {args.ollama_model} at {args.ollama_url}, seed {LLM_SEED}")

    for file_name, kinds, records in datasets:
        heading = f"{file_name}: {len(records)} sentences"
        if len(records) > SAMPLE_SIZE:
            records = random.Random(SAMPLE_SEED).sample(records, SAMPLE_SIZE)
            heading = f"{file_name}: {SAMPLE_SIZE} sentences sampled with seed {SAMPLE_SEED}"
        print(f"\n{heading}")
        print(TABLE_HEADER)
        seconds = {}
        predictions_by_detector = {}
        llm_names_not_in_text = 0
        for name, detect in detectors.items():
            started = time.perf_counter()
            predictions = [
                [span for span in detect(record["text"]) if span.kind in kinds]
                for record in records
            ]
            seconds[name] = time.perf_counter() - started
            predictions_by_detector[name] = predictions
            print_scores(name, records, predictions, kinds)
        if args.ollama_model:
            print(f"LLM names not in the text: {llm_names_not_in_text}")
        if "language" in records[0]:
            for language, language_name in LANGUAGE_NAMES.items():
                indices = [i for i, record in enumerate(records) if record["language"] == language]
                print(f"\n{file_name}, PERSON, {language_name} sentences ({len(indices)})")
                print(TABLE_HEADER)
                for name, predictions in predictions_by_detector.items():
                    print_scores(
                        name,
                        [records[i] for i in indices],
                        [predictions[i] for i in indices],
                        ("PERSON",),
                    )
        times = ", ".join(f"{name} {s:.1f}" for name, s in seconds.items())
        print(f"Wall time in seconds, single run on this machine, not a benchmark: {times}")


if __name__ == "__main__":
    main()
