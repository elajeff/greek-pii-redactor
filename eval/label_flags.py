"""Picks WikiANN PERSON labels a model doubts, for a hand check (confident learning).

A gold name is flagged when a model that did not train on the sentence gives its tokens a
probability of O at least as high as it gives O tokens on average. A name the model finds that has
no gold label is flagged when its name score is at least the average name score of gold names.
Random test sentences and synthetic ones go in the output too, so the noise rate is not only
measured on the picked sentences.

Test sentences are scored by the models in --test-model-dir (trained on all of WikiANN train),
train sentences by the --fold-model-dir model that left their fold out (train/train_tagger.py
--leave-out-fold). Needs a CUDA GPU.

Data (in --data-dir, default data/): python eval/download_ner.py and
python eval/make_synthetic.py --seed 7 --split test --count 2000
Run:
    python eval/label_flags.py --test-model-dir A B C --fold-model-dir F0 F1 F2 F3 F4
        --out cases.json
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from records import SAMPLE_SEED, SAMPLE_SIZE, load_records, require_epochs_file
from transformers import AutoModelForTokenClassification, AutoTokenizer

from greek_pii_redactor.rules import Span
from greek_pii_redactor.tagger import decode_spans

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from train_tagger import FOLDS, fold_of, person_spans, tokenize

TRAIN_FLAG_SAMPLE = 50  # there are hundreds of train flags, too many to read by hand
TRAIN_SAMPLE_SEED = 11
RANDOM_SAMPLE = 50
RANDOM_SAMPLE_SEED = 13
SYNTHETIC_SAMPLE = 50
SYNTHETIC_SAMPLE_SEED = 17
BATCH_SIZE = 64  # no gradients are stored, so this fits in 12 GB

GOLD_NOT_NAME = "gold_not_name"
UNLABELLED_NAME = "unlabelled_name"
NER_COMMAND = "python eval/download_ner.py"
SYNTHETIC_COMMAND = "python eval/make_synthetic.py --seed 7 --split test --count 2000"


@dataclass
class Scored:
    offsets: list[tuple[int, int]]
    probs: list[list[float]]  # per token: P(O), P(B-PERSON), P(I-PERSON)


@dataclass
class Flag:
    kind: str
    span: Span
    score: float  # P(O) of a gold name, name score of an unlabelled one
    threshold: float


def score_sentences(model_dir: Path, records: list[dict]) -> list[Scored]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to("cuda").eval()
    encoded = tokenize(tokenizer, records)
    scored = []
    for first in range(0, len(records), BATCH_SIZE):
        ids = encoded["input_ids"][first : first + BATCH_SIZE]
        batch = tokenizer.pad({"input_ids": ids}, return_tensors="pt").to("cuda")
        with torch.no_grad():
            probs = model(**batch).logits.float().softmax(-1).cpu()
        for row, offsets in enumerate(encoded["offset_mapping"][first : first + BATCH_SIZE]):
            scored.append(Scored(offsets, probs[row, : len(offsets)].tolist()))
    return scored


def score_with_all(model_dirs: list[Path], records: list[dict]) -> list[Scored]:
    """The probabilities averaged over the models."""
    runs = [score_sentences(model_dir, records) for model_dir in model_dirs]
    return [
        Scored(
            sentence[0].offsets,
            torch.tensor([s.probs for s in sentence]).mean(dim=0).tolist(),
        )
        for sentence in zip(*runs, strict=True)
    ]


def score_by_fold(fold_dirs: dict[int, Path], records: list[dict]) -> list[Scored]:
    scored: list[Scored | None] = [None] * len(records)
    for fold, model_dir in fold_dirs.items():
        positions = [i for i, record in enumerate(records) if fold_of(record) == fold]
        for position, sentence in zip(
            positions, score_sentences(model_dir, [records[i] for i in positions]), strict=True
        ):
            scored[position] = sentence
    return scored


def read_fold_dirs(model_dirs: list[Path]) -> dict[int, Path]:
    fold_dirs: dict[int, Path] = {}
    for model_dir in model_dirs:
        require_epochs_file(model_dir)
        fold = json.loads((model_dir / "epochs.json").read_text(encoding="utf-8")).get(
            "leave_out_fold"
        )
        if fold is None:
            raise SystemExit(f"{model_dir} was not trained with --leave-out-fold")
        if fold in fold_dirs:
            raise SystemExit(f"{model_dir} and {fold_dirs[fold]} both left out fold {fold}")
        fold_dirs[fold] = model_dir
    missing = set(range(FOLDS)) - set(fold_dirs)
    if missing:
        raise SystemExit(f"no --fold-model-dir left out fold {sorted(missing)}")
    return fold_dirs


def name_score(sentence: Scored, span: Span) -> float | None:
    """Mean of 1 - P(O) over the tokens the span overlaps; None if the length cut removed them."""
    scores = [
        1 - probs[0]
        for (start, end), probs in zip(sentence.offsets, sentence.probs, strict=True)
        if start != end and start < span.end and span.start < end
    ]
    return sum(scores) / len(scores) if scores else None


def thresholds(scored: list[Scored], gold: list[set[Span]]) -> tuple[float, float]:
    """(t_O, t_name): the mean P(O) outside every gold span, the mean name score of gold spans."""
    outside = []
    name_scores = []
    for sentence, spans in zip(scored, gold, strict=True):
        outside += [
            probs[0]
            for (start, end), probs in zip(sentence.offsets, sentence.probs, strict=True)
            if start != end and not any(start < s.end and s.start < end for s in spans)
        ]
        name_scores += [
            score for span in spans if (score := name_score(sentence, span)) is not None
        ]
    return sum(outside) / len(outside), sum(name_scores) / len(name_scores)


def flags_of(
    sentence: Scored, gold: set[Span], outside_threshold: float, name_threshold: float
) -> list[Flag]:
    found = []
    for span in sorted(gold):
        score = name_score(sentence, span)
        # 1 - score is the mean P(O) of the span's tokens
        if score is not None and 1 - score >= outside_threshold:
            found.append(Flag(GOLD_NOT_NAME, span, 1 - score, outside_threshold))
    labels = [max(range(3), key=probs.__getitem__) for probs in sentence.probs]
    for span in decode_spans(sentence.offsets, labels):
        if any(span.start < g.end and g.start < span.end for g in gold):
            continue
        score = name_score(sentence, span)
        if score is not None and score >= name_threshold:
            found.append(Flag(UNLABELLED_NAME, span, score, name_threshold))
    return found


def reason(flag: Flag) -> str:
    if flag.kind == GOLD_NOT_NAME:
        return (
            f"labelled a name, but the model gives its tokens P(O) {flag.score:.3f}; "
            f"O tokens average {flag.threshold:.3f}"
        )
    return (
        f"not labelled, but the model gives it name score {flag.score:.3f}; "
        f"labelled names average {flag.threshold:.3f}"
    )


def case(set_name: str, index: int, record: dict, flag: Flag | None) -> dict:
    gold = sorted(person_spans(record))
    return {
        "id": f"{set_name}:{index}" + (f":{flag.span.start}-{flag.span.end}" if flag else ""),
        "set": set_name,
        "text": record["text"],
        "gold": [[span.start, span.end] for span in gold],
        "flag": [flag.span.start, flag.span.end] if flag else None,
        "reason": reason(flag) if flag else None,
    }


def flag_set(set_name: str, records: list[dict], scored: list[Scored]) -> list[tuple[int, Flag]]:
    gold = [person_spans(record) for record in records]
    outside_threshold, name_threshold = thresholds(scored, gold)
    flags = [
        (index, flag)
        for index, (sentence, spans) in enumerate(zip(scored, gold, strict=True))
        for flag in flags_of(sentence, spans, outside_threshold, name_threshold)
    ]
    print(f"{set_name}: t_O {outside_threshold:.3f}, t_name {name_threshold:.3f}")
    for kind in (GOLD_NOT_NAME, UNLABELLED_NAME):
        print(f"{set_name}: {sum(flag.kind == kind for _, flag in flags)} {kind} flags")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-model-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--fold-model-dir", type=Path, nargs=FOLDS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON file for the cases")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Scoring on the CPU is not supported.")
    if not args.out.parent.is_dir():
        raise SystemExit(f"{args.out.parent} does not exist")
    for model_dir in args.test_model_dir + args.fold_model_dir:
        if not model_dir.is_dir():
            raise SystemExit(f"{model_dir} not found")
    fold_dirs = read_fold_dirs(args.fold_model_dir)
    cases = []

    test_records = load_records(args.data_dir, "wikiann_el_test.jsonl", NER_COMMAND)
    test_records = random.Random(SAMPLE_SEED).sample(test_records, SAMPLE_SIZE)
    test_scored = score_with_all(args.test_model_dir, test_records)
    test_flags = flag_set("test", test_records, test_scored)
    cases += [case("test", index, test_records[index], flag) for index, flag in test_flags]

    train_records = load_records(args.data_dir, "wikiann_el_train.jsonl", NER_COMMAND)
    train_flags = flag_set("train", train_records, score_by_fold(fold_dirs, train_records))
    picked = random.Random(TRAIN_SAMPLE_SEED).sample(
        train_flags, min(TRAIN_FLAG_SAMPLE, len(train_flags))
    )
    print(f"train: {len(picked)} of {len(train_flags)} flags go to the page")
    cases += [case("train", index, train_records[index], flag) for index, flag in picked]

    random_sample = random.Random(RANDOM_SAMPLE_SEED).sample(range(SAMPLE_SIZE), RANDOM_SAMPLE)
    cases += [case("random", index, test_records[index], None) for index in sorted(random_sample)]

    synthetic = load_records(args.data_dir, "synthetic_test.jsonl", SYNTHETIC_COMMAND)
    sample = random.Random(SYNTHETIC_SAMPLE_SEED).sample(range(len(synthetic)), SYNTHETIC_SAMPLE)
    cases += [case("synthetic", index, synthetic[index], None) for index in sorted(sample)]

    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    for set_name in ("test", "train", "random", "synthetic"):
        print(f"{set_name}: {sum(c['set'] == set_name for c in cases)} cases")
    for kind in (GOLD_NOT_NAME, UNLABELLED_NAME):
        print(f"train picks: {sum(flag.kind == kind for _, flag in picked)} {kind}")
    print(f"{len(cases)} cases written to {args.out}")


if __name__ == "__main__":
    main()
