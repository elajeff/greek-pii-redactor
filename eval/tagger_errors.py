"""Shows what the PERSON tagger gets wrong, for one or more trained directories.

Per directory it prints: exact-span PERSON scores on the same WikiANN el test sample as
evaluate.py; the errors split into exact, wrong boundary, missed and extra, with examples; recall
on names seen and not seen in training; recall on names with no Greek-name token; three probes on
synthetic_test.jsonl (lowercased name, surname alone, no article before the name); and word
recall on greek_legal_ner test: its PERSON spans merge people and carry titles, so it is scored
by word, and its names are real people's, so only counts are printed from it. With several
directories it ends with the mean and standard deviation of WikiANN F1, of each probe's edited
recall and of the two greek_legal_ner word shares.

Setup: the same as the tagger in evaluate.py (pip install -e ".[train]", train with
train/train_tagger.py), plus data/wiktionary_names.jsonl from eval/download_names.py.
Run:
    python eval/tagger_errors.py --model-dir <weights dir> [<weights dir> ...]
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

from evaluate import SAMPLE_SEED, SAMPLE_SIZE, load_records

from greek_pii_redactor.inflect import accents_by_plain_form, strip_accents
from greek_pii_redactor.rules import Span
from greek_pii_redactor.tagger import load_tagger

DOWNLOAD_NER = "python eval/download_ner.py"
TRAIN_FILES = (
    ("wikiann_el_train.jsonl", DOWNLOAD_NER),
    ("wikiann_el_validation.jsonl", DOWNLOAD_NER),
    ("synthetic_train.jsonl", "python eval/make_synthetic.py --seed 11 --split train --count 2000"),
)
SYNTHETIC_TEST = (
    "synthetic_test.jsonl",
    "python eval/make_synthetic.py --seed 7 --split test --count 2000",
)
LEGAL_TEST = ("greek_legal_ner_test.jsonl", DOWNLOAD_NER)
EXAMPLE_COUNT = 10  # enough to see the pattern in an error bucket without filling the screen
ARTICLES = ("ο", "η", "του", "της", "τον", "την")
GREEK_LETTER = re.compile("[Ά-ώ]")
ARTICLE_BEFORE = re.compile(rf"(?<!\w)({'|'.join(ARTICLES)}) $")

Pairs = set[tuple[int, int]]


def person_pairs(record: dict) -> Pairs:
    return {(start, end) for start, end, kind in record["spans"] if kind == "PERSON"}


def overlaps(first: tuple[int, int], others: Pairs) -> bool:
    return any(first[0] < end and start < first[1] for start, end in others)


def lowercase(text: str, start: int, end: int) -> tuple[int, str] | None:
    return start, text[start:end].lower()


def surname_alone(text: str, start: int, end: int) -> tuple[int, str] | None:
    words = text[start:end].split()
    return (start, words[-1]) if len(words) == 2 else None


def without_article(text: str, start: int, end: int) -> tuple[int, str] | None:
    article = ARTICLE_BEFORE.search(text[:start])
    return (article.start(), text[start:end]) if article else None


PROBES = (
    ("lowercase", lowercase),
    ("surname alone", surname_alone),
    ("no article", without_article),
)


def edit_record(
    record: dict, rewrite: Callable[[str, int, int], tuple[int, str] | None]
) -> tuple[str, Pairs, Pairs, Pairs] | None:
    """The record's text with every Greek PERSON span that `rewrite` accepts replaced, and the
    PERSON spans moved to match. Returns text, all PERSON spans, the edited ones, and where the
    edited ones were before. None if no span was edited."""
    text = record["text"]
    out = ""
    copied = 0
    everyone: Pairs = set()
    edited: Pairs = set()
    before: Pairs = set()
    for start, end in sorted(person_pairs(record)):
        edit = rewrite(text, start, end) if GREEK_LETTER.search(text[start:end]) else None
        cut, name = edit if edit else (start, text[start:end])
        out += text[copied:cut]
        new_span = (len(out), len(out) + len(name))
        out += name
        copied = end
        everyone.add(new_span)
        if edit:
            edited.add(new_span)
            before.add((start, end))
    out += text[copied:]
    return (out, everyone, edited, before) if edited else None


def exact_scores(gold: list[Pairs], found: list[Pairs]) -> tuple[int, int, int]:
    true_pos = sum(len(expected & got) for expected, got in zip(gold, found, strict=True))
    false_pos = sum(len(got - expected) for expected, got in zip(gold, found, strict=True))
    false_neg = sum(len(expected - got) for expected, got in zip(gold, found, strict=True))
    return true_pos, false_pos, false_neg


def f1_score(true_pos: int, false_pos: int, false_neg: int) -> float:
    return 2 * true_pos / (2 * true_pos + false_pos + false_neg)


def recall_line(label: str, hits: list[bool]) -> str:
    return f"{label:<30}{sum(hits):>5} of {len(hits):<5}recall {sum(hits) / len(hits):.3f}"


def known_greek_words(data_dir: Path) -> set[str]:
    known = set(accents_by_plain_form())
    for record in load_records(data_dir, "wiktionary_names.jsonl", "python eval/download_names.py"):
        for form in (
            record["word"],
            *record["genitive"],
            *record["accusative"],
            *record["vocative"],
        ):
            known.add(strip_accents(form).lower())
    return known


def seen_names(data_dir: Path) -> set[str]:
    return {
        record["text"][start:end].casefold()
        for file_name, command in TRAIN_FILES
        for record in load_records(data_dir, file_name, command)
        for start, end in person_pairs(record)
    }


def run_probes(
    tag: Callable[[str], list[Span]], probe_records: list[dict]
) -> list[tuple[str, int, float, float, int]]:
    rows = []
    for name, rewrite in PROBES:
        edited_records = [
            (record, edit) for record in probe_records if (edit := edit_record(record, rewrite))
        ]
        spans = hits = before_hits = false_pos = 0
        for record, (text, everyone, edited, before) in edited_records:
            found = {(span.start, span.end) for span in tag(text)}
            before_found = {(span.start, span.end) for span in tag(record["text"])}
            spans += len(edited)
            hits += len(edited & found)
            before_hits += len(before & before_found)
            false_pos += len(found - everyone)
        rows.append((name, spans, before_hits / spans, hits / spans, false_pos))
    return rows


def legal_word_coverage(
    tag: Callable[[str], list[Span]], records: list[dict]
) -> tuple[int, int, int, int]:
    words = covered = caps_words = caps_covered = 0
    for record in records:
        gold = person_pairs(record)
        if not gold:
            continue
        predicted = {(s.start, s.end) for s in tag(record["text"])}
        for start, end in gold:
            for word in re.finditer(r"\w+", record["text"][start:end]):
                word_start, word_end = start + word.start(), start + word.end()
                inside = any(s <= word_start and word_end <= e for s, e in predicted)
                words += 1
                covered += inside
                if word.group().isupper():
                    caps_words += 1
                    caps_covered += inside
    return words, covered, caps_words, caps_covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    # Redirected output on Windows is cp1252, which cannot print the Greek examples.
    sys.stdout.reconfigure(encoding="utf-8")
    for model_dir in args.model_dir:
        if not (model_dir / "epochs.json").is_file():
            raise SystemExit(
                f"{model_dir / 'epochs.json'} not found. Train: python train/train_tagger.py"
            )

    wikiann = load_records(args.data_dir, "wikiann_el_test.jsonl", DOWNLOAD_NER)
    sample = random.Random(SAMPLE_SEED).sample(wikiann, SAMPLE_SIZE)
    probe_records = load_records(args.data_dir, *SYNTHETIC_TEST)
    legal_records = load_records(args.data_dir, *LEGAL_TEST)
    seen = seen_names(args.data_dir)
    known = known_greek_words(args.data_dir)

    f1_per_dir: list[float] = []
    probe_per_dir: list[list[float]] = []
    legal_per_dir: list[tuple[float, float]] = []
    for model_dir in args.model_dir:
        started = time.perf_counter()
        with (model_dir / "epochs.json").open(encoding="utf-8") as file:
            setup = json.load(file)
        print(
            f"\n== {model_dir}: seed {setup['seed']}, train fraction {setup['train_fraction']}, "
            f"{setup['train_records']} train records, "
            f"without synthetic {setup['without_synthetic']}"
        )
        tag = load_tagger(model_dir)

        found = [{(s.start, s.end) for s in tag(record["text"])} for record in sample]
        gold = [person_pairs(record) for record in sample]
        true_pos, false_pos, false_neg = exact_scores(gold, found)
        f1 = f1_score(true_pos, false_pos, false_neg)
        f1_per_dir.append(f1)
        print(
            f"WikiANN test, {SAMPLE_SIZE} sentences sampled with seed {SAMPLE_SEED}: PERSON "
            f"TP {true_pos}, FP {false_pos}, FN {false_neg}, "
            f"precision {true_pos / (true_pos + false_pos):.3f}, "
            f"recall {true_pos / (true_pos + false_neg):.3f}, F1 {f1:.3f}"
        )

        exact = overlapping = 0
        wrong_boundary: list[tuple[str, str]] = []
        missed: list[str] = []
        extra: list[str] = []
        gold_count = predicted_count = 0
        gold_hits: list[tuple[str, bool]] = []
        for record, expected, got in zip(sample, gold, found, strict=True):
            text = record["text"]
            gold_count += len(expected)
            predicted_count += len(got)
            exact += len(expected & got)
            for start, end in sorted(expected):
                gold_hits.append((text[start:end], (start, end) in got))
                if (start, end) in got:
                    continue
                touching = [p for p in sorted(got) if p[0] < end and start < p[1]]
                if touching:
                    wrong_boundary.append(
                        (text[start:end], " | ".join(text[a:b] for a, b in touching))
                    )
                else:
                    missed.append(text[start:end])
            for start, end in sorted(got - expected):
                if overlaps((start, end), expected):
                    overlapping += 1
                else:
                    extra.append(text[start:end])
        print(
            f"Errors: exact {exact}, wrong boundary {len(wrong_boundary)}, missed {len(missed)}, "
            f"extra {len(extra)}"
        )
        print(
            f"  gold {gold_count} = {exact} + {len(wrong_boundary)} + {len(missed)}; predicted "
            f"{predicted_count} = {exact} + {len(extra)} + {overlapping} overlapping a gold span"
        )
        print("  wrong boundary (gold -> predicted):")
        for gold_text, predicted_text in wrong_boundary[:EXAMPLE_COUNT]:
            print(f"    {gold_text!r} -> {predicted_text!r}")
        print("  missed:", missed[:EXAMPLE_COUNT])
        print("  extra:", extra[:EXAMPLE_COUNT])

        print("Seen in training or not (casefolded span text):")
        print(" ", recall_line("seen", [hit for text, hit in gold_hits if text.casefold() in seen]))
        print(
            " ",
            recall_line("unseen", [hit for text, hit in gold_hits if text.casefold() not in seen]),
        )

        no_greek_name = [
            hit
            for text, hit in gold_hits
            if GREEK_LETTER.search(text)
            and not any(strip_accents(word).lower() in known for word in re.findall(r"\w+", text))
        ]
        print(recall_line("No Greek-name token", no_greek_name))

        print("Probes on synthetic_test.jsonl (PERSON spans with a Greek letter):")
        print(
            f"  {'probe':<15}{'spans':>7}{'recall before':>15}{'recall edited':>15}"
            f"{'FP edited':>11}"
        )
        probes = run_probes(tag, probe_records)
        for name, spans, before_recall, edited_recall, probe_fp in probes:
            print(
                f"  {name:<15}{spans:>7}{before_recall:>15.3f}{edited_recall:>15.3f}{probe_fp:>11}"
            )
        probe_per_dir.append([edited_recall for _, _, _, edited_recall, _ in probes])

        words, covered, caps_words, caps_covered = legal_word_coverage(tag, legal_records)
        legal_per_dir.append((covered / words, caps_covered / caps_words))
        print(
            f"greek_legal_ner test, gold PERSON words inside one predicted span: "
            f"{covered} of {words} ({covered / words:.3f}); all-caps words {caps_covered} of "
            f"{caps_words} ({caps_covered / caps_words:.3f})"
        )
        print(f"Time for this directory: {time.perf_counter() - started:.1f} s")

    if len(args.model_dir) > 1:
        print(f"\nMean and standard deviation over {len(args.model_dir)} directories:")
        series = {
            "WikiANN F1": f1_per_dir,
            **{
                f"{name} edited recall": [row[index] for row in probe_per_dir]
                for index, (name, _) in enumerate(PROBES)
            },
            "legal word share": [share for share, _ in legal_per_dir],
            "legal all-caps word share": [share for _, share in legal_per_dir],
        }
        for label, values in series.items():
            print(f"  {label:<28}{statistics.mean(values):.3f} {statistics.stdev(values):.3f}")


if __name__ == "__main__":
    main()
