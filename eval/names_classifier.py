"""Scores the name classifier against the inflection rules on names it was not trained on.

Reads a --names-dir built by train/train_inflection.py with a holdout. For each held-out name and
case: forward, whether the form from inflect_name, and the classifier's, is in the Wiktionary set;
read-back, whether nominative, and the classifier, give each Wiktionary form back as the name.
The classifier can't tell when its probability is below a threshold; right and wrong are printed
for several thresholds. Forward is also scored with the classifier answering at or above each of
FORWARD_THRESHOLDS and the rules below it; the threshold with the most names right where no role,
gender and case does worse than the rules alone is printed (the highest on a tie). Read-back is
also scored for Names.nominative, which looks at the lexicon and the Faker el_GR names first. Last,
both are scored on forms checked by hand.

    python eval/names_classifier.py --names-dir C:/Users/me/names-holdout
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from greek_pii_redactor.inflect import CASES, inflect_name, nominative
from greek_pii_redactor.names import HOLDOUT_FILE, MANUAL_NAMES, Names, load_names, read_names

THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)
FORWARD_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# (word, role, gender, case, forms accepted), checked by hand.
CHECKED_FORMS = [
    ("Μαλτέζος", "surname", "M", "vocative", {"Μαλτέζε"}),
    ("Ανάργυρος", "surname", "M", "vocative", {"Ανάργυρε"}),
    ("Τζαννετάκος", "first", "M", "vocative", {"Τζαννετάκο"}),
    ("Ντίνος", "first", "M", "vocative", {"Ντίνο"}),
    ("Ανδρόνικος", "first", "M", "vocative", {"Ανδρόνικε"}),
    ("Φόρος", "first", "M", "vocative", {"Φόρε", "Φόρο"}),
    ("Βούλγαρος", "surname", "M", "genitive", {"Βούλγαρου"}),
    ("Χαράλαμπος", "first", "M", "genitive", {"Χαράλαμπου", "Χαραλάμπου"}),
    ("Λαλές", "surname", "M", "genitive", {"Λαλέ"}),
    ("Στεφανάκις", "surname", "M", "genitive", {"Στεφανάκι"}),
    ("Λελούς", "surname", "M", "genitive", {"Λελού"}),
    ("Κεραμεύς", "surname", "M", "genitive", {"Κεραμέως"}),
    ("Δέδες", "surname", "M", "genitive", {"Δέδε"}),
    ("Κρέων", "first", "M", "accusative", {"Κρέοντα"}),
    ("Αγαμέμνων", "first", "M", "genitive", {"Αγαμέμνονος"}),
    ("Κυριακή", "first", "F", "genitive", {"Κυριακής"}),
    ("Αθηνά", "first", "F", "genitive", {"Αθηνάς"}),
    ("Καλούδα", "surname", "F", "genitive", {"Καλούδα"}),
    ("Ευφροσύνη", "surname", "F", "genitive", {"Ευφροσύνη"}),
]
# Per name and case: (classifier's form in the Wiktionary set, its probability, rules' form in it).
ForwardScore = tuple[bool, float, bool]


def share(part: int, whole: int) -> str:
    return f"{part / whole:>8.1%}" if whole else f"{'-':>8}"


def score_forward(held_out: list[dict], names: Names) -> dict[tuple, list[ForwardScore]]:
    forward_scores: dict[tuple, list[ForwardScore]] = defaultdict(list)
    for name in held_out:
        word, role, gender = name["word"], name["role"], name["gender"]
        for case in CASES:
            wiktionary = set(name[case])
            if not wiktionary:
                continue
            predicted, probability = names.predict(word, role, gender, case)
            rules_right = inflect_name(word, gender, case, role == "surname") in wiktionary
            forward_scores[role, gender, case].append(
                (predicted in wiktionary, probability, rules_right)
            )
    return forward_scores


def score_read_back(
    held_out: list[dict], names: Names
) -> tuple[dict[tuple, Counter], dict[tuple, Counter], dict[tuple, list[tuple[bool, float]]]]:
    """Per (role, gender, case): right / ? / wrong counts of the rules and of Names.nominative,
    and (read back right, probability) of the classifier for each Wiktionary form."""
    rules_read: dict[tuple, Counter] = defaultdict(Counter)
    names_read: dict[tuple, Counter] = defaultdict(Counter)
    classifier_read: dict[tuple, list[tuple[bool, float]]] = defaultdict(list)
    for name in held_out:
        word, role, gender = name["word"], name["role"], name["gender"]
        surname_only = role == "surname"
        for case in CASES:
            key = (role, gender, case)
            for form in set(name[case]):
                read = nominative(form, gender, case, surname_only)
                rules_read[key]["right" if read == word else "?" if read is None else "wrong"] += 1
                read = names.nominative(form, gender, case, surname_only)
                names_read[key]["right" if read == word else "?" if read is None else "wrong"] += 1
                read, probability = names.predict_nominative(form, role, gender, case)
                classifier_read[key].append((read == word, probability))
    return rules_read, names_read, classifier_read


def print_forward(
    forward_scores: dict[tuple, list[ForwardScore]], rules_read: dict[tuple, Counter]
) -> None:
    print("forward: share of names whose form is in the Wiktionary set")
    print("read-back: share of Wiktionary forms read back to the name, can't tell (?) or not\n")
    header = f"{'role':<9}{'sex':<4}{'case':<12}{'names':>7}{'rules':>8}{'classifier':>12}"
    print(header + f"{'right':>8}{'?':>8}{'wrong':>8}   (read-back by the rules)")
    for key in sorted(forward_scores):
        scores, read = forward_scores[key], rules_read[key]
        rules = sum(rules_right for _, _, rules_right in scores)
        classifier = sum(classifier_right for classifier_right, _, _ in scores)
        forms = sum(read.values())
        cells = [f"{key[0]:<9}{key[1]:<4}{key[2]:<12}{len(scores):>7}"]
        cells.append(share(rules, len(scores)))
        cells.append(f"{share(classifier, len(scores)):>12}")
        cells += [share(read[outcome], forms) for outcome in ("right", "?", "wrong")]
        print("".join(cells))


def print_forward_thresholds(forward_scores: dict[tuple, list[ForwardScore]]) -> None:
    print("\nforward with the rules below each threshold: share of names whose form is in the set")
    print(f"{'role':<9}{'sex':<4}{'case':<12}{'names':>7}{'rules':>8}", end="")
    print("".join(f"{threshold:>8}" for threshold in FORWARD_THRESHOLDS))
    right_at: Counter = Counter()
    below_rules = set()
    for key in sorted(forward_scores):
        scores = forward_scores[key]
        rules = sum(rules_right for _, _, rules_right in scores)
        cells = [f"{key[0]:<9}{key[1]:<4}{key[2]:<12}{len(scores):>7}{share(rules, len(scores))}"]
        for threshold in FORWARD_THRESHOLDS:
            right = sum(
                classifier_right if probability >= threshold else rules_right
                for classifier_right, probability, rules_right in scores
            )
            right_at[threshold] += right
            if right < rules:
                below_rules.add(threshold)
            cells.append(share(right, len(scores)))
        print("".join(cells))
    all_scores = [score for scores in forward_scores.values() for score in scores]
    rules = sum(rules_right for _, _, rules_right in all_scores)
    print(f"{'all':<25}{len(all_scores):>7}{share(rules, len(all_scores))}", end="")
    print("".join(share(right_at[threshold], len(all_scores)) for threshold in FORWARD_THRESHOLDS))
    allowed = [threshold for threshold in FORWARD_THRESHOLDS if threshold not in below_rules]
    best = max(allowed, key=lambda threshold: (right_at[threshold], threshold), default=None)
    print(f"most names right with no group below the rules: {best}")


def print_classifier_read_back(
    classifier_read: dict[tuple, list[tuple[bool, float]]],
    forward_scores: dict[tuple, list[ForwardScore]],
) -> None:
    print("\nread-back by the classifier: right / wrong at each threshold (the rest is ?)")
    print(f"{'role':<9}{'sex':<4}{'case':<12}{'forms':>7}", end="")
    print("".join(f"{threshold:>16}" for threshold in THRESHOLDS))
    for key in sorted(classifier_read):
        reads = classifier_read[key]
        cells = []
        for threshold in THRESHOLDS:
            sure = [right for right, probability in reads if probability >= threshold]
            cells.append(share(sum(sure), len(reads)) + share(len(sure) - sum(sure), len(reads)))
        print(f"{key[0]:<9}{key[1]:<4}{key[2]:<12}{len(reads):>7}" + "".join(cells))
    forward_fallbacks = sum(
        probability == 0 for scores in forward_scores.values() for _, probability, _ in scores
    )
    read_back_fallbacks = sum(
        probability == 0 for reads in classifier_read.values() for _, probability in reads
    )
    print(
        f"\nrules answered for the classifier: {forward_fallbacks} forward, "
        f"{read_back_fallbacks} read-back"
    )


def print_names_read_back(
    rules_read: dict[tuple, Counter], names_read: dict[tuple, Counter]
) -> None:
    print("\nread-back by the rules and by Names.nominative: share right / ? / wrong")
    print(f"{'role':<9}{'sex':<4}{'case':<12}{'forms':>7}{'rules':>24}{'Names.nominative':>24}")
    for key in sorted(names_read):
        cells = [f"{key[0]:<9}{key[1]:<4}{key[2]:<12}{sum(names_read[key].values()):>7}"]
        for read in (rules_read[key], names_read[key]):
            forms = sum(read.values())
            cells += [share(read[outcome], forms) for outcome in ("right", "?", "wrong")]
        print("".join(cells))


def print_checked_forms(names: Names) -> None:
    print("\nhand-checked forms (lexicon: the training names with the manual forms)")
    for word, role, gender, case, accepted in CHECKED_FORMS:
        rules = inflect_name(word, gender, case, surname_only=role == "surname")
        predicted, probability = names.predict(word, role, gender, case)
        listed = names.lookup(word, role, gender, case)
        if listed:
            lexicon = f"{'/'.join(listed)} {'ok' if set(listed) <= accepted else 'no'}"
        else:
            lexicon = "not in it"
        print(
            f"{word} {role} {gender} {case} -> {'/'.join(sorted(accepted))}: "
            f"rules {rules} {'ok' if rules in accepted else 'no'}, "
            f"classifier {predicted} ({probability:.2f}) "
            f"{'ok' if predicted in accepted else 'no'}, lexicon {lexicon}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names-dir", type=Path, required=True)
    args = parser.parse_args()
    # Redirected output on Windows is cp1252, which cannot print the Greek examples.
    sys.stdout.reconfigure(encoding="utf-8")
    holdout = args.names_dir / HOLDOUT_FILE
    if not holdout.is_file():
        raise SystemExit(
            f"{holdout} not found. Create it with: "
            f"python train/train_inflection.py --out {args.names_dir}"
        )
    if holdout.stat().st_size == 0:
        raise SystemExit(f"{holdout} is empty: train with --holdout above 0 to score")
    # read_names adds the hand-checked names, which train_inflection.py always trains on.
    held_out = [name for name in read_names(holdout) if name not in MANUAL_NAMES]
    names = load_names(args.names_dir)

    forward_scores = score_forward(held_out, names)
    rules_read, names_read, classifier_read = score_read_back(held_out, names)
    print_forward(forward_scores, rules_read)
    print_forward_thresholds(forward_scores)
    print_classifier_read_back(classifier_read, forward_scores)
    print_names_read_back(rules_read, names_read)
    print_checked_forms(names)


if __name__ == "__main__":
    main()
