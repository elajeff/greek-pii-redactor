"""Trains the name inflection classifiers on the Wiktionary case tables.

One logistic regression per direction (nominative to case form, and back) and case, over the
features of greek_pii_redactor.names. Each (name, form) pair of the training names is one row;
a case with two forms gives two rows. Female surnames are left out: they never change, so the
classifier is not asked about them.

Setup: pip install -e ".[names]", then eval/download_names.py.
Run (the output directory must be outside the repo, nothing from it is committed):
    python train/train_inflection.py --out C:/Users/me/greek-pii-redactor-weights/names
--holdout 0.1 keeps a tenth of the names out of training, in holdout.jsonl, for
eval/names_classifier.py. --holdout 0 trains on all names and leaves holdout.jsonl empty.
"""

import argparse
import json
import random
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from greek_pii_redactor.inflect import CASES
from greek_pii_redactor.names import (
    LEXICON_FILE,
    MODELS_FILE,
    derive_edit,
    features,
    is_female_surname,
    read_names,
)

HOLDOUT_FILE = "holdout.jsonl"
DIRECTIONS = ("forward", "reverse")  # nominative to a case form, and back


def write_names(path: Path, names: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for name in names:
            file.write(json.dumps(name, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for the names files")
    parser.add_argument("--names", type=Path, default=Path("data/wiktionary_names.jsonl"))
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    names = list(read_names(args.names))
    # All roles of a word go to the same side, so a held-out word is unseen in any role.
    words = sorted({name["word"] for name in names})
    random.Random(args.seed).shuffle(words)
    held_out = set(words[: round(args.holdout * len(words))])
    training = [name for name in names if name["word"] not in held_out]
    args.out.mkdir(parents=True, exist_ok=True)
    write_names(args.out / LEXICON_FILE, training)
    write_names(args.out / HOLDOUT_FILE, [n for n in names if n["word"] in held_out])
    print(f"{len(training)} names for training, {len(names) - len(training)} held out")

    models = {}
    started = time.perf_counter()
    for case in CASES:
        for direction in DIRECTIONS:
            rows, edits = [], []
            for name in training:
                if is_female_surname(name["role"], name["gender"]):
                    continue
                first_name = name["role"] == "first"
                for form in name[case]:
                    source, target = (
                        (name["word"], form) if direction == "forward" else (form, name["word"])
                    )
                    rows.append(features(source, name["role"], name["gender"]))
                    edit = derive_edit(source, target, first_name)
                    edits.append(json.dumps(edit, ensure_ascii=False))
            model = make_pipeline(DictVectorizer(), LogisticRegression())
            models[direction, case] = model.fit(rows, edits)
            print(f"{direction} {case}: {len(rows)} pairs, {len(set(edits))} edits")
    joblib.dump(models, args.out / MODELS_FILE)
    print(f"trained in {time.perf_counter() - started:.0f} s, written to {args.out}")


if __name__ == "__main__":
    main()
