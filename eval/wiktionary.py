"""Scores the name inflection rules against the case tables in Greek Wiktionary.

For every name in --names (written by eval/download_names.py), checks inflect_name (the name into
genitive, accusative and vocative) against its case table, and nominative (each form in the table
back to the name). Prints one table grouped by role, gender and the name's last two letters, then
totals for all names and for the names in the Faker lists the rules were built from. Every forward
disagreement and wrong read-back goes to --out, one JSON object per line.

    python eval/wiktionary.py
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from greek_pii_redactor.inflect import CASES, inflect_name, is_listed, nominative

ENDING_LETTERS = 2  # tells ος from ός and ας from ης, which the rules treat apart
MIN_GROUP = 20  # fewer names give a share too noisy to read; they go in one "other" row
READ_BACK = ("right", "can't tell", "wrong")


def new_counts() -> dict:
    return {
        "names": 0,
        "forward": {case: [0, 0] for case in CASES},  # agreeing, scored
        "read-back": Counter(),
    }


def add(counts: dict, other: dict) -> None:
    counts["names"] += other["names"]
    for case in CASES:
        counts["forward"][case][0] += other["forward"][case][0]
        counts["forward"][case][1] += other["forward"][case][1]
    counts["read-back"] += other["read-back"]


def row(label: list[str], counts: dict) -> str:
    cells = [f"{label[0]:<9}{label[1]:<4}{label[2]:<8}{counts['names']:>7}"]
    for case in CASES:
        agreeing, scored = counts["forward"][case]
        cells.append(f"{agreeing / scored:>8.1%}" if scored else f"{'-':>8}")
    read = sum(counts["read-back"].values())
    for outcome in READ_BACK:
        cells.append(f"{counts['read-back'][outcome] / read:>8.1%}" if read else f"{'-':>8}")
    return "".join(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names", type=Path, default=Path("data/wiktionary_names.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/wiktionary_disagreements.jsonl"))
    args = parser.parse_args()
    # Redirected output on Windows is cp1252, which cannot print the Greek endings.
    sys.stdout.reconfigure(encoding="utf-8")

    with args.names.open(encoding="utf-8") as file:
        names = [json.loads(line) for line in file]
    by_group: dict[tuple[str, str, str], dict] = defaultdict(new_counts)
    totals: dict[tuple[str, str, str], dict] = defaultdict(new_counts)
    disagreements = []
    for name in names:
        word, role, gender = name["word"], name["role"], name["gender"]
        surname_only = role == "surname"
        counts = new_counts()
        counts["names"] = 1
        for case in CASES:
            wiktionary = set(name[case])
            if not wiktionary:
                continue
            ours = inflect_name(word, gender, case, surname_only)
            counts["forward"][case][0] += ours in wiktionary
            counts["forward"][case][1] += 1
            found = {
                "word": word,
                "role": role,
                "gender": gender,
                "case": case,
                "wiktionary": sorted(wiktionary),
            }
            if ours not in wiktionary:
                disagreements.append({**found, "ours": ours, "kind": "forward"})
            for form in sorted(wiktionary):
                read = nominative(form, gender, case, surname_only)
                outcome = "right" if read == word else "can't tell" if read is None else "wrong"
                counts["read-back"][outcome] += 1
                if outcome == "wrong":
                    disagreements.append({**found, "form": form, "ours": read, "kind": "read-back"})
        add(by_group[role, gender, word[-ENDING_LETTERS:]], counts)
        add(totals[role, gender, "all"], counts)
        if is_listed(word):
            add(totals[role, gender, "listed"], counts)

    rows: dict[tuple[str, str, str], dict] = defaultdict(new_counts)
    for (role, gender, ending), counts in by_group.items():
        add(rows[role, gender, ending if counts["names"] >= MIN_GROUP else "other"], counts)
    header = f"{'role':<9}{'sex':<4}{'ending':<8}{'names':>7}"
    header += "".join(f"{case[:3]:>8}" for case in CASES)
    header += f"{'right':>8}{'?':>8}{'wrong':>8}"
    print("forward: share of names whose form is in the Wiktionary set for the case")
    print("read-back: share of Wiktionary forms read back to the name, can't tell (?) or not\n")
    print(header)
    for key, counts in sorted(rows.items(), key=lambda item: -item[1]["names"]):
        print(row(list(key), counts))
    print("\ntotals: all names, and listed (in the Faker lists the rules were built from)")
    print(header.replace("ending", "set   "))
    for key, counts in sorted(totals.items()):
        print(row(list(key), counts))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as file:
        for disagreement in disagreements:
            file.write(json.dumps(disagreement, ensure_ascii=False) + "\n")
    print(f"{len(disagreements)} disagreements written to {args.out}")


if __name__ == "__main__":
    main()
