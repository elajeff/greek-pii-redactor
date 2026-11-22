"""Scores the name inflection rules against the case tables in Greek Wiktionary.

For every modern Greek first name and surname in Wiktionary with a singular case table, checks
inflect_name (the name into genitive, accusative and vocative) against the table, and nominative
(each form in the table back to the name). A female surname marked indeclinable counts as a table
where every case is the name itself. Prints one table grouped by role, gender and the name's last
two letters, then totals for all names and for the names in the Faker lists the rules were built
from. Every forward disagreement and wrong read-back goes to --out, one JSON object per line.

Source: the kaikki.org extract of el.wiktionary (EXTRACT_URL), downloaded to --cache if the file
is missing. Wiktionary text is CC BY-SA 4.0.

    PYTHONIOENCODING=utf-8 python eval/wiktionary.py
"""

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from greek_pii_redactor.inflect import CASES, inflect_name, is_listed, nominative

EXTRACT_URL = "https://kaikki.org/dictionary/downloads/el/el-extract.jsonl.gz"
ROLE_CATEGORIES = {
    "Ανδρικά ονόματα (νέα ελληνικά)": ("first", "M"),
    "Γυναικεία ονόματα (νέα ελληνικά)": ("first", "F"),
    "Ανδρικά επώνυμα (νέα ελληνικά)": ("surname", "M"),
    "Γυναικεία επώνυμα (νέα ελληνικά)": ("surname", "F"),
}
# Indeclinable female surnames carry this category on the sense, or a header form tagged
# "invariable"; the entry's own categories never have it in this extract.
INDECLINABLE_CATEGORY = "Ουσιαστικά άκλιτα (νέα ελληνικά)"
# Wiktionary's declension templates fill one or the other (Κυριακή, Μιλτιάδης).
TABLE_SOURCES = ("declension", "inflection")
GREEK_WORD = re.compile(r"[Ά-ώ]+")  # the modern Greek block, tonos and diaeresis included
ENDING_LETTERS = 2  # tells ος from ός and ας from ης, which the rules treat apart
MIN_GROUP = 20  # fewer names give a share too noisy to read; they go in one "other" row
READ_BACK = ("right", "can't tell", "wrong")


def download(cache: Path) -> None:
    print(f"downloading {EXTRACT_URL} to {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_name(cache.name + ".part")
    # read() raises IncompleteRead on a cut connection; copying in chunks ends quietly with a
    # short file. The whole file in memory is about 116 MB.
    with urlopen(EXTRACT_URL) as response:
        part.write_bytes(response.read())
    part.replace(cache)


def read_tables(cache: Path, skipped: Counter) -> dict[tuple[str, str, str], dict[str, set[str]]]:
    """(word, role, gender) -> case -> the singular forms Wiktionary gives for that case."""
    tables: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    with gzip.open(cache, "rt", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            # Redirect lines have no lang_code.
            if entry.get("lang_code") != "el" or entry["pos"] != "name":
                continue
            # A sense with form_of or alt_of is a variant spelling (Θοδωρής of Θεόδωρος).
            senses = [s for s in entry["senses"] if not s.get("form_of") and not s.get("alt_of")]
            # Many senses have no categories key (18648 in the extract); null counts as none too.
            categories = {c for sense in senses for c in sense.get("categories") or []}
            roles = {ROLE_CATEGORIES[c] for c in categories if c in ROLE_CATEGORIES}
            if not roles:
                continue
            if len({gender for _, gender in roles}) > 1:
                skipped["entry with both genders"] += 1
                continue
            word = entry["word"]
            if not GREEK_WORD.fullmatch(word):
                skipped["not one word in Greek script"] += 1
                continue
            cases: dict[str, set[str]] = {case: set() for case in CASES}
            invariable = INDECLINABLE_CATEGORY in categories
            # 38 entries have no forms and 1546 forms have no tags.
            for form in entry.get("forms") or []:
                tags = form.get("tags") or []
                invariable = invariable or "invariable" in tags
                if form["source"] in TABLE_SOURCES and "singular" in tags:
                    for case in set(tags) & set(CASES):
                        cases[case].add(form["form"])
            for role, gender in sorted(roles):
                if any(cases.values()):
                    found = cases
                elif role == "surname" and gender == "F" and invariable:
                    found = {case: {word} for case in CASES}
                else:
                    skipped[f"{role} {gender} with no table"] += 1
                    continue
                table = tables.setdefault((word, role, gender), {case: set() for case in CASES})
                for case in CASES:
                    table[case] |= found[case]
    return tables


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
    parser.add_argument("--cache", type=Path, default=Path("data/el-extract.jsonl.gz"))
    parser.add_argument("--out", type=Path, default=Path("data/wiktionary_disagreements.jsonl"))
    args = parser.parse_args()
    if not args.cache.exists():
        download(args.cache)

    skipped: Counter = Counter()
    tables = read_tables(args.cache, skipped)
    by_group: dict[tuple[str, str, str], dict] = defaultdict(new_counts)
    totals: dict[tuple[str, str, str], dict] = defaultdict(new_counts)
    disagreements = []
    for (word, role, gender), table in tables.items():
        surname_only = role == "surname"
        counts = new_counts()
        counts["names"] = 1
        for case in CASES:
            wiktionary = table[case]
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
    print()
    for reason, count in sorted(skipped.items()):
        print(f"skipped, {reason}: {count}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as file:
        for disagreement in disagreements:
            file.write(json.dumps(disagreement, ensure_ascii=False) + "\n")
    print(f"{len(disagreements)} disagreements written to {args.out}")


if __name__ == "__main__":
    main()
