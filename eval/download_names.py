"""Writes the case tables of the modern Greek first names and surnames in Greek Wiktionary.

Output is one JSON object per line and per (word, role, gender):
{"word", "role", "gender", "genitive": [...], "accusative": [...], "vocative": [...]}, role
"first" or "surname", gender "M" or "F", each case the singular forms Wiktionary gives, sorted.
The extract lists a case's forms in table order with no mark of which one is preferred. A female
surname marked indeclinable counts as a table where every case is the name itself.
A table form that is not one word in modern Greek script is dropped and counted, and so is one
with more or fewer accents than the name (Μηλιά, genitive Μηλιας). An entry whose table has a
singular nominative other than the word is skipped and counted: the table is another word's
(Καλλιόπη filed with the table of Καλλιόπα).

Source: the kaikki.org extract of el.wiktionary (EXTRACT_URL), downloaded to --cache if the file
is missing. Wiktionary text is CC BY-SA 4.0.

    PYTHONIOENCODING=utf-8 python eval/download_names.py
"""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from greek_pii_redactor.inflect import CASES
from greek_pii_redactor.names import tonos_count

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
            nominatives = set()
            invariable = INDECLINABLE_CATEGORY in categories
            # 38 entries have no forms and 1546 forms have no tags.
            for form in entry.get("forms") or []:
                tags = form.get("tags") or []
                invariable = invariable or "invariable" in tags
                if form["source"] in TABLE_SOURCES and "singular" in tags:
                    # Polytonic forms (Ὺακίνθου) fall outside the block too.
                    if not isinstance(form.get("form"), str) or not GREEK_WORD.fullmatch(
                        form["form"]
                    ):
                        skipped["table form missing or not one word in Greek script"] += 1
                        continue
                    if "nominative" in tags:
                        nominatives.add(form["form"])
                    if tonos_count(form["form"]) != tonos_count(word):
                        skipped["table form with a different number of accents"] += 1
                        continue
                    for case in set(tags) & set(CASES):
                        cases[case].add(form["form"])
            if nominatives and word not in nominatives:
                skipped["entry with the table of another word"] += 1
                continue
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("data/el-extract.jsonl.gz"))
    parser.add_argument("--out", type=Path, default=Path("data/wiktionary_names.jsonl"))
    args = parser.parse_args()
    if not args.cache.exists():
        download(args.cache)

    skipped: Counter = Counter()
    tables = read_tables(args.cache, skipped)
    for reason, count in sorted(skipped.items()):
        print(f"skipped, {reason}: {count}")
    if not tables:
        raise SystemExit(f"no names with a case table in {args.cache}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as file:
        for (word, role, gender), table in tables.items():
            line = {"word": word, "role": role, "gender": gender}
            line |= {case: sorted(table[case]) for case in CASES}
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"{len(tables)} names written to {args.out}")


if __name__ == "__main__":
    main()
