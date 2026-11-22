"""Measures how often restore puts a name in the wrong case: seen forms only versus the new restore.

Each record has 2 or 3 Greek people from the test-split name pools, mixed genders. Every person is
mentioned once in the input in a random case (nominative, genitive or accusative) and once in a
reply in a different random case (those three, or the vocative). Per person, either a full name
or a surname after a title. A full name has an article ("του Νίκου Παπαδοπούλου"), except the
vocative, which is «Καλημέρα, …». A surname has the title with its article or, in the vocative,
"Αγαπητέ κύριε" or "Αγαπητή κυρία". Each input case has its own sentence (FRAMES). The input is
redacted with a detector that knows the name spans, the reply is the placeholders in a template,
and each restored mention is compared with the expected form.

"Seen forms only" is what restore did before inflect.py: the form seen with the same article case,
else the first form seen. The expected forms come from inflect_name, the same rules the new restore
uses, so this measures choosing and deriving the case, not whether the rules are good Greek; that
is checked by reading the printouts of the rules.

    python eval/inflection.py --seed 7 --count 500
"""

import argparse
import random
import re
from collections import defaultdict

from make_synthetic import Person, article, first_names, surnames

from greek_pii_redactor.inflect import inflect_name
from greek_pii_redactor.redact import (
    ARTICLE_CASES,
    PLACEHOLDER_PATTERN,
    derive_form,
    read_context,
    redact,
    restore,
    word_before,
)
from greek_pii_redactor.rules import Span

INPUT_CASES = ("nominative", "genitive", "accusative")
REPLY_CASES = (*INPUT_CASES, "vocative")
TITLES = {
    ("M", "nominative"): "ο κύριος",
    ("M", "genitive"): "του κυρίου",
    ("M", "accusative"): "τον κύριο",
    ("M", "vocative"): "Αγαπητέ κύριε",
    ("F", "nominative"): "η κυρία",
    ("F", "genitive"): "της κυρίας",
    ("F", "accusative"): "την κυρία",
    ("F", "vocative"): "Αγαπητή κυρία",
}
FRAMES = {  # sentence around a name, so each input case is natural Greek
    "nominative": ("", " ήρθε. "),
    "genitive": ("Το σπίτι ", ". "),
    "accusative": ("Σχετικά με ", ". "),
}
LONE_SURNAME_SHARE = 0.5  # the other mentions are a full name
MENTION_SEPARATOR = " ; "  # never inside a name, so the restored reply splits back into mentions


def restore_seen_forms_only(reply: str, placeholders: dict[str, list[dict[str, str]]]) -> str:
    def original(match: re.Match[str]) -> str:
        forms = placeholders[match.group()]
        case = ARTICLE_CASES.get(word_before(reply, match.start()))
        return next(
            (form["text"] for form in forms if case and ARTICLE_CASES.get(form["article"]) == case),
            forms[0]["text"],
        )

    return PLACEHOLDER_PATTERN.sub(original, reply)


def pick_people(rng: random.Random) -> list[Person]:
    genders = [rng.choice("MF") for _ in range(rng.choice([2, 3]))]
    if len(set(genders)) == 1:
        genders[-1] = "F" if genders[0] == "M" else "M"
    return [
        Person(
            gender,
            rng.choice(first_names("test", True, gender)),
            rng.choice(surnames("test", True, gender)),
            True,
        )
        for gender in genders
    ]


def prefix(person: Person, case: str, surname_only: bool) -> str:
    if surname_only:
        return TITLES[person.gender, case] + " "
    if case == "vocative":
        return "Καλημέρα, "
    return article(person.gender, case, person.first) + " "


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    # (input case, reply case) -> mentions, wrong with seen forms only, wrong with the new restore
    by_cases = defaultdict(lambda: [0, 0, 0])
    fell_back = skipped = 0
    for _ in range(args.count):
        people = pick_people(rng)
        lone_surname = [rng.random() < LONE_SURNAME_SHARE for _ in people]
        names = [
            person.surname if lone else f"{person.first} {person.surname}"
            for person, lone in zip(people, lone_surname, strict=True)
        ]
        input_cases = [rng.choice(INPUT_CASES) for _ in people]
        reply_cases = [rng.choice([c for c in REPLY_CASES if c != seen]) for seen in input_cases]

        text, spans = "", []
        for person, name, case, lone in zip(people, names, input_cases, lone_surname, strict=True):
            form = inflect_name(name, person.gender, case, lone)
            before, after = FRAMES[case]
            introduction = prefix(person, case, lone)
            text += before + (introduction.capitalize() if case == "nominative" else introduction)
            spans.append(Span(len(text), len(text) + len(form), "PERSON"))
            text += form + after
        _, placeholders = redact(text, lambda _, found=spans: found)
        if len(placeholders) != len(people):  # two people with the same name stem merged
            skipped += 1
            continue

        starts, expected = [], []
        for person, name, case, lone in zip(people, names, reply_cases, lone_surname, strict=True):
            starts.append(prefix(person, case, lone))
            expected.append(starts[-1] + inflect_name(name, person.gender, case, lone))
        reply = MENTION_SEPARATOR.join(
            f"{start}[PERSON_{number}]" for number, start in enumerate(starts, 1)
        )
        old = restore_seen_forms_only(reply, placeholders).split(MENTION_SEPARATOR)
        new = restore(reply, placeholders).split(MENTION_SEPARATOR)
        for number, (start, wanted, seen_case, reply_case, old_text, new_text) in enumerate(
            zip(starts, expected, input_cases, reply_cases, old, new, strict=True), 1
        ):
            counts = by_cases[seen_case, reply_case]
            counts[0] += 1
            counts[1] += old_text != wanted
            counts[2] += new_text != wanted
            forms = placeholders[f"[PERSON_{number}]"]
            seen_cases = {read_context(form["article"])[0] for form in forms}
            _, gender, _ = read_context(word_before(start, len(start)))
            if (
                reply_case not in seen_cases
                and derive_form(forms, reply_case, gender, None) is None
            ):
                fell_back += 1

    print(
        f"{'input case':<12}{'reply case':<12}{'mentions':>9}{'seen only':>11}{'new':>8}  (wrong)"
    )
    total = [0, 0, 0]
    for (seen, replied), counts in sorted(by_cases.items()):
        mentions, old_wrong, new_wrong = counts
        print(
            f"{seen:<12}{replied:<12}{mentions:>9}"
            f"{old_wrong / mentions:>11.1%}{new_wrong / mentions:>8.1%}"
        )
        total = [a + b for a, b in zip(total, counts, strict=True)]
    mentions, old_wrong, new_wrong = total
    print(f"{'all':<24}{mentions:>9}{old_wrong / mentions:>11.1%}{new_wrong / mentions:>8.1%}")
    print(f"the new restore fell back to the first form seen for {fell_back} mentions")
    print(f"records skipped because two people merged: {skipped}")


if __name__ == "__main__":
    main()
