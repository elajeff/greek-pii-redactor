"""Replaces personal data with placeholders like [PERSON_1] and puts it back in an LLM's reply.

The known cases where restore gets the Greek wrong are in docs/restore.md.
"""

import re
import unicodedata
from collections.abc import Callable, Iterator
from itertools import product
from typing import NamedTuple

from greek_pii_redactor.inflect import (
    accent_candidates,
    accent_on_second_to_last,
    drop_tonos_and_unseen_diaeresis,
    gender_of,
    inflect_name,
    is_listed,
    needs_accent,
    nominative,
    strip_accents,
)
from greek_pii_redactor.names import Names
from greek_pii_redactor.rules import Span, find_ids

KINDS = ("PERSON", "AFM", "AMKA", "IBAN", "PHONE", "EMAIL")
PLACEHOLDER_PATTERN = re.compile(rf"\[({'|'.join(KINDS)})_\d+\]")

# A one-letter stem would join unrelated names; Ζωή/Ζωής and Άρης/Άρη have two.
MIN_STEM = 2
# Case endings of one declension, normalised. Endings only match inside one group, so names with
# the same stem but different declensions (Νίκος, Νίκη, Νίκου/Νίκης) stay separate people.
ENDING_GROUPS = [
    {"οσ", "ου", "ο", "ε"},
    {"ησ", "η"},
    {"ασ", "α"},
    {"ωσ", "ω"},
]
ENDINGS_LONGEST_FIRST = sorted(set().union(*ENDING_GROUPS), key=len, reverse=True)

# Article -> (case, gender), None for a plural article, which has no gender.
ARTICLES = {
    "ο": ("nominative", "M"),
    "η": ("nominative", "F"),
    "οι": ("nominative", None),
    "ένας": ("nominative", "M"),
    "τον": ("accusative", "M"),
    "την": ("accusative", "F"),
    "τη": ("accusative", "F"),
    "στον": ("accusative", "M"),
    "στην": ("accusative", "F"),
    "στη": ("accusative", "F"),
    "τους": ("accusative", None),
    "τις": ("accusative", None),
    "του": ("genitive", "M"),
    "της": ("genitive", "F"),
    "των": ("genitive", None),
    "ενός": ("genitive", "M"),
}
# Title -> (case, gender). κυρία is also the accusative and vocative; a woman's name is the same
# in all three.
TITLES = {
    "κύριος": ("nominative", "M"),
    "κυρία": ("nominative", "F"),
    "κυρίου": ("genitive", "M"),
    "κυρίας": ("genitive", "F"),
    "κύριο": ("accusative", "M"),
    "κύριε": ("vocative", "M"),
}
# Greeting -> gender, None when the word has none. A name after one is a vocative.
GREETINGS = {
    "αγαπητέ": "M",
    "αγαπητή": "F",
    "καλημέρα": None,
    "καλησπέρα": None,
    "καλημέρα σας": None,
    "καλησπέρα σας": None,
    "γεια σου": None,
    "γεια σας": None,
}
# A greeting may be followed by a comma before the name: "Καλημέρα, [PERSON_1]".
GREETING_WORDS = "|".join(greeting.replace(" ", r"\s+") for greeting in GREETINGS)
WORD_BEFORE = re.compile(rf"\b({GREETING_WORDS}),?\s+$|(\w+)\s+$", re.IGNORECASE)

# Placeholder -> the forms seen in the input, each as {"text": ...}. A PERSON form also has
# "article", the lowercase word before it. Plain dicts and lists, so json.dumps works.
Placeholders = dict[str, list[dict[str, str]]]


class Context(NamedTuple):
    """What the word before a name asks for."""

    case: str
    gender: str | None
    after_title: bool


def normalise(name: str) -> str:
    return strip_accents(name).lower().replace("ς", "σ")


def split_ending(token: str) -> tuple[str, str]:
    for ending in ENDINGS_LONGEST_FIRST:
        if token.endswith(ending):
            return token[: -len(ending)], ending
    return token, ""


def tokens_match(first: str, second: str) -> bool:
    if first == second:
        return True
    first_stem, first_ending = split_ending(first)
    second_stem, second_ending = split_ending(second)
    return (
        first_stem == second_stem
        and len(first_stem) >= MIN_STEM
        and any({first_ending, second_ending} <= group for group in ENDING_GROUPS)
    )


def same_person(first: str, second: str) -> bool:
    first_tokens, second_tokens = normalise(first).split(), normalise(second).split()
    return len(first_tokens) == len(second_tokens) and all(
        tokens_match(a, b) for a, b in zip(first_tokens, second_tokens, strict=True)
    )


def id_key(kind: str, surface: str) -> str:
    if kind == "EMAIL":
        return surface.lower()
    return re.sub(r"[ -]", "", surface)


def same_value(kind: str, first: str, second: str) -> bool:
    if kind == "PERSON":
        return same_person(first, second)
    return id_key(kind, first) == id_key(kind, second)


def word_before(text: str, position: int) -> str:
    match = WORD_BEFORE.search(text[:position])
    return " ".join((match.group(1) or match.group(2)).lower().split()) if match else ""


def read_context(word: str) -> Context | None:
    if word in ARTICLES:
        return Context(*ARTICLES[word], after_title=False)
    if word in TITLES:
        return Context(*TITLES[word], after_title=True)
    if word in GREETINGS:
        return Context("vocative", GREETINGS[word], after_title=False)
    return None


def redact(
    text: str,
    detect_names: Callable[[str], list[Span]] | None = None,
    placeholders: Placeholders | None = None,
) -> tuple[str, Placeholders]:
    """Raises ValueError if the text already contains a placeholder, which restore would replace.

    Pass the placeholders of an earlier call to share them: the same value gets the same
    placeholder and new ones continue the numbering. They are extended in place and returned.
    """
    if PLACEHOLDER_PATTERN.search(text):
        raise ValueError("text already contains a placeholder such as [PERSON_1]")
    ids = find_ids(text)
    detected = detect_names(text) if detect_names else []
    # The rules are exact and the tagger is a guess, so a name span that overlaps an ID is dropped.
    detected = [n for n in detected if not any(n.start < i.end and i.start < n.end for i in ids)]
    # Overlapping name spans become one, so no tail of a name is left in the output.
    names: list[Span] = []
    for name in sorted(detected):
        if names and name.start < names[-1].end:
            names[-1] = Span(names[-1].start, max(names[-1].end, name.end), "PERSON")
        else:
            names.append(name)

    if placeholders is None:
        placeholders = {}
    pieces: list[str] = []
    position = 0
    for span in sorted(ids + names):
        surface = text[span.start : span.end]
        placeholder = next(
            (
                known
                for known, forms in placeholders.items()
                if known.startswith(f"[{span.kind}_")
                and same_value(span.kind, forms[0]["text"], surface)
            ),
            None,
        )
        if placeholder is None:
            number = sum(known.startswith(f"[{span.kind}_") for known in placeholders) + 1
            placeholder = f"[{span.kind}_{number}]"
            placeholders[placeholder] = []
        form = {"text": surface}
        if span.kind == "PERSON":
            form["article"] = word_before(text, span.start)
        if form not in placeholders[placeholder]:
            placeholders[placeholder].append(form)
        pieces += [text[position : span.start], placeholder]
        position = span.end
    pieces.append(text[position:])
    return "".join(pieces), placeholders


def accent_readings(name: str, guess_accents: bool) -> list[str]:
    """The name in lowercase where it was in capitals, because the ending rules match lowercase.
    With guess_accents, once for each accent its unaccented tokens can have."""
    # NFC, because upper() writes ΐ as Ι with two combining marks, lower() keeps the marks
    # separate, and the rules only know the single character ΐ.
    tokens = [
        unicodedata.normalize("NFC", token.lower()) if token.isupper() else token
        for token in name.split()
    ]
    if not guess_accents:
        return [" ".join(tokens)]
    return [
        " ".join(reading) for reading in product(*[accent_candidates(token) for token in tokens])
    ]


def fits_foreign_rule(seen: str, nominative_name: str) -> bool:
    """Whether every unaccented, unlisted token of the seen name that ends in -ος in the nominative
    has the accent on its second to last syllable."""
    return all(
        not needs_accent(seen_token)
        or is_listed(seen_token)
        or not strip_accents(token).endswith("ος")
        or accent_on_second_to_last(token)
        for seen_token, token in zip(seen.split(), nominative_name.split(), strict=True)
    )


def written_style(token: str) -> Callable[[str], str]:
    return next(
        (style for style in (str.upper, str.lower, str.title) if style(token) == token), str
    )


def nominative_candidates(
    forms: list[dict[str, str]],
    contexts: list[Context | None],
    gender: str | None,
    guess_accents: bool,
    names: Names | None,
) -> Iterator[tuple[str, list[str]]]:
    """For each seen form that can give a nominative: its text and the nominative of each accent
    reading."""
    read_back = names.nominative if names else nominative
    for form, context in zip(forms, contexts, strict=True):
        # No word right before the name (start of the text, or any punctuation) is taken as a
        # nominative.
        if form["article"] == "" or (context and context.case == "nominative"):
            yield form["text"], accent_readings(form["text"], guess_accents)
    if gender:
        for form, context in zip(forms, contexts, strict=True):
            if context and context.case != "nominative":
                lone_surname = context.after_title and len(form["text"].split()) == 1
                found = [
                    read_back(reading, gender, context.case, lone_surname)
                    for reading in accent_readings(form["text"], guess_accents)
                ]
                if found and all(found):
                    yield form["text"], found


def skip_shorter(
    candidates: Iterator[tuple[str, list[str]]],
) -> Iterator[tuple[str, list[str]]]:
    # A form with fewer tokens than one before it, such as a lone surname, would drop the first
    # name.
    longest_tried = 0
    for seen, readings in candidates:
        if len(seen.split()) < longest_tried:
            continue
        longest_tried = len(seen.split())
        yield seen, readings


def agreed_form(
    seen: str,
    readings: list[str],
    case: str,
    gender: str | None,
    after_title: bool,
    inflect: Callable[[str, str, str, bool], str],
) -> str | None:
    """The name in the case, written the way it was seen, if every reading gives the same text."""
    derived = set()
    for nominative_name in readings:
        name_gender = gender or gender_of(nominative_name)
        lone_surname = after_title and len(nominative_name.split()) == 1
        inflected = name_gender and inflect(nominative_name, name_gender, case, lone_surname)
        if not inflected:
            return None
        tokens = inflected.split()
        for i, seen_token in enumerate(seen.split()):
            if needs_accent(seen_token):
                tokens[i] = written_style(seen_token)(
                    drop_tonos_and_unseen_diaeresis(tokens[i], seen_token)
                )
            elif seen_token.isupper():
                tokens[i] = tokens[i].upper()
        derived.add(" ".join(tokens))
    return derived.pop() if len(derived) == 1 else None


def derive_form(
    forms: list[dict[str, str]], case: str, reply_gender: str | None, names: Names | None
) -> str | None:
    contexts = [read_context(form["article"]) for form in forms]
    gender = next(
        (context.gender for context in contexts if context and context.gender), reply_gender
    )
    # A single token is a surname only if the input had it after a title; a title in the reply
    # sets the case and gender, not the kind of name.
    after_title = any(context and context.after_title for context in contexts)
    inflect = names.inflect_name if names else inflect_name
    # Three tries, each only if the one before found nothing. First every accent the name could
    # have, and the text is used only if all give the same.
    candidates = nominative_candidates(forms, contexts, gender, True, names)
    for seen, readings in skip_shorter(candidates):
        form = agreed_form(seen, readings, case, gender, after_title, inflect)
        if form is not None:
            return form

    # Then, for a vocative, the accent a foreign name in -ος usually has, which gives -ο. This
    # always uses the plain rules, even with a names dir: it is the rule for foreign names.
    if case == "vocative":
        candidates = nominative_candidates(forms, contexts, gender, True, names)
        for seen, readings in skip_shorter(candidates):
            fitting = [name for name in readings if fits_foreign_rule(seen, name)]
            # Nothing filtered out means the first try already tried these readings.
            if len(fitting) == len(readings):
                continue
            form = agreed_form(seen, fitting, case, gender, after_title, inflect_name)
            if form is not None:
                return form

    # Last the name as written, with capitals lowercased, but not a listed name whose accents
    # disagree: that stays as seen.
    candidates = nominative_candidates(forms, contexts, gender, False, names)
    for seen, readings in skip_shorter(candidates):
        if any(
            needs_accent(token) and is_listed(token) and len(accent_candidates(token)) > 1
            for token in seen.split()
        ):
            continue
        form = agreed_form(seen, readings, case, gender, after_title, inflect)
        if form is not None:
            return form
    return None


def pick_form(forms: list[dict[str, str]], word: str, names: Names | None) -> str:
    context = read_context(word)
    if context is None:
        return forms[0]["text"]
    for form in forms:
        seen = read_context(form["article"])
        if seen and seen.case == context.case:
            return form["text"]
    return derive_form(forms, context.case, context.gender, names) or forms[0]["text"]


def restore(reply: str, placeholders: Placeholders, names: Names | None = None) -> str:
    def original(match: re.Match[str]) -> str:
        forms = placeholders.get(match.group())
        if forms is None:
            # Brackets off, so the next turn of a chat does not trip redact's placeholder check.
            return match.group()[1:-1]
        if match.group(1) == "PERSON":
            return pick_form(forms, word_before(reply, match.start()), names)
        return forms[0]["text"]

    return PLACEHOLDER_PATTERN.sub(original, reply)
