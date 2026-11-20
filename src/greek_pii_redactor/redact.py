"""Replaces personal data with placeholders like [PERSON_1] and puts it back in an LLM's reply.

Limit: restore can only use the name forms seen in the input. If the reply needs a case that the
input never had (a genitive after "του" when the input only had the nominative), it gets the first
form seen, so the Greek may be ungrammatical.

Limit: two forms of one person with no article before them (a bare nominative and a bare genitive)
cannot be told apart in the reply, so restore(redacted text) may not give the input back exactly.

Limit: an article that is not in ARTICLE_CASES, or words between the article and the placeholder
(του κ. [PERSON_1]), make restore use the first form seen.

Limit: a man and a woman can still merge when the tokens are identical (Κώστα), and surnames
spelled differently (Παπαδόπουλος, Παπαδόπουλλος) are separate people. The rare genitive
Γιάννου is not matched with Γιάννης.
"""

import re
import unicodedata
from collections.abc import Callable

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
]
ENDINGS_LONGEST_FIRST = sorted(set().union(*ENDING_GROUPS), key=len, reverse=True)

ARTICLE_CASES = {
    "ο": "nominative",
    "η": "nominative",
    "οι": "nominative",
    "ένας": "nominative",
    "τον": "accusative",
    "την": "accusative",
    "τη": "accusative",
    "στον": "accusative",
    "στην": "accusative",
    "στη": "accusative",
    "τους": "accusative",
    "τις": "accusative",
    "του": "genitive",
    "της": "genitive",
    "των": "genitive",
    "ενός": "genitive",
}

# Placeholder -> the forms seen in the input, each as {"text": ..., "article": ...} where article
# is the lowercase word before it. Plain dicts and lists, so json.dumps works.
Placeholders = dict[str, list[dict[str, str]]]


def normalise(name: str) -> str:
    decomposed = unicodedata.normalize("NFD", name.lower())
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    return unaccented.replace("ς", "σ")


def split_ending(token: str) -> tuple[str, str]:
    for ending in ENDINGS_LONGEST_FIRST:
        if token.endswith(ending):
            return token[: -len(ending)], ending
    return token, ""


def tokens_match(first: str, second: str) -> bool:
    if first == second:
        return True
    (first_stem, first_ending), (second_stem, second_ending) = (
        split_ending(first),
        split_ending(second),
    )
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
    match = re.search(r"(\w+)\s+$", text[:position])
    return match.group(1).lower() if match else ""


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
        form = {"text": surface, "article": word_before(text, span.start)}
        if form not in placeholders[placeholder]:
            placeholders[placeholder].append(form)
        pieces += [text[position : span.start], placeholder]
        position = span.end
    pieces.append(text[position:])
    return "".join(pieces), placeholders


def pick_form(forms: list[dict[str, str]], article: str) -> dict[str, str]:
    case = ARTICLE_CASES.get(article)
    for form in forms:
        if case and ARTICLE_CASES.get(form["article"]) == case:
            return form
    return forms[0]


def restore(reply: str, placeholders: Placeholders) -> str:
    def original(match: re.Match[str]) -> str:
        forms = placeholders.get(match.group())
        if forms is None:
            # Brackets off, so the next turn of a chat does not trip redact's placeholder check.
            return match.group()[1:-1]
        if match.group(1) == "PERSON":
            return pick_form(forms, word_before(reply, match.start()))["text"]
        return forms[0]["text"]

    return PLACEHOLDER_PATTERN.sub(original, reply)
