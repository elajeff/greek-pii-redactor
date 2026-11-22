"""Replaces personal data with placeholders like [PERSON_1] and puts it back in an LLM's reply.

Limit: without a names dir, restore builds the case a reply needs from the forms seen in the
input and the rules in inflect.py, so the Greek can still be wrong where the rules cannot tell: a
seen genitive in -ου gives the nominative only for a 2-syllable first name (Νίκου), -ίου
(Γεωργίου), -αίου (Ματθαίου) and -όπουλου/-οπούλου, so Κωνσταντίνου, Αλεξάνδρου and other
surnames in -ου are not derived from it; the folk first names Σπήλιος and Στέλλιος (read as
learned), a lone female name in -η, -ή or -ά that Faker does not list as a first name only (left as
it is), and surname-first order (Παπαδοπούλου Μαρία). Latin-script names have no rules, so they use
the seen forms; Greek-script foreign names are inflected like Greek ones (Τόμας Μίλερ gives Τόμα
Μίλερ). When nothing can be derived, restore uses the first form seen.

Limit: without a names dir, a first name of 4+ syllables in paroxytone -ος gets the vocative -ε
(Κωνσταντίνε); for some (Λεονάρδο/Λεονάρδε, Βενιζέλο/Βενιζέλε) both are in use and -ε is chosen.
When the genitive moves the accent onto ι or υ after a vowel, the diaeresis is lost: Βάιος
gives Βαίου (Βαΐου is right).

With a names dir (names.py), each token's forms come from the Wiktionary lexicon, or for a name
not in it from the classifier, and from the rules where the classifier is less than
FORWARD_THRESHOLD sure. Seen forms the lexicon lists are read back from it: «του Αλεξάνδρου» gives
«Ο Αλέξανδρος», Σπήλιος gives «του Σπήλιου», Βενιζέλος «Καλημέρα, Βενιζέλο» and Κρέων «του
Κρέοντα». A male surname in -ου that Faker lists as one stays as it is in the genitive: «του
κυρίου Σακελλαρίου» gives «Ο κύριος Σακελλαρίου». A form not in the lexicon is read back by the
rules when they give a known name, one in the lexicon or in Faker's el_GR lists («κύριε Γιαννακέα»
gives «Ο κύριος Γιαννακέας»). Otherwise the classifier reads it back when it is at least
READ_BACK_THRESHOLD sure, and the rules when it is less sure («κύριε Ζβορκέα» gives «Ο κύριος
Ζβορκέας»). A name from the classifier that is not known gives way to a known name spelt the same
without accents whose form by the rules is the one seen: «του Ανδρονίκου» gives «Ο Ανδρόνικος»,
not Ανδρονίκος.

Limit: with a names dir, a surname in -ός that is not in the lexicon, where its -ούς twin is, has
its genitive -ού read back as the twin. With the --holdout 0.1 --seed 7 build, Ντουρουντού gives
Ντουρουντούς; the --holdout 0 build lists both, so the rules' Ντουρουντός is used.

Limit: with a names dir, a male surname in -ου that neither the lexicon nor Faker knows, and that
the rules cannot read back, can be given a misspelt nominative by the classifier when it is at
least READ_BACK_THRESHOLD sure. With the --holdout 0.1 --seed 7 build, «του κυρίου
Χατζηκοκκίνου» gives «Ο κύριος Χατζηκοκκίνος»; the --holdout 0 build lists the name, so it gives
«Ο κύριος Χατζηκόκκινος».

Limit: two forms of one person with no article before them (a bare nominative and a bare genitive)
cannot be told apart in the reply, so restore(redacted text) may not give the input back exactly.

Limit: a word before the placeholder that is not an article, title or greeting (a number counts:
«Ήρθαν 3 Νίκος Παπαδόπουλος.» stores the article "3" and gives no nominative), or words between
them (του κ. [PERSON_1]), make restore use the first form seen.

Limit: a name seen after punctuation with no greeting (often a vocative: «Ευχαριστώ, Νίκο
Παπαδόπουλε») counts as a nominative and wins over a seen genitive: with «Το σπίτι του Νίκου
Παπαδοπούλου.» also seen, «Ο [PERSON_1]» restores as «Ο Νίκο Παπαδόπουλε». After a greeting
(«Καλημέρα, Νίκο») the case is read as a vocative.

Limit: how restore reads a name by how it was written (the lists are Faker's el_GR names):
- Lowercase with an accent: inflected from it («Ο Νίκος», «του [PERSON_1]» gives «του Νίκου»).
- Capitals, with or without an accent: inflected as lowercase and written back in capitals, with
  an accent only where one was seen («Ο ΝΊΚΟΣ» gives «ΝΊΚΟΥ», «Ο ΝΙΚΟΣ» gives «ΝΙΚΟΥ»).
- No accent and in the lists: the accent is taken from the list but not written (the vocative of
  «Νικος» is «Νικο»). A name listed with two accents that give different text (Ιωνα: «Ίωνα» or
  «Ιωνά») stays as seen.
- No accent and not in the lists: tried with the accent on each of the last three syllables, used
  only if all agree. Where they don't, a first name's -ος vocative is -ο («Ζβορκος» gives «Ζβορκο»,
  not «Ζβορκε»; without a names dir «Παμφιλος» gives «Παμφιλο», where «Πάμφιλε» is right).
  Without a names dir a surname's -ος vocative is -ε wherever the accent is (-άκος aside), so all
  readings agree («κύριε Ζβορκε», «Νίκο Ζβορκε»); with a names dir these come out the same.
  Otherwise the name is inflected as written, with no accent («Ζβορκια» and «της» give
  «Ζβορκιας»).
- Capitals mixed with lowercase, other than a leading capital (ΝΙΚΟς): the name stays as
  seen.

Limit: a man and a woman can still merge when the tokens are identical (Κώστα), and surnames
spelled differently (Παπαδόπουλος, Παπαδόπουλλος) are separate people. The rare genitive
Γιάννου is not matched with Γιάννης.
"""

import re
import unicodedata
from collections.abc import Callable, Iterator
from itertools import product

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

ARTICLE_GENDERS = {
    "ο": "M",
    "ένας": "M",
    "τον": "M",
    "στον": "M",
    "του": "M",
    "ενός": "M",
    "η": "F",
    "την": "F",
    "τη": "F",
    "στην": "F",
    "στη": "F",
    "της": "F",
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
    match = WORD_BEFORE.search(text[:position])
    return " ".join((match.group(1) or match.group(2)).lower().split()) if match else ""


def read_context(word: str) -> tuple[str, str | None, bool] | None:
    """(case, gender, is a title) that the word before a name asks for."""
    if word in ARTICLE_CASES:
        return ARTICLE_CASES[word], ARTICLE_GENDERS.get(word), False
    if word in TITLES:
        return *TITLES[word], True
    if word in GREETINGS:
        return "vocative", GREETINGS[word], False
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
    contexts: list[tuple[str, str | None, bool] | None],
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
        if form["article"] == "" or (context and context[0] == "nominative"):
            yield form["text"], accent_readings(form["text"], guess_accents)
    if gender:
        for form, context in zip(forms, contexts, strict=True):
            if context and context[0] != "nominative":
                lone_surname = context[2] and len(form["text"].split()) == 1
                found = [
                    read_back(reading, gender, context[0], lone_surname)
                    for reading in accent_readings(form["text"], guess_accents)
                ]
                if found and all(found):
                    yield form["text"], found


def derive_form(
    forms: list[dict[str, str]], case: str, reply_gender: str | None, names: Names | None
) -> str | None:
    contexts = [read_context(form["article"]) for form in forms]
    gender = next((context[1] for context in contexts if context and context[1]), reply_gender)
    # A single token is a surname only if the input had it after a title; a title in the reply
    # sets the case and gender, not the kind of name.
    after_title = any(context and context[2] for context in contexts)
    # Three tries, each only if the one before found nothing. First every accent the name could
    # have, and the text is used only if all give the same. Then, for a vocative, the accent a
    # foreign name in -ος usually has, which gives -ο. Last the name as written, with capitals
    # lowercased, but not a listed name whose accents disagree: that stays as seen.
    for attempt in ("guess", "foreign", "as_seen"):
        if attempt == "foreign" and case != "vocative":
            continue
        # The foreign attempt is the owner's rule for foreign names, so the names dir does not
        # change it.
        inflect = names.inflect_name if names and attempt != "foreign" else inflect_name
        longest_tried = 0
        candidates = nominative_candidates(forms, contexts, gender, attempt != "as_seen", names)
        for seen, readings in candidates:
            # A form with fewer tokens, such as a lone surname, would drop the first name.
            if len(seen.split()) < longest_tried:
                continue
            longest_tried = len(seen.split())
            if attempt == "as_seen" and any(
                needs_accent(token) and is_listed(token) and len(accent_candidates(token)) > 1
                for token in seen.split()
            ):
                continue
            if attempt == "foreign":
                fitting = [name for name in readings if fits_foreign_rule(seen, name)]
                # Nothing filtered out means the first try already tried these readings.
                if len(fitting) == len(readings):
                    continue
                readings = fitting
            derived = set()
            for nominative_name in readings:
                name_gender = gender or gender_of(nominative_name)
                lone_surname = after_title and len(nominative_name.split()) == 1
                inflected = name_gender and inflect(
                    nominative_name, name_gender, case, lone_surname
                )
                if not inflected:
                    derived.clear()
                    break
                tokens = inflected.split()
                for i, seen_token in enumerate(seen.split()):
                    if needs_accent(seen_token):
                        tokens[i] = written_style(seen_token)(
                            drop_tonos_and_unseen_diaeresis(tokens[i], seen_token)
                        )
                    elif seen_token.isupper():
                        tokens[i] = tokens[i].upper()
                derived.add(" ".join(tokens))
            if len(derived) == 1:
                return derived.pop()
    return None


def pick_form(forms: list[dict[str, str]], word: str, names: Names | None) -> str:
    context = read_context(word)
    if context is None:
        return forms[0]["text"]
    case, gender, _ = context
    for form in forms:
        seen = read_context(form["article"])
        if seen and seen[0] == case:
            return form["text"]
    return derive_form(forms, case, gender, names) or forms[0]["text"]


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
