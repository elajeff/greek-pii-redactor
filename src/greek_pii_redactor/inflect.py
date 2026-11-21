"""Greek name inflection: a nominative name into genitive, accusative or vocative, and back.

Case names match redact.ARTICLE_CASES: "nominative", "genitive", "accusative", "vocative".
In a full name the first token is the first name and later tokens are surnames. A single token is
a first name, except after a title (surname_only=True).

Limit: the rules follow the Faker el_GR name lists, not all Greek. Folk first names in -ιος
(Σπήλιος, Στέλλιος) are treated as learned (Σπηλίου instead of Σπήλιου). A lone female name in
-η, -ή or -ά is left unchanged, because the same endings are far more often female surnames.
"""

import re
import unicodedata
from functools import cache
from itertools import product

CASES = ("genitive", "accusative", "vocative")
# Ending -> (genitive, accusative, vocative) endings. A name matching no ending stays as it is.
# -ος has no vocative here: male_vocative chooses it.
MALE_ENDINGS = {
    "ος": ("ου", "ο"),
    "ός": ("ού", "ό", "έ"),
    "ης": ("η", "η", "η"),
    "ής": ("ή", "ή", "ή"),
    "ας": ("α", "α", "α"),
    "άς": ("ά", "ά", "ά"),
    "ων": ("ωνος", "ωνα", "ων"),
}
FEMALE_FIRST_ENDINGS = {
    "α": ("ας", "α", "α"),
    "η": ("ης", "η", "η"),
    "ή": ("ής", "ή", "ή"),
    "ά": ("άς", "ά", "ά"),
    "ω": ("ως", "ω", "ω"),
    "ώ": ("ώς", "ώ", "ώ"),
}
# Only these endings mark a lone token as a female first name; -η, -ή, -ά and -ου are also
# common female surnames.
LONE_FEMALE_FIRST_ENDINGS = ("α", "ω", "ώ")
FEMALE_GENITIVE_ENDINGS = ("ας", "άς", "ης", "ής", "ως", "ώς")

ACCENT = dict(zip("αεηιουωΑΕΗΙΟΥΩϊϋ", "άέήίόύώΆΈΉΊΌΎΏΐΰ", strict=True))
ACCENTED = set(ACCENT.values())
VOWELS = set(ACCENT) | ACCENTED
DIPHTHONGS = {"αι", "ει", "οι", "υι", "ου", "αυ", "ευ", "ηυ"}
DIAERESIS = "ϊϋΐΰ"  # breaks a diphthong: Βαϊτσόπουλος has four syllables before -λος
# A first name in -ιος is learned (Γεώργιος, Γεωργίου: the ι is a syllable); a surname is folk
# (Μπούσιος, Μπούσιου: the ι joins the next vowel).
LEARNED_FINAL_IOS = ("ιος", "ιου", "ιο", "ιε")
# The vocative of a first name of this many syllables or more takes -ε even when the accent is
# on the second to last (Κωνσταντίνε).
LONG_FIRST_NAME = 4
# Names kept out of the accent table and the synthetic pools: -ων names whose genitive is -ονος or
# -οντος (not -ωνος), Βάιος (genitive Βαΐου needs a diaeresis), and Faker names with a missing or
# doubled accent (Ζηναϊς for Ζηναΐς).
EXCLUDED_NAMES = {
    "Αγαμέμνων", "Δράκων", "Ιάσων", "Λέων", "Ναπολέων", "Παντελεήμων", "Σώζων", "Τιμολέων",
    "Φιλήμων", "Βάιος", "Κουτσογιάννόπουλος", "Τηλεμαχος", "Βουτσελας", "Τζαλλας", "Αίας",
    "Τσιωλξ", "Ζηναϊς",
}  # fmt: skip
TONOS = "\u0301"
DIAERESIS_MARK = "\u0308"
# Greek words are accented on one of the last three syllables.
ACCENT_REACH = 3
LETTER_WITH_MARKS = re.compile(r".[\u0300-\u036f]*")  # a letter and the combining marks after it


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def syllables(word: str, first_name: bool) -> list[tuple[int, int]]:
    """Spans of the vowel groups. An unaccented ι between a consonant and a vowel joins that
    vowel's syllable; a first name's final ι of -ιος stays a syllable of its own."""
    learned_start = -1
    if first_name:
        learned_start = next(
            (len(word) - len(ending) for ending in LEARNED_FINAL_IOS if word.endswith(ending)), -1
        )
    spans = []
    i = 0
    while i < len(word):
        if word[i] not in VOWELS:
            i += 1
            continue
        joins_next = (
            word[i] == "ι"
            and i != learned_start
            and 0 < i
            and word[i - 1].isalpha()
            and word[i - 1] not in VOWELS
            and word[i + 1 : i + 2] in VOWELS
        )
        if joins_next:
            i += 1
            continue
        pair = word[i : i + 2]
        if len(pair) == 2 and word[i].lower() + strip_accents(pair[1]).lower() in DIPHTHONGS:
            joined = pair[1] not in DIAERESIS
        else:
            joined = False
        spans.append((i, i + 1 + joined))
        i += 1 + joined
    return spans


def accented_syllables(word: str, spans: list[tuple[int, int]]) -> list[int]:
    return [i for i, (start, end) in enumerate(spans) if ACCENTED & set(word[start:end])]


def shift_accent(word: str, first_name: bool) -> str:
    """The long ending -ου cannot follow an accent on the third syllable from the end."""
    spans = syllables(word, first_name)
    accented = accented_syllables(word, spans)
    if not word.endswith("ου") or accented != [len(spans) - 3]:
        return word
    (old_start, old_end), (_, new_end) = spans[accented[0]], spans[accented[0] + 1]
    old = "".join(strip_accents(c) if c in ACCENTED else c for c in word[old_start:old_end])
    # The mark goes on the last vowel of a diphthong: ει -> εί.
    new_last = ACCENT[word[new_end - 1]]
    return word[:old_start] + old + word[old_end : new_end - 1] + new_last + word[new_end:]


def accent_third_from_end(word: str, first_name: bool) -> str | None:
    spans = syllables(word, first_name)
    if len(spans) < 3:
        return None
    end = spans[-3][1]
    return word[: end - 1] + ACCENT[word[end - 1]] + word[end:]


def male_vocative(token: str, first_name: bool) -> str:
    """The ending of the vocative of a name in -ος, which depends on where the accent is."""
    if token.endswith("αίος"):
        return "ε"
    spans = syllables(token, first_name)
    accented = accented_syllables(token, spans)
    from_end = len(spans) - accented[0] if accented else 0
    if from_end == 2 and not (first_name and len(spans) >= LONG_FIRST_NAME):
        return "ο"
    return "ε"


def inflect_male(token: str, case: str, first_name: bool) -> str:
    for ending, forms in MALE_ENDINGS.items():
        if token.endswith(ending):
            if ending == "ος" and case == "vocative":
                return token[:-2] + male_vocative(token, first_name)
            inflected = token[: -len(ending)] + forms[CASES.index(case)]
            return shift_accent(inflected, first_name) if case == "genitive" else inflected
    return token


def inflect_female_first(token: str, case: str) -> str:
    for ending, forms in FEMALE_FIRST_ENDINGS.items():
        if token.endswith(ending):
            return token[: -len(ending)] + forms[CASES.index(case)]
    return token


def inflect_name(name: str, gender: str, case: str, surname_only: bool) -> str:
    """name is nominative, gender "M" or "F". surname_only: a lone surname after a title."""
    if case == "nominative":
        return name
    tokens = name.split()
    inflected = []
    for index, token in enumerate(tokens):
        first_name = index == 0 and not surname_only
        if gender == "M":
            inflected.append(inflect_male(token, case, first_name))
        elif first_name and (len(tokens) > 1 or token.endswith(LONE_FEMALE_FIRST_ENDINGS)):
            inflected.append(inflect_female_first(token, case))
        else:
            inflected.append(token)
    return " ".join(inflected)


def male_nominative(token: str, case: str, first_name: bool) -> str | None:
    if case == "nominative":
        return token
    if case == "genitive":
        if token.endswith("πούλου"):
            return accent_third_from_end(token[: -len("πούλου")] + "πουλος", first_name)
        if token.endswith("ου"):
            if not first_name:
                return None
            if token.endswith("αίου"):
                return token[:-2] + "ος"  # Ματθαίος is already accented on the αί
            if token.endswith("ίου"):
                return accent_third_from_end(token[:-3] + "ιος", first_name)
            return token[:-2] + "ος" if len(syllables(token, first_name)) == 2 else None
        if token.endswith("ού"):
            return token[:-2] + "ός"
        if token.endswith("ωνος"):
            return token[:-2]
    else:
        if token.endswith("ου") and not first_name:
            return token
        if case == "accusative" and token.endswith("ωνα"):
            return token[:-1]
    if token.endswith(("η", "ή", "α", "ά")):
        return token + "ς"
    if case == "accusative" and token.endswith(("ο", "ό")):
        return token + "ς"
    if case == "vocative":
        if token.endswith(("ο", "ε")):
            return token[:-1] + "ος"
        if token.endswith("έ"):
            return token[:-1] + "ός"
    return None


def female_nominative(token: str, case: str, first_name: bool) -> str | None:
    if case != "genitive" or not first_name:
        return token
    return token[:-1] if token.endswith(FEMALE_GENITIVE_ENDINGS) else None


def nominative(form: str, gender: str, case: str, surname_only: bool) -> str | None:
    """The nominative of a form seen in the given case, or None when it can't be told."""
    nominatives = []
    for index, token in enumerate(form.split()):
        first_name = index == 0 and not surname_only
        if gender == "M":
            found = male_nominative(token, case, first_name)
        else:
            found = female_nominative(token, case, first_name)
        if found is None:
            return None
        nominatives.append(found)
    return " ".join(nominatives)


@cache
def accents_by_plain_form() -> dict[str, set[str]]:
    """Unaccented lowercase form -> the accented forms it fits, over every case of every name in
    the Faker el_GR lists."""
    from faker.providers.person import el_GR  # slow to import, and only unaccented names need it

    names = [
        (el_GR.Provider.first_names_male, "M", True),
        (el_GR.Provider.first_names_female, "F", True),
        (el_GR.Provider.last_names_male, "M", False),
        (el_GR.Provider.last_names_female, "F", False),
    ]
    accented: dict[str, set[str]] = {}
    for listed, gender, first_name in names:
        for name in set(listed) - EXCLUDED_NAMES:
            forms = [name]
            if gender == "M":
                forms += [inflect_male(name, case, first_name) for case in CASES]
            elif first_name:
                forms += [inflect_female_first(name, case) for case in CASES]
            for form in forms:
                accented.setdefault(strip_accents(form).lower(), set()).add(form)
    return accented


def needs_accent(token: str) -> bool:
    """A token with no tonos. A diaeresis alone is not an accent, and Greek does not accent a
    one-syllable word."""
    no_tonos = TONOS not in unicodedata.normalize("NFD", token)
    return no_tonos and len(syllables(token.lower(), False)) > 1


def is_listed(token: str) -> bool:
    """Whether the token, or the last part of a hyphenated one, is a name in the Faker lists."""
    return strip_accents(token.split("-")[-1]).lower() in accents_by_plain_form()


def accent_candidates(token: str) -> list[str]:
    """The accented forms a token can stand for: the listed ones, else the token with the accent
    on each of its last three syllables. Each part of a hyphenated token is looked up on its own.
    Empty for a token that mixes capitals and lowercase, other than a leading capital."""
    if not needs_accent(token):
        return [token]
    if token not in (token.upper(), token.lower(), token.title()):
        return []
    if "-" in token:
        return ["-".join(parts) for parts in product(*map(accent_candidates, token.split("-")))]
    if listed := accents_by_plain_form().get(strip_accents(token).lower()):
        return sorted(listed)
    # Lowercase, because the ending rules match lowercase; the caller puts the style back.
    word = token.lower()
    ends = [end for _, end in syllables(word, False)[-ACCENT_REACH:]]
    # The mark goes on the last vowel of a diphthong, as in shift_accent.
    return [word[: end - 1] + ACCENT[word[end - 1]] + word[end:] for end in ends]


def accent_on_second_to_last(token: str) -> bool:
    """Whether the accent is on the second to last syllable of the token, or of the last part of
    a hyphenated one."""
    last_part = token.split("-")[-1]
    spans = syllables(last_part, False)
    return accented_syllables(last_part, spans) == [len(spans) - 2]


def drop_tonos_and_unseen_diaeresis(written: str, seen: str) -> str:
    """An output token without its tonos. An ι or υ has a diaeresis exactly where the same letter
    of the token as seen in the input had one, so Χάιδως written for ΧΑΪΔΩ becomes ΧΑΪΔΩΣ."""
    seen_letters = LETTER_WITH_MARKS.findall(unicodedata.normalize("NFD", seen))
    kept = []
    for i, letter in enumerate(LETTER_WITH_MARKS.findall(unicodedata.normalize("NFD", written))):
        had_diaeresis = i < len(seen_letters) and DIAERESIS_MARK in seen_letters[i]
        kept.append(
            letter[0] + DIAERESIS_MARK if had_diaeresis and letter[0] in "ιυΙΥ" else letter[0]
        )
    return unicodedata.normalize("NFC", "".join(kept))


def gender_of(nominative_name: str) -> str | None:
    """From the first token alone: any name in Greek script ending in -ς is "M". -ου (male and
    female surnames) and Latin-script names give None."""
    first = nominative_name.split()[0]
    if first.endswith(("ς", "ων")):
        return "M"
    if first.endswith(("α", "ά", "η", "ή", "ω", "ώ")):
        return "F"
    return None
