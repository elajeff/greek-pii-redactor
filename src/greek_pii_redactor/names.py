"""Name forms from Wiktionary's case tables, and a classifier for names that are not in them.

An edit turns one token into another: cut letters off the end, add an ending, then move the accent
one syllable right (+1), left (-1) or not at all (0). It is taken from the common prefix of the
two tokens without accents, trying shift 0, +1, -1; if none gives the target, from the common
prefix as written, shift 0 (Ξανθός, Ξανθού). One classifier per direction and case picks the edit
from the token's ending, its accent, role and gender. A predicted edit must keep the token's
number of tonos marks; if it doesn't, the next most probable one is tried.

The files in a names directory are written by train/train_inflection.py.
"""

import json
import unicodedata
from collections.abc import Iterable, Iterator
from os.path import commonprefix
from pathlib import Path

import joblib

from .inflect import (
    ACCENT,
    CASES,
    FEMALE_FIRST_ENDINGS,
    MALE_ENDINGS,
    TONOS,
    accented_syllables,
    inflect_name,
    nominative,
    strip_accents,
    syllables,
)

Edit = tuple[int, str, int]  # letters cut off the end, ending added, accent shift

LEXICON_FILE = "lexicon.jsonl"
MODELS_FILE = "models.joblib"
SHIFTS = (0, 1, -1)
# The longest endings used as features, as written and without accents.
WRITTEN_ENDINGS = 5
PLAIN_ENDINGS = 4
# An edit that fits the training names can still give a misspelt word on a new one; past this
# many of the most probable edits the rules answer instead.
CANDIDATE_EDITS = 5
# Wiktionary gives two forms of a case for many names (Παπαδοπούλου, Παπαδόπουλου), so a nominative
# is read back when the form is either of its two most probable.
ROUND_TRIP_FORMS = 2
# ΐ and ΰ without the accent but with the diaeresis.
DIAERESIS_LETTERS = {"ΐ": "ϊ", "ΰ": "ϋ"}
# The owner's calls on single names where Wiktionary's table differs from how they are used. The
# owner's rule for whole groups is in owner_corrected.
OWNER_FORMS = {
    ("Ντίνος", "first", "M"): {"vocative": ["Ντίνο"]},
    ("Ανδρόνικος", "first", "M"): {"vocative": ["Ανδρόνικε"]},
    ("Φόρος", "first", "M"): {"vocative": ["Φόρε", "Φόρο"]},
}


def move_accent(word: str, first_name: bool, shift: int) -> str | None:
    """The word with its one accent moved shift syllables, or None if it has no single accent or
    there is no syllable there."""
    spans = syllables(word, first_name)
    accented = accented_syllables(word, spans)
    if len(accented) != 1 or not 0 <= accented[0] + shift < len(spans):
        return None
    (old_start, old_end), (_, new_end) = spans[accented[0]], spans[accented[0] + shift]
    letters = list(word)
    for i in range(old_start, old_end):
        # With the accent on the vowel right before, nothing joins them (Βαΐου, Βάιος); elsewhere
        # the diaeresis stays (Παΐσιος, Παϊσίου).
        if letters[i] in DIAERESIS_LETTERS and i - 1 != new_end - 1:
            letters[i] = DIAERESIS_LETTERS[letters[i]]
        else:
            letters[i] = strip_accents(letters[i])
    # The mark goes on the last vowel of a diphthong, as in inflect.shift_accent.
    letters[new_end - 1] = ACCENT[letters[new_end - 1]]
    return "".join(letters)


def apply_edit(token: str, edit: Edit, first_name: bool) -> str | None:
    cut, add, shift = edit
    edited = token[: len(token) - cut] + add
    return move_accent(edited, first_name, shift) if shift else edited


def derive_edit(source: str, target: str, first_name: bool) -> Edit:
    plain_prefix = len(commonprefix([strip_accents(source), strip_accents(target)]))
    for shift in SHIFTS:
        edit = (len(source) - plain_prefix, target[plain_prefix:], shift)
        if apply_edit(source, edit, first_name) == target:
            return edit
    prefix = len(commonprefix([source, target]))
    return (len(source) - prefix, target[prefix:], 0)


def features(token: str, role: str, gender: str) -> dict[str, str]:
    spans = syllables(token, role == "first")
    accented = accented_syllables(token, spans)
    plain = strip_accents(token)
    found = {f"written{n}": token[-n:] for n in range(1, WRITTEN_ENDINGS + 1)}
    found |= {f"plain{n}": plain[-n:] for n in range(1, PLAIN_ENDINGS + 1)}
    found["accent"] = str(len(spans) - accented[0]) if accented else "none"
    found["syllables"] = str(len(spans))
    found["role"] = role
    found["gender"] = gender
    return found


def tonos_count(token: str) -> int:
    return unicodedata.normalize("NFD", token).count(TONOS)


def owner_corrected(name: dict) -> dict:
    """The name with the owner's calls applied: OWNER_FORMS, and the vocative -ε for a male surname
    in -ος accented on the third syllable from the end (Ανάργυρε; Wiktionary gives -ο for 32)."""
    word, role, gender = name["word"], name["role"], name["gender"]
    corrected = {**name, **OWNER_FORMS.get((word, role, gender), {})}
    if role == "surname" and gender == "M" and word.endswith("ος"):
        spans = syllables(word, False)
        if accented_syllables(word, spans) == [len(spans) - 3]:
            corrected["vocative"] = [word[:-2] + "ε"]
    return corrected


def read_names(path: Path) -> Iterator[dict]:
    """The lines of a names file, as written by eval/download_names.py, one at a time, with the
    owner's calls applied, so training, the lexicon and the eval all use them."""
    count = 0
    with path.open(encoding="utf-8") as file:
        for count, line in enumerate(file, 1):
            name = json.loads(line)
            for case in CASES:
                forms = name.get(case)
                if not isinstance(forms, list) or not all(isinstance(f, str) for f in forms):
                    raise ValueError(f"{path} line {count}: {case} is not a list of strings")
            yield owner_corrected(name)
    if not count:
        raise ValueError(f"{path} has no names")


def is_female_surname(role: str, gender: str) -> bool:
    """The owner's rule: a female surname never changes, so it is not stored or predicted."""
    return role == "surname" and gender == "F"


class Names:
    def __init__(self, names: Iterable[dict], models: dict) -> None:
        # (word, role, gender) -> the forms for each case, in the order of CASES.
        self.forms: dict[tuple[str, str, str], tuple[tuple[str, ...], ...]] = {}
        self.nominatives: dict[tuple[str, str, str, str], tuple[str, ...]] = {}
        for name in names:
            if is_female_surname(name["role"], name["gender"]):
                continue
            key = (name["word"], name["role"], name["gender"])
            table = tuple(tuple(name[case]) for case in CASES)
            self.forms[key] = table
            for case, forms in zip(CASES, table, strict=True):
                for form in forms:
                    index = (form, *key[1:], case)
                    self.nominatives[index] = self.nominatives.get(index, ()) + (key[0],)
        self.models = models

    def lookup(self, token: str, role: str, gender: str, case: str) -> tuple[str, ...] | None:
        if is_female_surname(role, gender):
            return (token,)
        table = self.forms.get((token, role, gender))
        if table is None:
            return None
        return table[CASES.index(case)] or None

    def reverse(self, form: str, role: str, gender: str, case: str) -> tuple[str, ...] | None:
        if is_female_surname(role, gender):
            return (form,)
        return self.nominatives.get((form, role, gender, case))

    def predict(self, token: str, role: str, gender: str, case: str) -> tuple[str, float]:
        """The first of ranked_forms."""
        return self.ranked_forms(token, role, gender, case)[0]

    def ranked_forms(
        self, token: str, role: str, gender: str, case: str
    ) -> list[tuple[str, float]]:
        """The forms with as many tonos marks as the token, most probable first. When none of the
        top CANDIDATE_EDITS edits gives one, the rules' form with probability 0. A female surname,
        or a token in none of the rules' endings (Σμιθ), is the token itself with probability 1.
        The token is lowercase with at most a leading capital."""
        endings = tuple(MALE_ENDINGS if gender == "M" else FEMALE_FIRST_ENDINGS)
        if is_female_surname(role, gender) or not token.endswith(endings):
            return [(token, 1.0)]
        rules = inflect_name(token, gender, case, surname_only=role == "surname")
        return self.candidates("forward", token, role, gender, case) or [(rules, 0.0)]

    def predict_nominative(
        self, form: str, role: str, gender: str, case: str
    ) -> tuple[str | None, float]:
        """The most probable nominative whose first ROUND_TRIP_FORMS ranked_forms include the
        form. When none of the top CANDIDATE_EDITS does, the rules' nominative with probability
        0, or None when the rules can't tell. A female surname is the form itself. The form is
        lowercase with at most a leading capital."""
        if is_female_surname(role, gender):
            return form, 1.0
        for candidate, probability in self.candidates("reverse", form, role, gender, case):
            ranked = self.ranked_forms(candidate, role, gender, case)[:ROUND_TRIP_FORMS]
            if form in [inflected for inflected, _ in ranked]:
                return candidate, probability
        return nominative(form, gender, case, surname_only=role == "surname"), 0.0

    def candidates(
        self, direction: str, token: str, role: str, gender: str, case: str
    ) -> list[tuple[str, float]]:
        """Outputs of the most probable edits that cut less than the whole token and keep its
        number of tonos marks, most probable first."""
        model = self.models[direction, case]
        probabilities = model.predict_proba([features(token, role, gender)])[0]
        found = []
        for index in probabilities.argsort()[::-1][:CANDIDATE_EDITS]:
            edit = tuple(json.loads(model.classes_[index]))
            if edit[0] >= len(token):
                continue
            edited = apply_edit(token, edit, role == "first")
            if edited is not None and tonos_count(edited) == tonos_count(token):
                found.append((edited, float(probabilities[index])))
        return found


def load_names(names_dir: Path) -> Names:
    return Names(read_names(names_dir / LEXICON_FILE), joblib.load(names_dir / MODELS_FILE))
