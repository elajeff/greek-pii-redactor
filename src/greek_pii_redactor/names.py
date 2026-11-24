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
from functools import cache
from os.path import commonprefix
from pathlib import Path

from greek_pii_redactor.inflect import (
    ACCENT,
    CASES,
    FEMALE_FIRST_ENDINGS,
    MALE_ENDINGS,
    TONOS,
    accented_syllables,
    inflect_name,
    inflection_roles,
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
# Below this probability the rules give a form instead of the classifier: the threshold in
# eval/names_classifier.py with the most held-out names (one per case) whose form is in their
# Wiktionary set, where no role, gender and case has fewer than the rules alone. On a tie the
# highest: the classifier's less sure answers add nothing there, and they can be near-ties on
# endings it has few names for (full build, Κίμων accusative: Κίμωνα 0.26, Κίμων 0.24).
FORWARD_THRESHOLD = 0.7
# Below this probability the rules read a form back instead of the classifier: the lowest threshold
# in eval/names_classifier.py where the classifier's wrong read-backs are no more than the rules' in
# every role, gender and case.
READ_BACK_THRESHOLD = 0.9
# The Faker el_GR list of each role and gender the lexicon holds; a female surname never changes.
FAKER_LISTS = {
    ("first", "M"): "first_names_male",
    ("first", "F"): "first_names_female",
    ("surname", "M"): "last_names_male",
}
# ΐ and ΰ without the accent but with the diaeresis.
DIAERESIS_LETTERS = {"ΐ": "ϊ", "ΰ": "ϋ"}
# Forms checked by hand where Wiktionary's table differs from how the names are used. The rule
# chosen by hand for whole groups is in corrected.
MANUAL_FORMS = {
    ("Ντίνος", "first", "M"): {"vocative": ["Ντίνο"]},
    ("Ανδρόνικος", "first", "M"): {"vocative": ["Ανδρόνικε"]},
    ("Φόρος", "first", "M"): {"vocative": ["Φόρε", "Φόρο"]},
}
# Modern Greek inflects the -οντα names from their accusative, like γέρων, γέροντας, whose
# Wiktionary table (and άρχοντας') gives -οντα for the genitive, accusative and vocative.
# Wiktionary has none of these names. Φιλήμων is set by hand too: its Ancient Greek Wiktionary
# entry gives the genitive -ονος with no table, jw.org writes the accusative Φιλήμονα, and
# Παντελεήμων's table gives the vocative -ον.
MANUAL_FIRST_NAMES = [
    {"word": word, "role": "first", "gender": "M"} | {case: [stem + "οντα"] for case in CASES}
    for stem in ("Κρέ", "Λέ", "Δράκ", "Τιμολέ", "Ναπολέ")
    for word in (stem + "ων", stem + "οντας")
] + [
    {
        "word": "Φιλήμων",
        "role": "first",
        "gender": "M",
        "genitive": ["Φιλήμονος"],
        "accusative": ["Φιλήμονα"],
        "vocative": ["Φιλήμον"],
    }
]
# Vocatives checked by hand for Faker male surnames in paroxytone -ος that Wiktionary has no table
# for. Φαλιέρος, Χέλιος and Κατσίφος have the rules' -ε: no use of either vocative was found for
# any of them. The genitive and accusative are stem+ου and stem+ο, as Wiktionary gives them for
# nearly all paroxytone male surnames in -ος.
MANUAL_SURNAME_VOCATIVES = {
    "Ότσος": ["Ότσο"], "Βούλκος": ["Βούλκο"], "Βώσσος": ["Βώσσο"], "Γεροδήμος": ["Γεροδήμο"],
    "Γιαννέλος": ["Γιαννέλο"], "Γκανάτσιος": ["Γκανάτσιο"], "Γκούσκος": ["Γκούσκο"],
    "Γούσιος": ["Γούσιο"], "Δέλιος": ["Δέλιο"], "Ζιώγος": ["Ζιώγο"], "Ζώνιος": ["Ζώνιο"],
    "Ζώτος": ["Ζώτο"], "Θεολόγος": ["Θεολόγε"], "Θηβαίος": ["Θηβαίε"], "Κακοσαίος": ["Κακοσαίε"],
    "Καλιάμπος": ["Καλιάμπε"], "Καπετάνιος": ["Καπετάνιε", "Καπετάνιο"], "Καραδήμος": ["Καραδήμο"],
    "Καραμάνος": ["Καραμάνε", "Καραμάνο"], "Καρατζίκος": ["Καρατζίκο"], "Καρούντζος": ["Καρούντζο"],
    "Καρούσος": ["Καρούσο"], "Κατσίφος": ["Κατσίφε"], "Κουλίζος": ["Κουλίζο"],
    "Κυπραίος": ["Κυπραίε", "Κυπραίο"], "Λάτσκος": ["Λάτσκο"], "Λώλος": ["Λώλο"],
    "Μάνδρος": ["Μάνδρε"], "Μάνος": ["Μάνο"], "Μίσχος": ["Μίσχο"],
    "Μαντζώρος": ["Μαντζώρε", "Μαντζώρο"], "Μαυρούτσος": ["Μαυρούτσο"], "Μπέκος": ["Μπέκο"],
    "Μπούσιος": ["Μπούσιο"], "Νάστος": ["Νάστο"], "Νέλος": ["Νέλο"], "Ντόκος": ["Ντόκο"],
    "Πάνος": ["Πάνο"], "Πίσσιος": ["Πίσσιο"], "Παντζέκος": ["Παντζέκο"], "Πιάγκος": ["Πιάγκο"],
    "Πιπερίγκος": ["Πιπερίγκο"], "Πουλιέζος": ["Πουλιέζο"], "Ρέντζος": ["Ρέντζο"],
    "Ρέππος": ["Ρέππο"], "Ρίγκος": ["Ρίγκο"], "Σαλίχος": ["Σαλίχο"], "Σεφέκος": ["Σεφέκο"],
    "Στόγιος": ["Στόγιο"], "Τζιόρτζιος": ["Τζιόρτζιο"], "Τσάρκος": ["Τσάρκο"],
    "Τσέλλος": ["Τσέλλο"], "Τσεντούρος": ["Τσεντούρο"], "Τσομώκος": ["Τσομώκο"],
    "Τσούμος": ["Τσούμο"], "Φαλιέρος": ["Φαλιέρε"], "Χέλιος": ["Χέλιε"],
    "Χατζηκύρκος": ["Χατζηκύρκο"],
}
MANUAL_NAMES = MANUAL_FIRST_NAMES + [
    {
        "word": word,
        "role": "surname",
        "gender": "M",
        "genitive": [word[:-2] + "ου"],
        "accusative": [word[:-2] + "ο"],
        "vocative": vocatives,
    }
    for word, vocatives in MANUAL_SURNAME_VOCATIVES.items()
]


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
    # The edit cuts and appends characters, so commonprefix comparing characters is what's wanted.
    plain_prefix = len(commonprefix([strip_accents(source), strip_accents(target)]))  # noqa: RUF071
    for shift in SHIFTS:
        edit = (len(source) - plain_prefix, target[plain_prefix:], shift)
        if apply_edit(source, edit, first_name) == target:
            return edit
    prefix = len(commonprefix([source, target]))  # noqa: RUF071
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


def corrected(name: dict) -> dict:
    """The name with MANUAL_FORMS applied, and the vocative -ε for a male surname in -ος accented
    on the third syllable from the end (Ανάργυρε; Wiktionary gives -ο for 32)."""
    word, role, gender = name["word"], name["role"], name["gender"]
    fixed = {**name, **MANUAL_FORMS.get((word, role, gender), {})}
    if role == "surname" and gender == "M" and word.endswith("ος"):
        spans = syllables(word, False)
        if accented_syllables(word, spans) == [len(spans) - 3]:
            fixed["vocative"] = [word[:-2] + "ε"]
    return fixed


def read_names(path: Path) -> Iterator[dict]:
    """The lines of a names file, as written by eval/download_names.py, one at a time, with the
    hand-checked forms applied, then the MANUAL_NAMES the file has no line for, so training, the
    lexicon and the eval all use them."""
    count = 0
    keys = set()
    with path.open(encoding="utf-8") as file:
        for count, line in enumerate(file, 1):
            name = json.loads(line)
            for case in CASES:
                forms = name.get(case)
                if not isinstance(forms, list) or not all(isinstance(f, str) for f in forms):
                    raise ValueError(f"{path} line {count}: {case} is not a list of strings")
            keys.add((name["word"], name["role"], name["gender"]))
            yield corrected(name)
    if not count:
        raise ValueError(f"{path} has no names")
    for name in MANUAL_NAMES:
        if (name["word"], name["role"], name["gender"]) not in keys:
            yield name


def is_female_surname(role: str, gender: str) -> bool:
    """Female surnames never change, so they are not stored or predicted."""
    return role == "surname" and gender == "F"


@cache
def faker_names(role: str, gender: str) -> dict[str, set[str]]:
    """The Faker el_GR names of a role and gender, by their spelling without accents."""
    from faker.providers.person import el_GR  # slow to import, as in inflect.accents_by_plain_form

    by_spelling: dict[str, set[str]] = {}
    for name in getattr(el_GR.Provider, FAKER_LISTS[role, gender]):
        by_spelling.setdefault(strip_accents(name), set()).add(name)
    return by_spelling


class Names:
    def __init__(self, names: Iterable[dict], models: dict) -> None:
        # (word, role, gender) -> the forms for each case, in the order of CASES.
        self.forms: dict[tuple[str, str, str], tuple[tuple[str, ...], ...]] = {}
        self.nominatives: dict[tuple[str, str, str, str], list[str]] = {}
        # (word without accents, role, gender) -> the words spelt so.
        self.words_by_spelling: dict[tuple[str, str, str], set[str]] = {}
        for name in names:
            if is_female_surname(name["role"], name["gender"]):
                continue
            key = (name["word"], name["role"], name["gender"])
            table = tuple(tuple(name[case]) for case in CASES)
            self.forms[key] = table
            spelling = (strip_accents(key[0]), *key[1:])
            self.words_by_spelling.setdefault(spelling, set()).add(key[0])
            for case, forms in zip(CASES, table, strict=True):
                for form in forms:
                    index = (form, *key[1:], case)
                    self.nominatives.setdefault(index, []).append(key[0])
        self.models = models

    def inflect_name(self, name: str, gender: str, case: str, surname_only: bool) -> str:
        """As inflect.inflect_name, with each token's form from the lexicon (the rules' form if it
        is one of the listed ones, else the first) or, for a token not in it, from predict when it
        is at least FORWARD_THRESHOLD sure, else from the rules."""
        if case == "nominative":
            return name
        tokens = name.split()
        capitalised = [with_capital(token) for token in tokens]
        rules = inflect_name(" ".join(capitalised), gender, case, surname_only).split()
        roles = inflection_roles(capitalised, gender, surname_only)
        inflected = []
        for token, capital, role, rules_form in zip(tokens, capitalised, roles, rules, strict=True):
            listed = self.lookup(capital, role, gender, case)
            if listed:
                form = rules_form if rules_form in listed else listed[0]
            else:
                form, probability = self.predict(capital, role, gender, case)
                if probability < FORWARD_THRESHOLD:
                    form = rules_form
            inflected.append(written_like(form, token))
        return " ".join(inflected)

    def nominative(self, form: str, gender: str, case: str, surname_only: bool) -> str | None:
        """As inflect.nominative, with each token read back in this order. A male surname in -ου
        that is a Faker el_GR male surname is itself in the genitive (Σακελλαρίου). A form the
        lexicon lists gives its name; one it lists for several names gives the rules' nominative
        if it is one of them, else None. Any other form is read back by unlisted_nominative."""
        nominatives = []
        for index, token in enumerate(form.split()):
            role = "surname" if index or surname_only else "first"
            capital = with_capital(token)
            listed = self.reverse(capital, role, gender, case)
            if (
                (role, gender, case) == ("surname", "M", "genitive")
                and capital.endswith("ου")
                and capital in faker_names(role, gender).get(strip_accents(capital), ())
            ):
                found = capital
            elif listed and len(listed) == 1:
                found = listed[0]
            elif listed:
                rules = nominative(capital, gender, case, surname_only=role == "surname")
                found = rules if rules in listed else None
            else:
                found = self.unlisted_nominative(capital, role, gender, case)
            if found is None:
                return None
            nominatives.append(written_like(found, token))
        return " ".join(nominatives)

    def unlisted_nominative(self, form: str, role: str, gender: str, case: str) -> str | None:
        """The rules' nominative if it is a known name, one of known_names (Γιαννακέα:
        Γιαννακέας), else predict_nominative's when it is at least READ_BACK_THRESHOLD sure, else
        the rules'. A predicted name that is not known gives way to the one known name spelt like
        it without accents whose form in this case by the rules is the given form (Ανδρονίκου:
        Ανδρόνικος, not Ανδρονίκος)."""
        surname_only = role == "surname"
        rules = nominative(form, gender, case, surname_only=surname_only)
        if rules is not None and rules in self.known_names(rules, role, gender):
            return rules
        found, probability = self.predict_nominative(form, role, gender, case)
        if probability < READ_BACK_THRESHOLD:
            return rules
        known = self.known_names(found, role, gender)
        if found in known:
            return found
        fitting = [
            name for name in known if inflect_name(name, gender, case, surname_only) == form
        ]
        return fitting[0] if len(fitting) == 1 else found

    def known_names(self, name: str, role: str, gender: str) -> set[str]:
        """The names of the lexicon and of the Faker el_GR list of a role and gender that are
        spelt like the name without accents."""
        spelling = strip_accents(name)
        listed = self.words_by_spelling.get((spelling, role, gender), set())
        return listed | faker_names(role, gender).get(spelling, set())

    def lookup(self, token: str, role: str, gender: str, case: str) -> tuple[str, ...] | None:
        if is_female_surname(role, gender):
            return (token,)
        table = self.forms.get((token, role, gender))
        if table is None:
            return None
        return table[CASES.index(case)] or None

    def reverse(self, form: str, role: str, gender: str, case: str) -> list[str] | None:
        if is_female_surname(role, gender):
            return [form]
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


def with_capital(token: str) -> str:
    """The lexicon and the models hold names with a leading capital."""
    return token[0].upper() + token[1:] if token.islower() else token


def written_like(form: str, token: str) -> str:
    """The form with its first letter lowercased again when the token was all lowercase."""
    return form[0].lower() + form[1:] if token.islower() else form


def load_names(names_dir: Path) -> Names:
    # Imported here so redact.py can import this module without the names extra.
    import joblib

    return Names(read_names(names_dir / LEXICON_FILE), joblib.load(names_dir / MODELS_FILE))
