"""Seeded synthetic Greek and English sentences with labelled personal data.

Every ID is generated forwards (random digits, then the check digit), separately from the
validators in rules.py, so the test data does not depend on the code it tests. All people and
numbers are invented.

Train and test split the name pools by sorted-index parity over name stems, shared by first names
and surnames of both genders, and use separate hand-written template lists. A stem sits in one
split only, so no name token is used in both (the male genitive Ρωμαίου is also the female
nominative, which is why stems are compared). Only names the ending rules can inflect are used.

The genitive moves the accent (Παπαδοπούλου), which Wiktionary lists as the more formal form;
the unmoved Παπαδόπουλου is also used in everyday text and is not generated.

Record format, one JSON object per line: {"text": ..., "spans": [[start, end, kind], ...]}
with character offsets, end exclusive.
"""

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path

from faker.providers.person import el_GR, en_US

SPLIT_PARITY = {"train": 0, "test": 1}
GREEK_SHARE = 0.75  # the tool targets Greek text first, so most records are Greek
MOBILE_SHARE = 0.6  # mobiles are the more common number in forms and messages
COMPACT_IBAN_SHARE = 0.5  # IBANs appear both with and without spaces
EMAIL_NUMBER_SHARE = 0.3  # many addresses carry a number to stay unique
EMAIL_NUMBER_DIGITS = 2
AFM_BODY_DIGITS = 8  # the ninth digit is the check digit
AMKA_SERIAL_DIGITS = 4  # after DDMMYY; the eleventh digit is the check digit
MOBILE_PREFIX, MOBILE_DIGITS = "69", 8
LANDLINE_PREFIX, LANDLINE_DIGITS = "2", 9
PHONE_GROUP = 3  # separators split a number into 3-3-rest digits

CASES = ("gen", "acc", "voc")
# Ending -> (genitive, accusative, vocative) endings. A name matching no ending stays as it is.
MALE_ENDINGS = {
    "ος": ("ου", "ο", "ε"),
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
}

# Names kept out of the pools: -ων names whose genitive is -ονος or -οντος (not -ωνος), Βάιος
# (genitive Βαΐου needs a diaeresis), and Faker names with a missing or doubled accent.
EXCLUDED_NAMES = {
    "Αγαμέμνων", "Δράκων", "Ιάσων", "Λέων", "Ναπολέων", "Παντελεήμων", "Σώζων", "Τιμολέων",
    "Φιλήμων", "Βάιος", "Κουτσογιάννόπουλος", "Τηλεμαχος", "Βουτσελας", "Τζαλλας", "Αίας",
    "Τσιωλξ",
}  # fmt: skip

ACCENT = dict(zip("αεηιουωΑΕΗΙΟΥΩ", "άέήίόύώΆΈΉΊΌΎΏ", strict=True))
ACCENTED = set(ACCENT.values()) | set("ΐΰ")
VOWELS = set(ACCENT) | ACCENTED | set("ϊϋ")
DIPHTHONGS = {"αι", "ει", "οι", "υι", "ου", "αυ", "ευ", "ηυ"}
DIAERESIS = "ϊϋΐΰ"  # breaks a diphthong: Βαϊτσόπουλος has four syllables before -λος

ARTICLES = {
    ("M", "nom"): "ο",
    ("M", "gen"): "του",
    ("M", "acc"): "τον",
    ("F", "nom"): "η",
    ("F", "gen"): "της",
    ("F", "acc"): "την",
}

# The accusative article keeps its -ν (την, τον) even before consonants, as written text does.
# Placeholders: {PERSON:case} for Greek (adds the article), {PERSON} for English, {AFM} etc.
# {FIRST:voc} and {SURNAME:voc} are the first name or surname alone, {ADDRESS} is
# "Αγαπητέ κύριε" or "Αγαπητή κυρία" by the person's gender.
TEMPLATES = {
    "train": {
        "greek": [
            "Η αίτηση {PERSON:gen} καταχωρήθηκε με ΑΦΜ {AFM}.",
            "Καλημέρα, {PERSON:voc}. Ο ΑΜΚΑ σας είναι {AMKA} και το τηλέφωνό σας {PHONE}.",
            "Σύμφωνα με το αρχείο, {PERSON:nom} έστειλε τα έγγραφα στο {EMAIL}.",
            "Η εφορία ειδοποίησε {PERSON:acc} για την επιστροφή φόρου στο IBAN {IBAN}.",
            "{ADDRESS} {SURNAME:voc}, η δήλωσή σας με ΑΦΜ {AFM} εγκρίθηκε.",
            "{FIRST:voc}, σου έστειλα το αρχείο. Γράψε μου στο {EMAIL}.",
        ],
        "english": [
            "Please update the record for {PERSON}: tax ID {AFM}, phone {PHONE}.",
            "Patient {PERSON} has social security number {AMKA}. Reply to {EMAIL}.",
            "Invoice sent to {PERSON}, payable to {IBAN}.",
        ],
    },
    "test": {
        "greek": [
            "Το ραντεβού {PERSON:gen} έχει οριστεί για Τρίτη. ΑΜΚΑ {AMKA}, τηλ. {PHONE}.",
            "Είδα {PERSON:acc} χθες στο ιατρείο. Στείλτε τα αποτελέσματα στο {EMAIL}.",
            "Σύμφωνα με την τράπεζα, {PERSON:nom} έχει λογαριασμό {IBAN} και ΑΦΜ {AFM}.",
            "Καλησπέρα, {PERSON:voc}. Θα σας καλέσουμε στο {PHONE}.",
            "{ADDRESS} {SURNAME:voc}, ο λογαριασμός {IBAN} είναι ενεργός.",
            "{FIRST:voc}, πάρε με στο {PHONE} όταν μπορείς.",
        ],
        "english": [
            "Please call {PERSON} on {PHONE} about the claim, tax ID {AFM}.",
            "Refund for {PERSON}, social security number {AMKA}, to {IBAN}. Contact {EMAIL}.",
        ],
    },
}

ADDRESS_FORMS = {"M": "Αγαπητέ κύριε", "F": "Αγαπητή κυρία"}
SLOT = re.compile(r"\{(\w+)(?::(\w+))?\}")

GREEK_TO_LATIN = {
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th",
    "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p",
    "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps",
    "ω": "o",
}  # fmt: skip
EMAIL_DOMAINS = ["example.gr", "example.com", "mail.example.org"]
GR_IBAN_PREFIX_DIGITS = "1627"  # G=16, R=27 as in the IBAN check-digit rule
IBAN_BODY_DIGITS = 23
IBAN_MODULUS = 97
BIRTH_RANGE = (date(1930, 1, 1), date(2020, 12, 31))


@dataclass
class Person:
    gender: str  # "M" or "F"
    first: str
    surname: str
    greek: bool

    def first_form(self, case: str) -> str:
        if not self.greek:
            return self.first
        return inflect(
            self.first, MALE_ENDINGS if self.gender == "M" else FEMALE_FIRST_ENDINGS, case
        )

    def surname_form(self, case: str) -> str:
        # Female surnames (-ου, -η, -α) do not change between cases.
        if not self.greek or self.gender == "F":
            return self.surname
        return inflect(self.surname, MALE_ENDINGS, case)

    def name(self, case: str) -> str:
        return f"{self.first_form(case)} {self.surname_form(case)}"


def syllables(word: str) -> list[tuple[int, int]]:
    spans = []
    i = 0
    while i < len(word):
        if word[i] not in VOWELS:
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


def shift_accent(word: str) -> str:
    """The long ending -ου cannot follow an accent on the third syllable from the end."""
    spans = syllables(word)
    accented = [i for i, (start, end) in enumerate(spans) if ACCENTED & set(word[start:end])]
    if not word.endswith("ου") or accented != [len(spans) - 3]:
        return word
    (old_start, old_end), (_, new_end) = spans[accented[0]], spans[accented[0] + 1]
    old = "".join(strip_accents(c) if c in ACCENTED else c for c in word[old_start:old_end])
    # The mark goes on the last vowel of a diphthong: ει -> εί.
    new_last = ACCENT[word[new_end - 1]]
    return word[:old_start] + old + word[old_end : new_end - 1] + new_last + word[new_end:]


def inflect(name: str, endings: dict[str, tuple[str, str, str]], case: str) -> str:
    if case == "nom":
        return name
    for ending, forms in endings.items():
        if name.endswith(ending):
            inflected = name[: -len(ending)] + forms[CASES.index(case)]
            return shift_accent(inflected) if case == "gen" else inflected
    return name


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def stem(name: str) -> str:
    return re.sub(r"(ος|ης|ας|ου|η|α|ο)$", "", strip_accents(name))


@cache
def split_stems(split: str, greek: bool) -> set[str]:
    """Stems of every first name and surname of a language that belong to this split."""
    provider = el_GR.Provider if greek else en_US.Provider
    last_names = (
        provider.last_names_male + provider.last_names_female if greek else provider.last_names
    )
    names = [*provider.first_names_male, *provider.first_names_female, *last_names]
    return set(sorted({stem(name) for name in names})[SPLIT_PARITY[split] :: 2])


@cache
def first_names(split: str, greek: bool, gender: str) -> list[str]:
    provider = el_GR.Provider if greek else en_US.Provider
    names = provider.first_names_male if gender == "M" else provider.first_names_female
    endings = MALE_ENDINGS if gender == "M" else FEMALE_FIRST_ENDINGS
    return [
        name
        for name in names
        if stem(name) in split_stems(split, greek)
        and (not greek or (name.endswith(tuple(endings)) and name not in EXCLUDED_NAMES))
    ]


@cache
def surnames(split: str, greek: bool, gender: str) -> list[str]:
    if not greek:
        return [n for n in en_US.Provider.last_names if stem(n) in split_stems(split, False)]
    names = el_GR.Provider.last_names_male if gender == "M" else el_GR.Provider.last_names_female
    # Female surnames never inflect, so only male ones need a matching ending.
    return [
        name
        for name in names
        if stem(name) in split_stems(split, True)
        and (gender == "F" or name.endswith(tuple(MALE_ENDINGS)))
        and name not in EXCLUDED_NAMES
    ]


def latin_letters(name: str) -> str:
    stripped = strip_accents(name.lower())
    latin = "".join(GREEK_TO_LATIN.get(c, c) for c in stripped.replace("ου", "ou"))
    return "".join(c for c in latin if c.isascii() and c.isalnum())


def digits(rng: random.Random, count: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(count))


def make_afm(rng: random.Random) -> str:
    body = digits(rng, AFM_BODY_DIGITS)
    powers = range(AFM_BODY_DIGITS, 0, -1)
    weighted = sum(int(d) * 2**power for d, power in zip(body, powers, strict=True))
    return body + str(weighted % 11 % 10)


def make_amka(rng: random.Random) -> str:
    birth = date.fromordinal(rng.randint(BIRTH_RANGE[0].toordinal(), BIRTH_RANGE[1].toordinal()))
    body = birth.strftime("%d%m%y") + digits(rng, AMKA_SERIAL_DIGITS)
    total = 0
    for position, char in enumerate(reversed(body)):
        digit = int(char)
        if position % 2 == 0:  # the check digit will sit to the right of the last body digit
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return body + str((10 - total % 10) % 10)


def make_iban(rng: random.Random) -> str:
    body = digits(rng, IBAN_BODY_DIGITS)
    check = IBAN_MODULUS + 1 - int(body + GR_IBAN_PREFIX_DIGITS + "00") % IBAN_MODULUS
    compact = f"GR{check:02d}{body}"
    if rng.random() < COMPACT_IBAN_SHARE:
        return compact
    return " ".join(compact[i : i + 4] for i in range(0, len(compact), 4))


def make_phone(rng: random.Random) -> str:
    if rng.random() < MOBILE_SHARE:
        number = MOBILE_PREFIX + digits(rng, MOBILE_DIGITS)
    else:
        number = LANDLINE_PREFIX + digits(rng, LANDLINE_DIGITS)
    separator = rng.choice(["", " ", "-"])
    if separator:
        groups = [
            number[:PHONE_GROUP],
            number[PHONE_GROUP : 2 * PHONE_GROUP],
            number[2 * PHONE_GROUP :],
        ]
        number = separator.join(groups)
    prefix = rng.choice(["", "+30", "0030"])
    return prefix + (separator if prefix else "") + number


def make_email(rng: random.Random, person: Person) -> str:
    first, surname = latin_letters(person.first), latin_letters(person.surname)
    local = rng.choice([f"{first}.{surname}", f"{first[0]}.{surname}", f"{first}{surname}"])
    if rng.random() < EMAIL_NUMBER_SHARE:
        local += digits(rng, EMAIL_NUMBER_DIGITS)
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def pick_person(rng: random.Random, split: str, greek: bool) -> Person:
    gender = rng.choice(["M", "F"])
    first = rng.choice(first_names(split, greek, gender))
    return Person(gender, first, rng.choice(surnames(split, greek, gender)), greek)


def make_record(rng: random.Random, split: str) -> dict:
    greek = rng.random() < GREEK_SHARE
    template = rng.choice(TEMPLATES[split]["greek" if greek else "english"])
    person = pick_person(rng, split, greek)
    id_makers = {"AFM": make_afm, "AMKA": make_amka, "IBAN": make_iban, "PHONE": make_phone}

    text = ""
    spans = []
    cursor = 0
    for slot in SLOT.finditer(template):
        text += template[cursor : slot.start()]
        cursor = slot.end()
        kind, case = slot.groups()
        if kind == "ADDRESS":
            text += ADDRESS_FORMS[person.gender]
            continue
        if kind == "PERSON":
            if greek and case != "voc":
                text += ARTICLES[person.gender, case] + " "
            surface = person.name(case or "nom")
        elif kind == "FIRST":
            surface = person.first_form(case)
        elif kind == "SURNAME":
            surface = person.surname_form(case)
        elif kind == "EMAIL":
            surface = make_email(rng, person)
        else:
            surface = id_makers[kind](rng)
        if kind in ("FIRST", "SURNAME"):
            kind = "PERSON"
        spans.append([len(text), len(text) + len(surface), kind])
        text += surface
    text += template[cursor:]
    return {"text": text, "spans": spans}


def generate(seed: int, split: str, count: int) -> list[dict]:
    rng = random.Random(seed)
    return [make_record(rng, split) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=SPLIT_PARITY, default="train")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or Path("data") / f"synthetic_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as file:
        for record in generate(args.seed, args.split, args.count):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
