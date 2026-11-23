"""Finds ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers and emails with regexes; the first three need a valid
check digit."""

import re
from datetime import date
from typing import NamedTuple

AFM_DIGITS = 9
AMKA_DIGITS = 11
AFM_MODULUS = 11
IBAN_LENGTH = 27  # ISO 13616 length for Greece
IBAN_MODULUS = 97
LUHN_MODULUS = 10


class Span(NamedTuple):
    start: int
    end: int
    kind: str


AFM_PATTERN = re.compile(rf"(?<!\d)\d{{{AFM_DIGITS}}}(?!\d)")
AMKA_PATTERN = re.compile(rf"(?<!\d)\d{{{AMKA_DIGITS}}}(?!\d)")
IBAN_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])GR\d{2}(?: ?[0-9A-Z]{4}){5} ?[0-9A-Z]{3}(?![0-9A-Za-z])"
)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
# Mobile 69 + 8 digits, landline 2 + 9 digits; a space or hyphen may sit between any two digits.
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:\+30|0030)[ -]?)?(?:69(?:[ -]?\d){8}|2(?:[ -]?\d){9})(?!\d)"
)

# Listed in the order that wins when two matches overlap.
PATTERNS = [
    ("IBAN", IBAN_PATTERN),
    ("EMAIL", EMAIL_PATTERN),
    ("AMKA", AMKA_PATTERN),
    ("AFM", AFM_PATTERN),
    ("PHONE", PHONE_PATTERN),
]


def afm_is_valid(digits: str) -> bool:
    weighted_sum = sum(int(d) * 2 ** (AFM_DIGITS - 1 - i) for i, d in enumerate(digits[:-1]))
    return int(digits[-1]) == weighted_sum % AFM_MODULUS % 10


def luhn_is_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % LUHN_MODULUS == 0


def amka_is_valid(digits: str) -> bool:
    day, month, year = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
    # The century is not stored, so the date only has to exist in one of them.
    for century in (1900, 2000):
        try:
            date(century + year, month, day)
        except ValueError:
            continue
        return luhn_is_valid(digits)
    return False


def iban_is_valid(iban: str) -> bool:
    iban = iban.replace(" ", "")
    if len(iban) != IBAN_LENGTH:
        return False
    rearranged = iban[4:] + iban[:4]
    as_digits = "".join(str(int(char, 36)) for char in rearranged)
    return int(as_digits) % IBAN_MODULUS == 1


VALIDATORS = {"AFM": afm_is_valid, "AMKA": amka_is_valid, "IBAN": iban_is_valid}


def find_ids(text: str) -> list[Span]:
    accepted: list[Span] = []
    claimed: list[Span] = []
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(text):
            span = Span(match.start(), match.end(), kind)
            if any(span.start < c.end and c.start < span.end for c in claimed):
                continue
            valid = kind not in VALIDATORS or VALIDATORS[kind](match.group())
            # A mistyped IBAN still claims its text, or its digit groups would match as phones.
            if valid or kind == "IBAN":
                claimed.append(span)
            if valid:
                accepted.append(span)
    return sorted(accepted)
