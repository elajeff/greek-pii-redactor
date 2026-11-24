import pytest

from greek_pii_redactor.redact import redact, restore
from greek_pii_redactor.rules import Span, find_ids


@pytest.mark.parametrize(
    "text, kind",
    [
        ("123456783", "AFM"),
        ("15038512347", "AMKA"),
        ("29020005673", "AMKA"),
        ("GR16 0110 1250 0000 0001 2300 695", "IBAN"),
        ("GR1601101250000000012300695", "IBAN"),
        ("6912345678", "PHONE"),
        ("2101234567", "PHONE"),
        ("+30 691 234 5678", "PHONE"),
        ("0030-210-123-4567", "PHONE"),
        ("maria.papa+work@example.gr", "EMAIL"),
    ],
)
def test_finds_valid_id(text: str, kind: str) -> None:
    sentence = f"Στείλτε στο {text} σήμερα."
    start = sentence.index(text)
    assert find_ids(sentence) == [Span(start, start + len(text), kind)]


@pytest.mark.parametrize(
    "text",
    [
        "123456784",  # AFM check digit off by one
        "15038512346",  # AMKA Luhn digit wrong
        "31139912341",  # AMKA with an invalid month
        "GR16 0110 1250 0000 0001 2300 696",  # IBAN mod-97 fails
        "gr16 0110 1250 0000 0001 2300 695",  # lowercase prefix
        "GR27 1013 8032 3411 1455 9335 347",  # bad IBAN whose first groups look like a phone
    ],
)
def test_rejects_failed_validation(text: str) -> None:
    assert find_ids(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "1234567830",  # valid AFM plus one digit
        "0123456783",
        "150385123470",
        "69123456789",  # mobile plus one digit
        "306912345678",  # no + before the country code
    ],
)
def test_ignores_digits_inside_longer_run(text: str) -> None:
    assert find_ids(text) == []


def test_several_ids_in_one_sentence() -> None:
    text = (
        "Ο Γιώργος, ΑΦΜ 123456783, ΑΜΚΑ 15038512347, τηλ. 6912345678, "
        "email giorgos@example.gr, IBAN GR16 0110 1250 0000 0001 2300 695."
    )
    found = [(text[s.start : s.end], s.kind) for s in find_ids(text)]
    assert found == [
        ("123456783", "AFM"),
        ("15038512347", "AMKA"),
        ("6912345678", "PHONE"),
        ("giorgos@example.gr", "EMAIL"),
        ("GR16 0110 1250 0000 0001 2300 695", "IBAN"),
    ]


def test_overlapping_matches_keep_highest_priority() -> None:
    assert find_ids("123456783@example.gr") == [Span(0, 20, "EMAIL")]
    assert find_ids("2101234567") == [Span(0, 10, "PHONE")]


@pytest.mark.parametrize("digits", ["０１２３４５６７８９", "٠١٢٣٤٥٦٧٨٩"])
def test_finds_ids_written_in_other_digits(digits: str) -> None:
    to_script = str.maketrans("0123456789", digits)
    ascii_ids = ("123456783", "15038512347", "GR16 0110 1250 0000 0001 2300 695", "6912345678")
    afm, amka, iban, phone = (number.translate(to_script) for number in ascii_ids)
    text = f"ΑΦΜ {afm}, ΑΜΚΑ {amka}, IBAN {iban}, τηλ. {phone}."
    found = [(text[s.start : s.end], s.kind) for s in find_ids(text)]
    assert found == [(afm, "AFM"), (amka, "AMKA"), (iban, "IBAN"), (phone, "PHONE")]
    redacted, placeholders = redact(text)
    assert restore(redacted, placeholders) == text
