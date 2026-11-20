import re
from collections.abc import Callable

import pytest

from greek_pii_redactor.redact import redact, restore
from greek_pii_redactor.rules import Span


def find_names(*names: str) -> Callable[[str], list[Span]]:
    def detect(text: str) -> list[Span]:
        return sorted(
            Span(match.start(), match.end(), "PERSON")
            for name in names
            for match in re.finditer(rf"\b{re.escape(name)}\b", text)
        )

    return detect


@pytest.mark.parametrize(
    "text, names",
    [
        (
            "Ο Νίκος Παπαδόπουλος έδωσε το ΑΦΜ 123456783 και το τηλέφωνο 691 234 5678.",
            ("Νίκος Παπαδόπουλος",),
        ),
        (
            "Μίλησα με τον Νίκο Παπαδόπουλο και την Μαρία Γεωργίου. "
            "Email: nikos@example.gr, IBAN GR16 0110 1250 0000 0001 2300 695.",
            ("Νίκο Παπαδόπουλο", "Μαρία Γεωργίου"),
        ),
        (
            "Please call John Smith on 210 123 4567 or write to john.smith@example.com "
            "about AMKA 15038512347.",
            ("John Smith",),
        ),
    ],
)
def test_restore_of_redacted_text_gives_the_input_back(text: str, names: tuple[str, ...]) -> None:
    redacted, placeholders = redact(text, find_names(*names))
    assert text not in redacted
    assert restore(redacted, placeholders) == text


@pytest.mark.parametrize(
    "names, people",
    [
        (("Νίκος", "Νίκου", "Νίκο"), 1),
        (("Νίκος Παπαδόπουλος", "Νίκου Παπαδοπούλου"), 1),
        (("Γιάννης", "Γιάννη"), 1),
        (("Κώστας", "Κώστα"), 1),
        (("Μαρία", "Μαρίας"), 1),
        (("Ελένη", "Ελένης"), 1),
        (("Νίκος", "Νίκη"), 2),
        (("Αλέξανδρος", "Αλεξάνδρα"), 2),
        (("Μαρία", "Μαρίνα"), 2),
        (("Μαρία", "Μαριάνα"), 2),
        (("Γεωργία", "Γεωργίου"), 2),
        (("Γιάννης", "Γιάννα"), 2),
        (("Νίκος Παπαδόπουλος", "Νίκη Παπαδοπούλου"), 2),
        (("Νίκος Παπαδόπουλος", "Νίκος Γεωργίου"), 2),
        (("Νίκου", "Νίκη"), 2),
        (("Νίκου", "Νίκης"), 2),
        (("Κυριακή", "Κυριάκου"), 2),
        (("Κυριακής", "Κυριάκου"), 2),
        (("Στεφάνου", "Στέφανη"), 2),
        (("Δάφνη", "Δάφνου"), 2),
        (("Νίκη Παπαδοπούλου", "Νίκου Παπαδοπούλου"), 2),
        (("Ζωή", "Ζωής"), 1),
        (("Άρης", "Άρη"), 1),
        (("Λία", "Λίας"), 1),
        (("Εύα", "Εύας"), 1),
    ],
)
def test_forms_of_one_person_share_a_placeholder(names: tuple[str, ...], people: int) -> None:
    text = " και ".join(names) + "."
    _, placeholders = redact(text, find_names(*names))
    assert len(placeholders) == people


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Call 6912345678 or 691 234 5678, tax ID 123456783 and again 123456783.",
            "Call [PHONE_1] or [PHONE_1], tax ID [AFM_1] and again [AFM_1].",
        ),
        (
            "IBAN GR1601101250000000012300695, mail Nikos@Example.gr and nikos@example.gr.",
            "IBAN [IBAN_1], mail [EMAIL_1] and [EMAIL_1].",
        ),
    ],
)
def test_same_id_shares_a_placeholder(text: str, expected: str) -> None:
    redacted, _ = redact(text)
    assert redacted == expected


def test_id_wins_over_overlapping_name_span() -> None:
    text = "Καλέστε το 6912345678 σήμερα."
    start = text.index("το")

    def detect(_: str) -> list[Span]:
        return [Span(start, text.index("σήμερα") - 1, "PERSON")]

    redacted, placeholders = redact(text, detect)
    assert redacted == "Καλέστε το [PHONE_1] σήμερα."
    assert list(placeholders) == ["[PHONE_1]"]


@pytest.mark.parametrize(
    "text, names, reply, expected",
    [
        *[
            (
                "Ο Νίκος Παπαδόπουλος ήρθε. Είδα τον Νίκο Παπαδόπουλο.",
                ("Νίκος Παπαδόπουλος", "Νίκο Παπαδόπουλο"),
                reply,
                expected,
            )
            for reply, expected in [
                ("Ευχαριστώ τον [PERSON_1].", "Ευχαριστώ τον Νίκο Παπαδόπουλο."),
                ("Ο [PERSON_1] απάντησε.", "Ο Νίκος Παπαδόπουλος απάντησε."),
                ("ο [PERSON_1] απάντησε.", "ο Νίκος Παπαδόπουλος απάντησε."),
                # The input has no genitive, so the first form seen is used.
                ("Το βιβλίο του [PERSON_1].", "Το βιβλίο του Νίκος Παπαδόπουλος."),
            ]
        ],
        *[
            ("Η Ελένη ήρθε. Το βιβλίο της Ελένης.", ("Ελένη", "Ελένης"), reply, expected)
            for reply, expected in [
                ("Η [PERSON_1] ήρθε.", "Η Ελένη ήρθε."),
                ("Είπα στον [PERSON_1] ότι ήρθε.", "Είπα στον Ελένη ότι ήρθε."),
                ("Μίλησα με την [PERSON_1].", "Μίλησα με την Ελένη."),
                ("Το βιβλίο της [PERSON_1].", "Το βιβλίο της Ελένης."),
            ]
        ],
    ],
)
def test_restore_picks_the_form_that_fits_the_article(
    text: str, names: tuple[str, ...], reply: str, expected: str
) -> None:
    _, placeholders = redact(text, find_names(*names))
    assert restore(reply, placeholders) == expected


def test_restore_when_the_reply_dropped_or_invented_placeholders() -> None:
    text = "Ο Νίκος Παπαδόπουλος, ΑΦΜ 123456783, τηλ. 6912345678."
    _, placeholders = redact(text, find_names("Νίκος Παπαδόπουλος"))
    reply = "Το ΑΦΜ [AFM_1] και το [AFM_7]."
    assert restore(reply, placeholders) == "Το ΑΦΜ 123456783 και το [AFM_7]."


@pytest.mark.parametrize("spans", [[(2, 7), (2, 20)], [(8, 20), (2, 13)]])
def test_overlapping_name_spans_become_one(spans: list[tuple[int, int]]) -> None:
    text = "Ο Νίκος Παπαδόπουλος ήρθε."

    def detect(_: str) -> list[Span]:
        return [Span(start, end, "PERSON") for start, end in spans]

    redacted, placeholders = redact(text, detect)
    assert redacted == "Ο [PERSON_1] ήρθε."
    assert placeholders["[PERSON_1]"][0]["text"] == "Νίκος Παπαδόπουλος"


def test_text_that_already_has_a_placeholder_is_refused() -> None:
    with pytest.raises(ValueError, match="already contains"):
        redact("Στείλτε το στον [PERSON_1].")
