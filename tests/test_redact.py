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
    redacted, placeholders = redact(text)
    assert redacted == expected
    # Only a name's form depends on the word before it, so an ID is stored once per spelling.
    assert all(form.keys() == {"text"} for forms in placeholders.values() for form in forms)


def test_placeholders_are_shared_across_calls() -> None:
    _, placeholders = redact("Mail nikos@example.gr.")
    text = "Mail nikos@example.gr or maria@example.gr."
    redacted, placeholders = redact(text, None, placeholders)
    assert redacted == "Mail [EMAIL_1] or [EMAIL_2]."
    assert list(placeholders) == ["[EMAIL_1]", "[EMAIL_2]"]


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
                # The input has no genitive, so it is derived from the nominative.
                ("Το βιβλίο του [PERSON_1].", "Το βιβλίο του Νίκου Παπαδοπούλου."),
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
        # A token without accents or in capitals gets its accent from a list of names, or tries each
        # position if unlisted, and its output is written back the way it was seen. An unlisted -ος
        # name takes the vocative -ο (Ζβορκο, not Ζβορκέ) unless a seen form with an accent decides.
        *[
            (text, (name,), reply, expected)
            for text, name, reply, expected in [
                (
                    "Ο Νικος Παπαδοπουλος ήρθε.",
                    "Νικος Παπαδοπουλος",
                    "Καλημέρα, [PERSON_1].",
                    "Καλημέρα, Νικο Παπαδοπουλε.",
                ),
                (
                    "Ο Νικος Παπαδόπουλος ήρθε.",
                    "Νικος Παπαδόπουλος",
                    "Καλημέρα, [PERSON_1].",
                    "Καλημέρα, Νικο Παπαδόπουλε.",
                ),
                (
                    "Ο ΝΙΚΟΣ Παπαδοπουλος ήρθε.",
                    "ΝΙΚΟΣ Παπαδοπουλος",
                    "Το βιβλίο του [PERSON_1].",
                    "Το βιβλίο του ΝΙΚΟΥ Παπαδοπουλου.",
                ),
                ("Ο ΚΑΡΑΪΣΚΟΣ ήρθε.", "ΚΑΡΑΪΣΚΟΣ", "Του [PERSON_1].", "Του ΚΑΡΑΪΣΚΟΥ."),
                ("Ο ΚΑΡΑΙΣΚΟΣ ήρθε.", "ΚΑΡΑΙΣΚΟΣ", "Του [PERSON_1].", "Του ΚΑΡΑΙΣΚΟΥ."),
                ("Ο ΗΛΙΑΣ ήρθε.", "ΗΛΙΑΣ", "Καλημέρα, [PERSON_1].", "Καλημέρα, ΗΛΙΑ."),
                ("Ο νικος ήρθε.", "νικος", "Καλημέρα, [PERSON_1].", "Καλημέρα, νικο."),
                ("Το βιβλίο του Γεωργιου.", "Γεωργιου", "Ο [PERSON_1] ήρθε.", "Ο Γεωργιος ήρθε."),
                ("Ο Ζβορκος ήρθε.", "Ζβορκος", "Καλημέρα, [PERSON_1].", "Καλημέρα, Ζβορκο."),
                ("Ο ΝΊΚΟΣ ήρθε.", "ΝΊΚΟΣ", "Του [PERSON_1].", "Του ΝΊΚΟΥ."),
                ("Ο ΓΙΑΝΝΗΣ ΣΚΑΣ ήρθε.", "ΓΙΑΝΝΗΣ ΣΚΑΣ", "Του [PERSON_1].", "Του ΓΙΑΝΝΗ ΣΚΑ."),
                ("Ο ΖΒΟΡΚΟΣ ήρθε.", "ΖΒΟΡΚΟΣ", "Του [PERSON_1].", "Του ΖΒΟΡΚΟΥ."),
            ]
        ],
        # The -ο rule goes by the nominative, so it also works from a seen genitive.
        (
            "Το σπίτι του Ζβορκου.",
            ("Ζβορκου",),
            "Καλημέρα, [PERSON_1].",
            "Καλημέρα, Ζβορκο.",
        ),
        # Each part of a hyphenated surname is looked up on its own: Κωστόπουλος is listed.
        (
            "Ο Νικος Παπαδοπουλος-Κωστοπουλος ήρθε.",
            ("Νικος Παπαδοπουλος-Κωστοπουλος",),
            "Καλημέρα, [PERSON_1].",
            "Καλημέρα, Νικο Παπαδοπουλος-Κωστοπουλε.",
        ),
        # An unlisted name whose accent readings disagree is inflected as it was written.
        ("Η Κλαρα ήρθε.", ("Κλαρα",), "Το βιβλίο της [PERSON_1].", "Το βιβλίο της Κλαρας."),
        # The diaeresis of the name as seen stays, though the listed form (Χάιδω) has none.
        ("Η ΧΑΪΔΩ ήρθε.", ("ΧΑΪΔΩ",), "Το βιβλίο της [PERSON_1].", "Το βιβλίο της ΧΑΪΔΩΣ."),
        # A listed name whose accents disagree (Ίωνα or Ιωνά) stays as seen, not guessed.
        ("Μίλησα με τον ΙΩΝΑ.", ("ΙΩΝΑ",), "Του [PERSON_1].", "Του ΙΩΝΑ."),
        # A seen form with an accent decides before the -ο rule for unlisted names.
        (
            "Ο Ζβορκος ήρθε. Του Ζβόρκου.",
            ("Ζβορκος", "Ζβόρκου"),
            "Καλημέρα, [PERSON_1].",
            "Καλημέρα, Ζβόρκο.",
        ),
    ],
)
def test_restore_picks_the_form_that_fits_the_article(
    text: str, names: tuple[str, ...], reply: str, expected: str
) -> None:
    _, placeholders = redact(text, find_names(*names))
    assert restore(reply, placeholders) == expected


@pytest.mark.parametrize(
    "reply, expected",
    [
        (
            "Το βιβλίο του [PERSON_1] και της [PERSON_2].",
            "Το βιβλίο του Νίκου Παπαδοπούλου και της Μαρίας Γεωργίου.",
        ),
        (
            "Είδα τον [PERSON_1] και την [PERSON_2].",
            "Είδα τον Νίκο Παπαδόπουλο και την Μαρία Γεωργίου.",
        ),
        ("Καλημέρα, [PERSON_1].", "Καλημέρα, Νίκο Παπαδόπουλε."),
        ("Καλησπέρα σας [PERSON_1].", "Καλησπέρα σας Νίκο Παπαδόπουλε."),
    ],
)
def test_restore_derives_the_case_the_reply_needs(reply: str, expected: str) -> None:
    text = "Ο Νίκος Παπαδόπουλος ήρθε με τη Μαρία Γεωργίου."
    _, placeholders = redact(text, find_names("Νίκος Παπαδόπουλος", "Μαρία Γεωργίου"))
    assert restore(reply, placeholders) == expected


@pytest.mark.parametrize(
    "text, name, reply, expected",
    [
        (
            "Ο κύριος Παπαδόπουλος ήρθε.",
            "Παπαδόπουλος",
            "Αγαπητέ κύριε [PERSON_1],",
            "Αγαπητέ κύριε Παπαδόπουλε,",
        ),
        # A title in the reply does not make a first name a surname.
        ("Ο Γεώργιος ήρθε.", "Γεώργιος", "Του κυρίου [PERSON_1].", "Του κυρίου Γεωργίου."),
        # The title in the input makes the name a surname, even when the reply has no title.
        (
            "Ο κύριος Μπούσιος ήρθε.",
            "Μπούσιος",
            "Το βιβλίο του [PERSON_1].",
            "Το βιβλίο του Μπούσιου.",
        ),
    ],
)
def test_restore_treats_a_lone_name_after_a_title_as_a_surname(
    text: str, name: str, reply: str, expected: str
) -> None:
    _, placeholders = redact(text, find_names(name))
    assert restore(reply, placeholders) == expected


def test_restore_when_the_reply_dropped_or_invented_placeholders() -> None:
    text = "Ο Νίκος Παπαδόπουλος, ΑΦΜ 123456783, τηλ. 6912345678."
    _, placeholders = redact(text, find_names("Νίκος Παπαδόπουλος"))
    reply = "Το ΑΦΜ [AFM_1] και το [AFM_7]."
    # An unknown placeholder loses its brackets, so the next turn passes redact's placeholder check.
    restored = restore(reply, placeholders)
    assert restored == "Το ΑΦΜ 123456783 και το AFM_7."
    redact(restored)


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
