import pytest

from greek_pii_redactor.inflect import inflect_name, nominative


@pytest.mark.parametrize(
    "name, gender, case, surname_only, expected",
    [
        ("Νίκος", "M", "vocative", False, "Νίκο"),
        ("Γιώργος", "M", "vocative", False, "Γιώργο"),
        ("Κυριάκος", "M", "vocative", False, "Κυριάκο"),
        ("Κωνσταντίνος", "M", "vocative", False, "Κωνσταντίνε"),
        ("Ματθαίος", "M", "vocative", False, "Ματθαίε"),
        ("Ξανθός", "M", "vocative", False, "Ξανθέ"),
        ("Λουκάς", "M", "vocative", False, "Λουκά"),
        ("Γεώργιος", "M", "genitive", False, "Γεωργίου"),
        ("Νίκος Μπούσιος", "M", "genitive", False, "Νίκου Μπούσιου"),
        ("Νίκος Παπαδόπουλος", "M", "accusative", False, "Νίκο Παπαδόπουλο"),
        ("Παπαδόπουλος", "M", "genitive", True, "Παπαδοπούλου"),
        ("Παπαδόπουλος", "M", "vocative", True, "Παπαδόπουλε"),
        ("Γιάννης Αντωνίου", "M", "genitive", False, "Γιάννη Αντωνίου"),
        ("Μαρία", "F", "genitive", False, "Μαρίας"),
        ("Μαντώ", "F", "genitive", False, "Μαντώς"),
        ("Μαρία Παπαδοπούλου", "F", "genitive", False, "Μαρίας Παπαδοπούλου"),
        ("Ελένη Παπαδοπούλου", "F", "genitive", False, "Ελένης Παπαδοπούλου"),
        # Alone, -η is more often a surname than a first name, so it is not changed.
        ("Ελένη", "F", "genitive", False, "Ελένη"),
    ],
)
def test_inflect_name(name: str, gender: str, case: str, surname_only: bool, expected: str) -> None:
    assert inflect_name(name, gender, case, surname_only) == expected


@pytest.mark.parametrize(
    "form, gender, case, surname_only, expected",
    [
        ("Νίκου Παπαδοπούλου", "M", "genitive", False, "Νίκος Παπαδόπουλος"),
        ("Γιώργου", "M", "genitive", False, "Γιώργος"),
        ("Γεωργίου", "M", "genitive", False, "Γεώργιος"),
        ("Μαρίου", "M", "genitive", False, "Μάριος"),
        ("Ματθαίου", "M", "genitive", False, "Ματθαίος"),
        ("Ξανθού", "M", "genitive", False, "Ξανθός"),
        ("Γιάννη", "M", "genitive", False, "Γιάννης"),
        ("Νίκο Παπαδόπουλο", "M", "accusative", False, "Νίκος Παπαδόπουλος"),
        ("Νίκο", "M", "vocative", False, "Νίκος"),
        ("Ξανθέ", "M", "vocative", True, "Ξανθός"),
        ("Αντωνίου", "M", "accusative", True, "Αντωνίου"),
        ("Μαρίας Γεωργίου", "F", "genitive", False, "Μαρία Γεωργίου"),
        ("Μαντώς", "F", "genitive", False, "Μαντώ"),
        ("Ραϊπούλου", "M", "genitive", True, "Ραΐπουλος"),
        # Κωνσταντίνος or Αλέξανδρος: the accent of the nominative is unknown.
        ("Κωνσταντίνου", "M", "genitive", False, None),
        # A surname in -ου that never changes, or the genitive of a name in -ος (Βλάχου).
        ("Αντωνίου", "M", "genitive", True, None),
        # No -ς, so this is not a genitive form.
        ("Ελένη", "F", "genitive", False, None),
    ],
)
def test_nominative(
    form: str, gender: str, case: str, surname_only: bool, expected: str | None
) -> None:
    assert nominative(form, gender, case, surname_only) == expected
