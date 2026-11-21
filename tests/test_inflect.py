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
        ("Μαλτέζος", "M", "vocative", True, "Μαλτέζε"),
        ("Παυλάκος", "M", "vocative", True, "Παυλάκο"),
        ("Τζαννετάκος", "M", "vocative", False, "Τζαννετάκο"),
        ("Βούλγαρος", "M", "genitive", True, "Βούλγαρου"),
        ("Λαλές", "M", "genitive", True, "Λαλέ"),
        ("Δέδες", "M", "genitive", True, "Δέδε"),
        ("Στεφανάκις", "M", "vocative", True, "Στεφανάκι"),
        ("Παππούς", "M", "accusative", True, "Παππού"),
        ("Κεραμεύς", "M", "genitive", True, "Κεραμέως"),
        ("Κεραμεύς", "M", "accusative", True, "Κεραμέα"),
        ("Κεραμεύς", "M", "vocative", True, "Κεραμεύ"),
        ("Γιάννης Αντωνίου", "M", "genitive", False, "Γιάννη Αντωνίου"),
        ("Μαρία", "F", "genitive", False, "Μαρίας"),
        ("Μαντώ", "F", "genitive", False, "Μαντώς"),
        ("Μαρία Παπαδοπούλου", "F", "genitive", False, "Μαρίας Παπαδοπούλου"),
        ("Ελένη Παπαδοπούλου", "F", "genitive", False, "Ελένης Παπαδοπούλου"),
        # Alone, -η is changed only for a Faker first name that is not also a Faker surname.
        ("Ελένη", "F", "genitive", False, "Ελένης"),
        ("Κυριακή", "F", "genitive", False, "Κυριακής"),
        ("Δάφνη", "F", "genitive", False, "Δάφνη"),
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
        # -ός is far more common than -ούς, so Παππούς is not read back from its genitive.
        ("Παππού", "M", "genitive", True, "Παππός"),
        ("Γιάννη", "M", "genitive", False, "Γιάννης"),
        ("Νίκο Παπαδόπουλο", "M", "accusative", False, "Νίκος Παπαδόπουλος"),
        ("Νίκο", "M", "vocative", False, "Νίκος"),
        ("Ξανθέ", "M", "vocative", True, "Ξανθός"),
        ("Μαλτέζε", "M", "vocative", True, "Μαλτέζος"),
        ("Τζαννετάκο", "M", "vocative", False, "Τζαννετάκος"),
        ("Λαλέ", "M", "genitive", True, "Λαλές"),
        ("Δέδε", "M", "accusative", True, "Δέδες"),
        ("Στεφανάκι", "M", "genitive", True, "Στεφανάκις"),
        ("Παππού", "M", "accusative", True, "Παππούς"),
        ("Κεραμέως", "M", "genitive", True, "Κεραμεύς"),
        ("Παπαδόπουλου", "M", "genitive", True, "Παπαδόπουλος"),
        ("Κυριακής", "F", "genitive", False, "Κυριακή"),
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
