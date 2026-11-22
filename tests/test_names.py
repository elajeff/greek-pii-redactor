import pytest

from greek_pii_redactor.names import apply_edit, derive_edit


@pytest.mark.parametrize(
    "source, target, first_name, edit",
    [
        ("Θεόδωρος", "Θεοδώρου", True, (1, "υ", 1)),
        ("Θεοδώρου", "Θεόδωρος", True, (1, "ς", -1)),
        ("Παππούς", "Παππού", False, (1, "", 0)),
        ("Κρέων", "Κρέοντα", True, (2, "οντα", 0)),
        ("Νίκος", "Νίκο", True, (1, "", 0)),
        ("Ξανθός", "Ξανθού", False, (2, "ού", 0)),
        ("Παΐσιος", "Παϊσίου", True, (1, "υ", 1)),
        ("Βαΐου", "Βάιος", True, (1, "ς", -1)),
    ],
)
def test_edit_gives_the_form_back(
    source: str, target: str, first_name: bool, edit: tuple[int, str, int]
) -> None:
    assert derive_edit(source, target, first_name) == edit
    assert apply_edit(source, edit, first_name) == target
