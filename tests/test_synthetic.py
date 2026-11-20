import pytest
from eval.make_synthetic import generate

from greek_pii_redactor.rules import Span, find_ids

ID_KINDS = {"AFM", "AMKA", "IBAN", "PHONE", "EMAIL"}


@pytest.mark.parametrize("split", ["train", "test"])
def test_labelled_ids_are_found_at_the_labelled_offsets(split: str) -> None:
    records = generate(seed=7, split=split, count=200)
    for record in records:
        labelled = [Span(*span) for span in record["spans"] if span[2] in ID_KINDS]
        assert find_ids(record["text"]) == labelled
    assert {span[2] for record in records for span in record["spans"]} >= ID_KINDS
