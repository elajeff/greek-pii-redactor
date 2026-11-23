# Redact and restore

`find_ids` finds Greek ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers and emails in a string and returns their positions:

```python
>>> from greek_pii_redactor.rules import find_ids
>>> find_ids("ΑΦΜ 123456783, τηλ. 6912345678, giorgos@example.gr")
[Span(start=4, end=13, kind='AFM'), Span(start=20, end=30, kind='PHONE'), Span(start=32, end=50, kind='EMAIL')]
```

ΑΦΜ, ΑΜΚΑ and IBAN are only reported when their check digit is right, so a mistyped number is left alone. Phone numbers have no check digit, so any 10-digit number with the right start (69 or 2, optional +30 or 0030) counts.

`redact` swaps what it finds for placeholders and `restore` puts it back in the LLM's reply:

```python
>>> from greek_pii_redactor.redact import redact, restore
>>> text, mapping = redact("ΑΦΜ 123456783, τηλ. 6912345678, giorgos@example.gr")
>>> text
'ΑΦΜ [AFM_1], τηλ. [PHONE_1], [EMAIL_1]'
>>> restore("Έλαβα: " + text, mapping)
'Έλαβα: ΑΦΜ 123456783, τηλ. 6912345678, giorgos@example.gr'
```

Pass `redact` a name detector as its second argument to replace names too, for example the tagger from `load_tagger` in `greek_pii_redactor.tagger` (see [the name tagger](name-tagger.md)). The same person in different Greek cases gets one placeholder (`[PERSON_1]` for Νίκος and Νίκου).

## Greek cases

On restore, the word before the placeholder (an article, a title like κυρίου, or a greeting like Καλημέρα) sets the case. If the input never had that case, restore builds it with the rules in `inflect.py`: Νίκος Παπαδόπουλος seen, «του [PERSON_1]» gives «του Νίκου Παπαδοπούλου» and «Καλημέρα, [PERSON_1]» gives «Καλημέρα, Νίκο Παπαδόπουλε». Where the rules can't tell (a genitive like Αλεξάνδρου or Κωνσταντίνου doesn't show whether the accent moved: Αλέξανδρος, Κωνσταντίνος), restore uses the first form seen.

With `--names-dir` (in code, `restore(reply, mapping, load_names(names_dir))` from `names.py`), the forms come from the case tables in Greek Wiktionary. For a name not in them they come from a classifier trained on those tables by `train/train_inflection.py`, or from the rules where the classifier is unsure. `eval/download_names.py` downloads the tables first.

`python eval/inflection.py --seed 7 --count 500` restores 1242 names, each seen once in the input in one case and asked for in the reply in another. 5.3% come out in the wrong case, against 53.5% when restore only picks among the forms seen in the input. The expected forms come from the same rules restore uses, so this measures choosing and building the case, not whether the rules' Greek is right; `eval/wiktionary.py` checks the rules against Wiktionary's tables.

A placeholder that isn't in the mapping (the LLM made it up, or the CLI got the wrong mapping file) comes back without its brackets, as `PERSON_7`. The known cases where restore gets the Greek wrong are listed at the top of `src/greek_pii_redactor/redact.py`.
