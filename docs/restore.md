# Redact and restore

`find_ids` finds Greek ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers and emails in a string and returns their positions:

```python
>>> from greek_pii_redactor.rules import find_ids
>>> find_ids("ΑΦΜ 123456783, τηλ. 6912345678, giorgos@example.gr")
[Span(start=4, end=13, kind='AFM'), Span(start=20, end=30, kind='PHONE'), Span(start=32, end=50, kind='EMAIL')]
```

ΑΦΜ, ΑΜΚΑ and IBAN are only reported when their check digit is right, so a mistyped number is left alone. Phone numbers have no check digit, so any 10-digit number with the right start (69 or 2, optional +30 or 0030) counts. Digits from other scripts, such as fullwidth `６９` or Arabic-Indic `٦٩`, are read as 0-9.

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

A placeholder that isn't in the mapping (the LLM made it up, or the CLI got the wrong mapping file) comes back without its brackets, as `PERSON_7`. The known cases where restore gets the Greek wrong are listed below.

## Known limits

### Without a names dir

- Restore builds the case a reply needs from the forms seen in the input and the rules in `inflect.py`, so the Greek can still be wrong where the rules can't tell:
  - A seen genitive in -ου gives the nominative only for a 2-syllable first name (Νίκου), -ίου (Γεωργίου), -αίου (Ματθαίου) and -όπουλου/-οπούλου, so Κωνσταντίνου, Αλεξάνδρου and other surnames in -ου are not derived from it.
  - The folk first names Σπήλιος and Στέλλιος are read as learned.
  - A lone female name in -η, -ή or -ά that Faker doesn't list as a first name only is left as it is.
  - Surname-first order (Παπαδοπούλου Μαρία).
  - Latin-script names have no rules, so they use the seen forms. Greek-script foreign names are inflected like Greek ones (Τόμας Μίλερ gives Τόμα Μίλερ).
  - When nothing can be derived, restore uses the first form seen.
- A first name of 4+ syllables in paroxytone -ος gets the vocative -ε (Κωνσταντίνε). For some (Λεονάρδο/Λεονάρδε, Βενιζέλο/Βενιζέλε) both are in use and -ε is chosen.
- When the genitive moves the accent onto ι or υ after a vowel, the diaeresis is lost: Βάιος gives Βαίου (Βαΐου is right).

### With a names dir

Each token's forms come from the Wiktionary lexicon, or for a name not in it from the classifier, and from the rules where the classifier is less than `FORWARD_THRESHOLD` (in `names.py`) sure. Seen forms the lexicon lists are read back from it: «του Αλεξάνδρου» gives «Ο Αλέξανδρος», Σπήλιος gives «του Σπήλιου», Βενιζέλος «Καλημέρα, Βενιζέλο» and Κρέων «του Κρέοντα». A male surname in -ου that Faker lists as one stays as it is in the genitive: «του κυρίου Σακελλαρίου» gives «Ο κύριος Σακελλαρίου».

A form not in the lexicon is read back by the rules when they give a known name, one in the lexicon or in Faker's el_GR lists («κύριε Γιαννακέα» gives «Ο κύριος Γιαννακέας»). Otherwise the classifier reads it back when it is at least `READ_BACK_THRESHOLD` (in `names.py`) sure, and the rules when it is less sure («κύριε Ζβορκέα» gives «Ο κύριος Ζβορκέας»). A name from the classifier that is not known gives way to a known name spelt the same without accents whose form by the rules is the one seen: «του Ανδρονίκου» gives «Ο Ανδρόνικος», not Ανδρονίκος.

- A surname in -ός that is not in the lexicon, where its -ούς twin is, has its genitive -ού read back as the twin. With the `--holdout 0.1 --seed 7` build, Ντουρουντού gives Ντουρουντούς; the `--holdout 0` build lists both, so the rules' Ντουρουντός is used.
- A male surname in -ου that neither the lexicon nor Faker knows, and that the rules can't read back, can be given a misspelt nominative by the classifier when it is at least `READ_BACK_THRESHOLD` sure. With the `--holdout 0.1 --seed 7` build, «του κυρίου Χατζηκοκκίνου» gives «Ο κύριος Χατζηκοκκίνος»; the `--holdout 0` build lists the name, so it gives «Ο κύριος Χατζηκόκκινος».

### Either way

- Two forms of one person with no article before them (a bare nominative and a bare genitive) can't be told apart in the reply, so `restore` of the redacted text may not give the input back exactly.
- A word before the placeholder that is not an article, title or greeting, or words between them (του κ. [PERSON_1]), make restore use the first form seen. A number counts: «Ήρθαν 3 Νίκος Παπαδόπουλος.» stores the article "3" and gives no nominative.
- A name seen after punctuation with no greeting (often a vocative: «Ευχαριστώ, Νίκο Παπαδόπουλε») counts as a nominative and wins over a seen genitive: with «Το σπίτι του Νίκου Παπαδοπούλου.» also seen, «Ο [PERSON_1]» restores as «Ο Νίκο Παπαδόπουλε». After a greeting («Καλημέρα, Νίκο») the case is read as a vocative.
- A man and a woman can still merge when the tokens are identical (Κώστα), and surnames spelled differently (Παπαδόπουλος, Παπαδόπουλλος) are separate people. The rare genitive Γιάννου is not matched with Γιάννης.

How restore reads a name depends on how it was written (the lists are Faker's el_GR names):

- Lowercase with an accent: inflected from it («Ο Νίκος», «του [PERSON_1]» gives «του Νίκου»).
- Capitals, with or without an accent: inflected as lowercase and written back in capitals, with an accent only where one was seen («Ο ΝΊΚΟΣ» gives «ΝΊΚΟΥ», «Ο ΝΙΚΟΣ» gives «ΝΙΚΟΥ»).
- No accent and in the lists: the accent is taken from the list but not written (the vocative of «Νικος» is «Νικο»). A name listed with two accents that give different text (Ιωνα: «Ίωνα» or «Ιωνά») stays as seen.
- No accent and not in the lists: tried with the accent on each of the last three syllables, used only if all agree. Where they don't, a first name's -ος vocative is -ο («Ζβορκος» gives «Ζβορκο», not «Ζβορκε»; without a names dir «Παμφιλος» gives «Παμφιλο», where «Πάμφιλε» is right). Without a names dir a surname's -ος vocative is -ε wherever the accent is (-άκος aside), so all readings agree («κύριε Ζβορκε», «Νίκο Ζβορκε»); with a names dir these come out the same. Otherwise the name is inflected as written, with no accent («Ζβορκια» and «της» give «Ζβορκιας»).
- Capitals mixed with lowercase, other than a leading capital (ΝΙΚΟς): the name stays as seen.
