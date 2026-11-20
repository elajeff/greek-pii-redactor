# greek pii redactor

Removes personal data from Greek and English text before it goes to an LLM, and puts it back in the reply. The plan is a Python library plus a local proxy that speaks the OpenAI chat API, measured against Presidio and a rules-only baseline.

## What it does so far

`find_ids` finds Greek ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers and emails in a string and returns their positions:

```python
>>> from greek_pii_redactor.rules import find_ids
>>> find_ids("ΑΦΜ 123456783, τηλ. 6912345678, giorgos@example.gr")
[Span(start=4, end=13, kind='AFM'), Span(start=20, end=30, kind='PHONE'), Span(start=32, end=50, kind='EMAIL')]
```

ΑΦΜ, ΑΜΚΑ and IBAN are only reported when their check digit is right, so a mistyped number is left alone. Phone numbers have no check digit, so any 10-digit number with the right start (69 or 2, optional +30 or 0030) counts.

`load_tagger` in `greek_pii_redactor.tagger` loads a fine-tuned XLM-R model that finds person names (see Name tagger below). Addresses, the placeholder and restore step and the proxy aren't built yet.

## Test data

`eval/make_synthetic.py` writes invented Greek and English sentences with labelled names and IDs. The same seed gives the same file, and train and test use different names and sentence templates:

```powershell
.\.venv\Scripts\python.exe eval/make_synthetic.py --seed 7 --split test --count 2000 --out data/synthetic_test.jsonl
```

`eval/download_ner.py` downloads real Greek text with person labels into `data/`. Nothing from these datasets is stored in the repo.

| Dataset | Licence on its Hugging Face page |
| --- | --- |
| [unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann) (`el`) | unknown; labels are machine-generated from Wikipedia |
| [joelniklaus/greek_legal_ner](https://huggingface.co/datasets/joelniklaus/greek_legal_ner) | CC BY-NC-SA 4.0 |

## Baselines

`eval/evaluate.py` scores exact-match precision and recall per kind for the rules alone and for Presidio with the Greek spaCy model. It needs the data above and the model:

```powershell
.\.venv\Scripts\python.exe -m spacy download el_core_news_lg
.\.venv\Scripts\python.exe eval/evaluate.py
```

`el_core_news_lg` is CC BY-NC-SA 3.0, so it is only used here for comparison and isn't part of the package.

## Name tagger

`train/train_tagger.py` fine-tunes `FacebookAI/xlm-roberta-base` (MIT) to tag person names. It trains on the WikiANN train split plus 2000 synthetic sentences and keeps the epoch with the best F1 on the WikiANN validation split. It needs a CUDA GPU. The default PyPI torch wheel on Windows is CPU-only, so install the CUDA build first:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
.\.venv\Scripts\python.exe eval/make_synthetic.py --seed 11 --split train --count 2000 --out data/synthetic_train.jsonl
.\.venv\Scripts\python.exe train/train_tagger.py --out C:\Users\you\weights\xlmr-person
```

The weights stay outside the repo. Pass the folder to the eval to add the tagger to the table:

```powershell
.\.venv\Scripts\python.exe eval/evaluate.py --model-dir C:\Users\you\weights\xlmr-person
```

`download_ner.py` drops WikiANN test sentences that also appear in its train or validation split. Many person names still appear in both, which is how WikiANN is built, so the WikiANN score is for names the model has partly seen.

## Run from source

Python 3.13 (64-bit). From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,eval]"
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

## Licence

MIT. See [LICENSE](LICENSE).
