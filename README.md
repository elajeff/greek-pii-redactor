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

`load_tagger` in `greek_pii_redactor.tagger` loads a fine-tuned XLM-R model that finds person names (see Name tagger below).

`redact` swaps what it finds for placeholders and `restore` puts it back in the LLM's reply:

```python
>>> from greek_pii_redactor.redact import redact, restore
>>> text, mapping = redact("ΑΦΜ 123456783, τηλ. 6912345678, giorgos@example.gr")
>>> text
'ΑΦΜ [AFM_1], τηλ. [PHONE_1], [EMAIL_1]'
>>> restore("Έλαβα: " + text, mapping)
'Έλαβα: ΑΦΜ 123456783, τηλ. 6912345678, giorgos@example.gr'
```

Pass `redact` a name detector (for example the tagger) as its second argument to replace names too. The same person in different Greek cases gets one placeholder (`[PERSON_1]` for Νίκος and Νίκου). On restore, the word before the placeholder (an article, a title like κυρίου, or a greeting like Καλημέρα) sets the case. If the input never had that case, restore builds it with the rules in `inflect.py`: Νίκος Παπαδόπουλος seen, «του [PERSON_1]» gives «του Νίκου Παπαδοπούλου» and «Καλημέρα, [PERSON_1]» gives «Καλημέρα, Νίκο Παπαδόπουλε». Where the rules can't tell (a genitive like Αλεξάνδρου or Κωνσταντίνου doesn't show whether the accent moved: Αλέξανδρος, Κωνσταντίνος), restore uses the first form seen. On 500 seeded records, 5.3% of restored names come out in the wrong case, against 53.5% when restore only picks among the seen forms (`python eval/inflection.py --seed 7 --count 500`). A placeholder that isn't in the mapping (the LLM made it up, or the CLI got the wrong mapping file) comes back without its brackets, as `PERSON_7`. The limits are listed at the top of `redact.py`. Addresses aren't detected yet.

## Command line

Installing the package adds a `greek-pii-redactor` command. `redact` prints the file with placeholders and writes the mapping next to it, `restore` puts the originals back into a reply:

```powershell
.\.venv\Scripts\greek-pii-redactor.exe redact letter.txt --model-dir C:\Users\you\weights\xlmr-person > letter.redacted.txt
.\.venv\Scripts\greek-pii-redactor.exe restore reply.txt --mapping letter.txt.mapping.json --names-dir C:\Users\you\weights\names
```

`--model-dir` needs the `train` extra (torch, transformers). Without it only IDs are replaced. `--names-dir` takes the name forms built by `train/train_inflection.py` and needs the `names` extra (scikit-learn); without it restore uses the rules in `inflect.py`. Its `models.joblib` is a pickle, which can run code when loaded, so only pass a names dir you built yourself. The mapping file holds the original personal data, so treat it like the original text.

## Proxy

`serve` runs a local server that speaks the OpenAI chat completions API. Point a client at it instead of the provider: it redacts every message, forwards the request to `--upstream`, and restores the reply.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[serve,train,names]"
.\.venv\Scripts\greek-pii-redactor.exe serve --upstream http://localhost:11434/v1 --model-dir C:\Users\you\weights\xlmr-person --names-dir C:\Users\you\weights\names
```

It listens on `http://127.0.0.1:8080/v1` and passes the `Authorization` header on to the upstream. It doesn't log requests or replies. With `--names-dir`, replies are restored with the name forms from `train/train_inflection.py`, as in the `restore` command.

What it refuses with a 400, without forwarding anything:

- `"stream": true`, tools and function calls
- any request field outside `model`, `messages`, `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `seed`, `stop`, `n`, `presence_penalty`, `frequency_penalty` and `stream`, or one of these with the wrong type
- message keys other than `role` and `content`, a role other than `system`, `developer`, `user` and `assistant`, a `content` that is null, and parts other than `{"type": "text", "text": ...}`
- a message that already contains a placeholder

Message text and `stop` strings are redacted. `model` is sent as it is.

If the upstream can't be reached it answers 502 with only the error type. Requests are redacted one at a time. `eval/latency.py` measures the time the proxy adds per request against an in-process fake upstream.

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

`train/train_tagger.py` fine-tunes `FacebookAI/xlm-roberta-base` (MIT) to tag person names. It trains on the WikiANN train split plus 2000 synthetic sentences and keeps the epoch with the best F1 on the WikiANN validation split, stopping after 2 epochs without a gain (15 at most). `--seed`, `--without-synthetic` and `--train-fraction` change the seed, drop the synthetic sentences and train on a seeded share of the records; each epoch's numbers go to `epochs.json` in the output folder. It needs a CUDA GPU. The default PyPI torch wheel on Windows is CPU-only, so install the CUDA build first:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
.\.venv\Scripts\python.exe eval/make_synthetic.py --seed 11 --split train --count 2000 --out data/synthetic_train.jsonl
.\.venv\Scripts\python.exe train/train_tagger.py --out C:\Users\you\weights\xlmr-person
```

For example, `.\.venv\Scripts\python.exe train/train_tagger.py --out C:\Users\you\weights\wikiann-only --without-synthetic --train-fraction 0.5 --seed 1` trains on half of WikiANN with seed 1.

The weights stay outside the repo. Pass the folder to the eval to add the tagger to the table:

```powershell
.\.venv\Scripts\python.exe eval/evaluate.py --model-dir C:\Users\you\weights\xlmr-person
```

`download_ner.py` drops WikiANN test sentences that also appear in its train or validation split. Many person names still appear in both, which is how WikiANN is built, so the WikiANN score is for names the model has partly seen.

## Placeholder survival

An LLM can rewrite a placeholder (drop the brackets, translate it) and then restore has nothing to match. `eval/survival.py` sends redacted test sentences to a local [Ollama](https://ollama.com) model, asks for a polite rewrite, and counts how many placeholders come back unchanged. Temperature is 0 and the seed is set, so a run repeats.

```powershell
ollama pull qwen3:4b-instruct
.\.venv\Scripts\python.exe eval/survival.py --model-dir C:\Users\you\weights\xlmr-person
```

Without `--model-dir` only IDs are redacted. On one sentence template the 4B model then wrapped the whole reply in brackets and dropped the placeholder's own brackets, so the ID-only numbers and the with-names numbers are not comparable.

## Run from source

Python 3.13 (64-bit). From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,eval]"
```

`dev` includes the `serve` dependencies, which the proxy tests need.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

## Licence

MIT. See [LICENSE](LICENSE).
