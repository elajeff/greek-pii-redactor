# greek pii redactor

Removes personal data from Greek and English text before it goes to an LLM, and puts it back in the reply. It is a Python library plus a local proxy that speaks the OpenAI chat API, measured against Presidio, a rules-only baseline and a small local LLM.

## Quick start

Python 3.13 (64-bit), from the repository root in PowerShell, with [Ollama](https://ollama.com) as the upstream:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[serve]"
ollama pull qwen3:4b-instruct
.\.venv\Scripts\greek-pii-redactor.exe serve --upstream http://localhost:11434/v1
```

Then point an OpenAI client at the proxy instead of the provider:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen3:4b-instruct",
    messages=[{"role": "user", "content": "Γράψε ένα σύντομο ευχαριστήριο email προς τον Νίκο Παπαδόπουλο (ΑΦΜ 123456783)."}],
)
print(reply.choices[0].message.content)
```

The key is passed on to the upstream as the `Authorization` header; Ollama ignores it, a hosted provider needs its real key there. Started like this, the proxy only replaces IDs. To replace names too, train the tagger (see Name tagger below; no weights are published) and add `--model-dir`. With it, the upstream gets `Γράψε ένα σύντομο ευχαριστήριο email προς τον [PERSON_1] (ΑΦΜ [AFM_1]).` and the name and ΑΦΜ are put back in the reply.

## What it does

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

Pass `redact` a name detector (for example the tagger) as its second argument to replace names too. The same person in different Greek cases gets one placeholder (`[PERSON_1]` for Νίκος and Νίκου). On restore, the word before the placeholder (an article, a title like κυρίου, or a greeting like Καλημέρα) sets the case. If the input never had that case, restore builds it with the rules in `inflect.py`: Νίκος Παπαδόπουλος seen, «του [PERSON_1]» gives «του Νίκου Παπαδοπούλου» and «Καλημέρα, [PERSON_1]» gives «Καλημέρα, Νίκο Παπαδόπουλε». Where the rules can't tell (a genitive like Αλεξάνδρου or Κωνσταντίνου doesn't show whether the accent moved: Αλέξανδρος, Κωνσταντίνος), restore uses the first form seen. A placeholder that isn't in the mapping (the LLM made it up, or the CLI got the wrong mapping file) comes back without its brackets, as `PERSON_7`.

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

## Results

Exact-span precision and recall: a span counts only if start, end and kind all match. Every number in this section comes from the script named with it, run on my machine (RTX 5070). The tagger is the one trained with `train/train_tagger.py --seed 1` on WikiANN plus the synthetic train file (`eval/make_synthetic.py --seed 11 --split train --count 2000`).

### Names

`python eval/evaluate.py --model-dir <tagger> --ollama-model qwen3:4b-instruct`, on the synthetic test file (`eval/make_synthetic.py --seed 7 --split test --count 2000`) and on 2000 WikiANN `el` test sentences sampled with seed 7. The LLM row asks the model to list the names in each sentence (seed 7, temperature 0) and counts every place a listed name appears in the text.

| PERSON, precision / recall | Synthetic, Greek sentences (1515 names) | Synthetic, English sentences (485 names) | WikiANN el (894 names) |
| --- | --- | --- | --- |
| Presidio + spaCy `el_core_news_lg` | 0.986 / 0.698 | 0.609 / 0.225 | 0.639 / 0.649 |
| XLM-R tagger | 1.000 / 0.987 | 1.000 / 1.000 | 0.909 / 0.932 |
| qwen3:4b-instruct through Ollama | 0.983 / 0.866 | 0.998 / 0.998 | 0.640 / 0.619 |

The rules alone find no names. On the 2000 synthetic sentences the tagger took 24.7 s and the LLM 4334 s.

The synthetic sentences come from my own templates, and the tagger was trained on sentences from the same generator (other names and other templates), so its synthetic score is a best case. WikiANN shares many names between its splits, so the WikiANN score is for names the tagger has partly seen. Ollama's replies aren't exactly repeatable even at temperature 0 with a fixed seed: between two runs the LLM row's true positives differed by 5 on the synthetic file and by 8 on WikiANN.

### IDs

Same run, synthetic test file:

| Kind | Rules | Presidio + spaCy `el_core_news_lg` |
| --- | --- | --- |
| ΑΦΜ (487) | 1.000 / 1.000 | finds none |
| ΑΜΚΑ (500) | 1.000 / 1.000 | finds none |
| IBAN (713) | 1.000 / 1.000 | 1.000 / 1.000 |
| Phone (1024) | 1.000 / 1.000 | 0.756 / 0.815 |
| Email (493) | 1.000 / 1.000 | 1.000 / 1.000 |

The test IDs are written in the formats I generate, which are the formats the rules look for, so the rules' perfect score shows they match those formats, not how they do on IDs typed by people.

### What the synthetic sentences add to the tagger

`python eval/tagger_errors.py --model-dir <seed 1> <seed 2> <seed 3>`, once for taggers trained with `train/train_tagger.py --seed N` and once for `--without-synthetic --seed N`. Mean over the three seeds, standard deviation in brackets. The probes edit the Greek names in the synthetic test file: lowercase the name, keep only the surname of a two-word name, or drop the article before it.

| | WikiANN + synthetic | WikiANN only |
| --- | --- | --- |
| WikiANN F1 | 0.914 (0.005) | 0.915 (0.003) |
| Recall, lowercased name | 0.769 (0.043) | 0.327 (0.036) |
| Recall, surname alone | 0.956 (0.043) | 0.493 (0.105) |
| Recall, no article | 1.000 (0.001) | 0.824 (0.021) |
| greek_legal_ner test, PERSON words inside one predicted span | 0.769 (0.051) | 0.723 (0.079) |
| Same, all-caps words only | 0.728 (0.063) | 0.696 (0.110) |

The synthetic sentences don't change the WikiANN score and teach the tagger lowercase names and surnames on their own. The probes are edits of the generator's own sentences, so they likely favour the tagger trained on the generator's sentences.

### Restoring the case

`python eval/inflection.py --seed 7 --count 500`: 1242 names, each seen once in the input in one case and asked for in the reply in another. 5.3% come out in the wrong case, against 53.5% when restore only picks among the forms seen in the input. The expected forms come from the same rules restore uses, so this measures choosing and building the case, not whether the rules' Greek is right.

## Limits

- Only names and the five ID kinds above are replaced. Addresses, ID card numbers, dates of birth and anything else go to the upstream as they are, and so do the names the tagger misses.
- Lowercase and all-caps names are the tagger's weak spots (see the table above). In greek_legal_ner, where most PERSON words are in capitals, about a quarter of the all-caps PERSON words aren't inside one predicted span.
- The LLM can rewrite a placeholder, and restore then has nothing to match. In one run through the proxy without the tagger, qwen3:4b-instruct wrote `ΑΦΜ: AFM_1`, and the reply came back with `AFM_1` instead of the number.
- Restore can put a name in the wrong Greek case. The known cases are listed at the top of `src/greek_pii_redactor/redact.py`.
- The proxy has no streaming and no tools, and redacts one request at a time.
- No weights are published. Training the tagger needs a CUDA GPU.

## Test data

`eval/make_synthetic.py` writes invented Greek and English sentences with labelled names and IDs, about three quarters of them Greek. The same seed gives the same file, and train and test use different names and sentence templates:

```powershell
.\.venv\Scripts\python.exe eval/make_synthetic.py --seed 7 --split test --count 2000 --out data/synthetic_test.jsonl
```

`eval/download_ner.py` downloads real Greek text with person labels into `data/`, and `eval/download_names.py` the Wiktionary case tables of Greek names. Nothing from these datasets is stored in the repo, and no weights are either.

| Data or model | Used for | Licence on its own page |
| --- | --- | --- |
| [unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann) (`el`) | training and testing | unknown; labels are machine-generated from Wikipedia |
| [joelniklaus/greek_legal_ner](https://huggingface.co/datasets/joelniklaus/greek_legal_ner) | testing the tagger (`eval/tagger_errors.py`) | CC BY-NC-SA 4.0 |
| Greek Wiktionary, through the [kaikki.org](https://kaikki.org/elwiktionary/rawdata.html) extract | the name forms in `--names-dir`, and checking the inflection rules | Wiktionary text is CC BY-SA 4.0; the kaikki.org page names no other licence |
| [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | base model of the tagger | MIT |
| [el_core_news_lg](https://spacy.io/models/el#el_core_news_lg) | the Presidio baseline | CC BY-NC-SA 3.0 |
| [qwen3:4b-instruct](https://ollama.com/library/qwen3) | the LLM baseline and placeholder survival | Apache 2.0 |

## Baselines

`eval/evaluate.py` scores the rules alone and Presidio with the Greek spaCy model, plus the tagger with `--model-dir` and the LLM with `--ollama-model`. It needs the data above and the spaCy model:

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

An LLM can rewrite a placeholder (drop the brackets, translate it) and then restore has nothing to match. `eval/survival.py` sends redacted test sentences to a local [Ollama](https://ollama.com) model, asks for a polite rewrite, and counts how many placeholders come back unchanged. Temperature is 0 and the seed is set, but Ollama's replies can still differ slightly from run to run.

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
