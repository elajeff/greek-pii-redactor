# greek pii redactor

Removes personal data from Greek and English text before it goes to an LLM, and puts it back in the reply. It finds person names, ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers and emails, swaps them for placeholders like `[PERSON_1]`, and on the way back writes each name in the Greek case the reply needs.

Built with Python 3.13, PyTorch and Hugging Face Transformers (a fine-tuned XLM-RoBERTa tags the names), FastAPI for the proxy and scikit-learn for the name forms. I wanted to see whether a small model trained for one language and one job does better than a general tool or a general LLM, so the repo also has the scripts that compare them.

## How it compares

On Greek text: the 1515 Greek sentences of my synthetic test file and 2000 sentences of the WikiANN `el` test split. Names are scored by F1 on exact spans. IDs are the ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers and emails in the synthetic sentences.

| | Names, synthetic (1515) | Names, WikiANN (894) | IDs found (2017) | Time for 2000 WikiANN sentences |
| --- | --- | --- | --- | --- |
| Rules only (`rules.py`) | finds none | finds none | 2017 | under 0.1 s |
| Presidio + spaCy `el_core_news_lg` | 0.818 | 0.644 | 1323, no ΑΦΜ or ΑΜΚΑ | 7 to 21 s |
| qwen3:4b-instruct through Ollama, plus the rules | 0.922 | 0.637 | 2017 | 71 min |
| XLM-R tagger, plus the rules | 0.994 | 0.920 | 2017 | 22 s on the GPU, 59 s on the CPU |

The tagger is better than Presidio and qwen3:4b-instruct at finding Greek names: on WikiANN it finds 833 of the 894 names, against 580 for Presidio and 561 for the LLM, with 83 wrong hits against their 328 and 306. It does the 2000 sentences in 22 s, where the LLM needs 71 minutes on the same GPU. With the rules it also finds every ΑΦΜ and ΑΜΚΑ in the test file, and Presidio finds none.

The accuracy numbers come from one run of `eval/evaluate.py` on my machine (RTX 5070) and the CPU time from a second run without the LLM, with the seeds and commands in [docs/evaluation.md](docs/evaluation.md). The tagger was trained on sentences from the same generator as the synthetic test file (other names and templates), so that column is its best case. The test IDs use the formats the rules look for, so the rules' full score says they match those formats, not how they do on IDs typed by people.

## What it does

- `redact` swaps names and IDs for placeholders and returns the mapping; `restore` puts them back into the LLM's reply.
- ΑΦΜ, ΑΜΚΑ and IBAN only count when their check digit is right, so a mistyped number is left alone.
- The same person in different Greek cases gets one placeholder. On restore, the word before the placeholder sets the case: with Νίκος Παπαδόπουλος in the input, «του [PERSON_1]» comes back as «του Νίκου Παπαδοπούλου».
- `greek-pii-redactor serve` runs a local proxy that speaks the OpenAI chat completions API and does both steps around every request.
- `greek-pii-redactor redact` and `restore` do the same for files.

## Running it

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

The key is passed on to the upstream as the `Authorization` header; Ollama ignores it, a hosted provider needs its real key there. Started like this, the proxy only replaces IDs. To replace names too, [train the tagger](docs/name-tagger.md) (no weights are published) and add `--model-dir`. With it, the upstream gets `Γράψε ένα σύντομο ευχαριστήριο email προς τον [PERSON_1] (ΑΦΜ [AFM_1]).` and the name and ΑΦΜ are put back in the reply.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

## Project layout

| Path | Contents |
| --- | --- |
| `src/greek_pii_redactor/` | The library: ID rules, the tagger, redact and restore, Greek inflection, the CLI and the proxy |
| `train/` | Training the name tagger and the name-form classifier |
| `eval/` | Dataset downloads, the synthetic data generator and the comparison scripts |
| `tests/` | pytest tests |
| `docs/` | Longer docs |

## More docs

- [Evaluation](docs/evaluation.md): full results, commands, test data and licences
- [Name tagger](docs/name-tagger.md): training it and what the synthetic sentences add
- [Redact and restore](docs/restore.md): the Python API and how the Greek cases are restored
- [Proxy and command line](docs/proxy.md)

## Limitations

This is a learning project, not a tested privacy tool. Only names and the five ID kinds are replaced: addresses, ID card numbers, dates of birth and the names the tagger misses go to the upstream as they are. Lowercase and all-caps names are the tagger's weak spots. The LLM can rewrite a placeholder, and restore then has nothing to match; in one run through the proxy without the tagger, qwen3:4b-instruct wrote `ΑΦΜ: AFM_1` and the reply came back with `AFM_1` instead of the number. Restore can put a name in the wrong Greek case, and the known cases are listed at the top of `src/greek_pii_redactor/redact.py`. The proxy has no streaming and no tools, and training the tagger needs a CUDA GPU.

## License

MIT. See [LICENSE](LICENSE).
