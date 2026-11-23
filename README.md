# greek pii redactor

Removes personal data from Greek and English text before it goes to an LLM, and puts it back in the reply. It finds person names, ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers and emails and swaps them for placeholders like `[PERSON_1]`. Greek names change with the case, so on the way back it writes each name in the case the reply needs.

Built with Python 3.13, PyTorch and Hugging Face Transformers (a fine-tuned XLM-RoBERTa finds the names), FastAPI for the proxy and scikit-learn for the name forms. I started it to see whether a small model trained for one language and one task could beat a general tool and a general LLM at it. The comparison is below.

## How it compares

I compared the name detection with Presidio (using its Greek spaCy model) and with a local LLM, qwen3:4b-instruct run through Ollama and asked to list the names in each sentence. All three only saw Greek text: 1515 Greek sentences from my synthetic test file, and 2000 sentences from the Greek part of WikiANN, which is real Wikipedia text. A hit only counts if the detector marks exactly the right span. A wrong hit is a span that isn't a labelled name.

| | Names found, synthetic (of 1515) | Names found, WikiANN (of 894) | Wrong hits, WikiANN | Time for 2000 WikiANN sentences |
| --- | --- | --- | --- | --- |
| Presidio | 1058 | 580 | 328 | 8 s |
| qwen3:4b-instruct | 1317 | 561 | 306 | 71 min |
| greek-pii-redactor | 1496 | 833 | 83 | 14 s on the GPU, 74 s on the CPU |

greek-pii-redactor finds more names than both and makes far fewer wrong hits. On WikiANN it finds 833 of 894 names (93%), Presidio 580 and the LLM 561. It takes a few seconds longer than Presidio, and 14 seconds against the LLM's 71 minutes.

The synthetic column flatters it: I trained on sentences from the same generator (other names and templates), so WikiANN is the number to trust. The times are averages of three runs on my RTX 5070, except the LLM, which I ran once. Precision, recall and the commands are in [docs/evaluation.md](docs/evaluation.md).

## What it does

- `redact` swaps names and IDs for placeholders and returns the mapping; `restore` puts them back into the LLM's reply.
- IDs (ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers, emails) are found by rules, and names by the tagger. ΑΦΜ, ΑΜΚΑ and IBAN only count when their check digit is right, so a mistyped number is left alone.
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
