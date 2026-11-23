# greek pii redactor

Removes personal data from Greek and English text before it goes to an LLM, and puts it back in the reply. It finds person names, ΑΦΜ, ΑΜΚΑ, IBAN, phone numbers and emails, swaps them for placeholders like `[PERSON_1]`, and on the way back writes each name in the Greek case the reply needs.

Built with Python 3.13, PyTorch and Hugging Face Transformers (a fine-tuned XLM-RoBERTa tags the names), FastAPI for the proxy and scikit-learn for the name forms. I wanted to see whether a small model trained for one language and one job does better than a general tool or a general LLM, so the repo also has the scripts that compare them.

## How it compares

I ran four detectors on Greek text only: the rules alone, Presidio with the Greek spaCy model, a small local LLM (qwen3:4b-instruct through Ollama), and my tagger. The text is the 1515 Greek sentences of my synthetic test file (1515 names, 502 ΑΦΜ and ΑΜΚΑ numbers) and 2000 real sentences from WikiANN `el` (894 names). A name counts as found only when the detector marks exactly the right span.

| | Names found, synthetic | Names found, WikiANN | Wrong name hits, WikiANN | ΑΦΜ and ΑΜΚΑ found | Time for 2000 WikiANN sentences |
| --- | --- | --- | --- | --- | --- |
| Rules only | 0 of 1515 | 0 of 894 | 0 | 502 of 502 | under 0.1 s |
| Presidio + spaCy `el_core_news_lg` | 1058 (70%) | 580 (65%) | 328 | 0 of 502 | 7 to 21 s |
| qwen3:4b-instruct + rules | 1317 (87%) | 561 (63%) | 306 | 502 of 502 | 71 min |
| **XLM-R tagger + rules** | **1496 (99%)** | **833 (93%)** | **83** | 502 of 502 | 22 s on the GPU, 59 s on the CPU |

What I take from it:

- **Names:** the tagger finds the most names in both sets and makes the fewest wrong hits. On real WikiANN text it finds 93% of the names. Presidio finds 65% and the LLM 63%, each with about four times as many wrong hits.
- **Speed:** the tagger does 2000 sentences in 22 s. The LLM needs 71 minutes on the same GPU.
- **IDs:** Presidio finds none of the ΑΦΜ or ΑΜΚΑ numbers. The rules find all of them by checking the check digit. Presidio also misses a quarter of the phone numbers.

What the table doesn't show:

- The synthetic column is the tagger's best case. It trained on sentences from the same generator, with other names and templates. WikiANN is the fairer test.
- The test IDs use the formats the rules look for, so their full score says they match those formats, not how they do on IDs typed by people.
- Presidio's time was 7 s in one run and 21 s in another, and I haven't found why. Treat every time as rough.

The numbers come from `eval/evaluate.py` on my machine (RTX 5070), plus a second run without the LLM for the CPU time. Precision, recall, F1 and the commands are in [docs/evaluation.md](docs/evaluation.md).

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
