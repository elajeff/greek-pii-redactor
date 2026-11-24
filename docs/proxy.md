# Proxy and command line

## Proxy

`serve` runs a local server that speaks the OpenAI chat completions API. Point a client at it instead of the provider: it redacts every message, forwards the request to `--upstream`, and restores the reply.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[serve,tagger,names]"
.\.venv\Scripts\greek-pii-redactor.exe serve --upstream http://localhost:11434/v1 --model-dir C:\Users\you\weights\xlmr-person --names-dir C:\Users\you\weights\names
```

It listens on `http://127.0.0.1:8080/v1` and passes the `Authorization` header on to the upstream. It doesn't log requests or replies. Without `--model-dir` only IDs are replaced. With `--names-dir`, replies are restored with the name forms from `train/train_inflection.py`, as in the `restore` command below.

What it refuses with a 400, without forwarding anything:

- `"stream": true`, tools and function calls
- any request field outside `model`, `messages`, `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `seed`, `stop`, `n`, `presence_penalty`, `frequency_penalty` and `stream`, or one of these with the wrong type
- message keys other than `role` and `content`, a role other than `system`, `developer`, `user` and `assistant`, a `content` that is null, and parts other than `{"type": "text", "text": ...}`
- a message that already contains a placeholder

Message text and `stop` strings are redacted. `model` is sent as it is.

If the upstream can't be reached it answers 502 with only the error type. Requests are redacted one at a time. `eval/latency.py` measures the time the proxy adds per request against an in-process fake upstream.

## Command line

Installing the package adds a `greek-pii-redactor` command. `redact` prints the file with placeholders and writes the mapping next to it, `restore` puts the originals back into a reply:

```powershell
.\.venv\Scripts\greek-pii-redactor.exe redact letter.txt --model-dir C:\Users\you\weights\xlmr-person > letter.redacted.txt
.\.venv\Scripts\greek-pii-redactor.exe restore reply.txt --mapping letter.txt.mapping.json --names-dir C:\Users\you\weights\names
```

`--model-dir` needs the `tagger` extra (torch, transformers). Without it only IDs are replaced. `--names-dir` takes the name forms built by `train/train_inflection.py` and needs the `names` extra (scikit-learn); without it restore uses the rules in `inflect.py`. Its `models.joblib` is a pickle, which can run code when loaded, so only pass a names dir you built yourself. The mapping file holds the original personal data, so treat it like the original text.
