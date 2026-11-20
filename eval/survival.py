"""Measures how many placeholders a local Ollama model returns unchanged when it rewrites text.

Takes a seeded sample of data/synthetic_test.jsonl (python eval/make_synthetic.py --seed 7
--split test --count 2000), redacts each sentence, asks the model to rewrite it politely, then
counts the placeholders still in the reply and the records whose originals come back through
restore. The "restored" column covers all kinds: for the ID kinds the original string must appear
as a whole word; for PERSON any form of the name seen in the input counts, not the exact case.
The request uses temperature 0 and the --seed value, so a run can be repeated.
With --model-dir the PERSON tagger is used too; without it only IDs are redacted, so PERSON
placeholders are absent from the numbers.

Also reports a round trip of the redacted text itself, which does not depend on the model.
"""

import argparse
import json
import random
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from greek_pii_redactor.redact import KINDS, redact, restore

DATA_FILE = Path("data/synthetic_test.jsonl")
# No example token here: a model given one copies it as its whole answer.
INSTRUCTION = (
    "Rewrite the user's message politely, in the same language as the message. "
    "Copy every token in square brackets exactly, character for character. "
    "Reply with the rewritten message only."
)
# Stops a hung server from blocking the run forever.
REQUEST_TIMEOUT_SECONDS = 120


def ask_model(url: str, model: str, seed: int, message: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": seed},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/api/chat", body, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw_reply = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode()
        # Ollama answers 404 for an unknown URL too, so the body has to name the model.
        if error.code == 404 and "model" in detail and "not found" in detail:
            raise SystemExit(
                f"Model {model} not found on {url}. Pull it: ollama pull {model}"
            ) from error
        raise SystemExit(
            f"Ollama at {url} returned HTTP {error.code}: {detail}"
        ) from error
    except OSError as error:
        raise SystemExit(f"Cannot reach Ollama at {url}: {error}") from error

    unexpected = f"Ollama at {url} sent an unexpected reply"
    try:
        content = json.loads(raw_reply)["message"]["content"]
    except (ValueError, KeyError, TypeError) as error:
        # Not repr(error): for a body that isn't UTF-8 it holds the whole body.
        raise SystemExit(f"{unexpected} ({type(error).__name__}: {error})") from error
    if not isinstance(content, str):
        raise SystemExit(f"{unexpected} (content is not a string)")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, help="weights from train/train_tagger.py")
    parser.add_argument("--url", default="http://localhost:11434")
    args = parser.parse_args()
    if not DATA_FILE.exists():
        raise SystemExit(
            f"{DATA_FILE} not found. Create it with: "
            "python eval/make_synthetic.py --seed 7 --split test --count 2000"
        )
    if args.model_dir and not args.model_dir.is_dir():
        raise SystemExit(f"{args.model_dir} not found. Train it with: python train/train_tagger.py")

    detect_names = None
    if args.model_dir:
        # Imported here so the script runs without torch and transformers installed.
        from greek_pii_redactor.tagger import load_tagger

        detect_names = load_tagger(args.model_dir)

    with DATA_FILE.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    if args.count > len(records):
        raise SystemExit(f"--count {args.count} exceeds the {len(records)} records in {DATA_FILE}")
    sample = random.Random(args.seed).sample(records, args.count)

    sent: Counter[str] = Counter()
    unchanged: Counter[str] = Counter()
    records_with: Counter[str] = Counter()
    records_restored: Counter[str] = Counter()
    round_trips = 0
    for record in sample:
        redacted, placeholders = redact(record["text"], detect_names)
        round_trips += restore(redacted, placeholders) == record["text"]
        reply = ask_model(args.url, args.model, args.seed, redacted)
        restored = restore(reply, placeholders)
        for kind in KINDS:
            of_kind = {p: forms for p, forms in placeholders.items() if p.startswith(f"[{kind}_")}
            if not of_kind:
                continue
            sent[kind] += len(of_kind)
            unchanged[kind] += sum(placeholder in reply for placeholder in of_kind)
            records_with[kind] += 1
            # Any seen form counts: restore picks the one that fits the reply, not always the first.
            records_restored[kind] += all(
                any(
                    re.search(rf"(?<!\w){re.escape(form['text'])}(?!\w)", restored)
                    for form in forms
                )
                for forms in of_kind.values()
            )

    tagger = "PERSON tagger from --model-dir" if args.model_dir else "IDs only (no --model-dir)"
    print(f"model {args.model}, seed {args.seed}, {args.count} sentences, redaction: {tagger}")
    print(f"Round trip of the redacted text (no model): {round_trips} of {args.count}")
    print(f"{'kind':<8}{'sent':>6}{'unchanged':>11}{'share':>8}{'records':>9}{'restored':>10}")
    for kind in KINDS:
        if sent[kind]:
            print(
                f"{kind:<8}{sent[kind]:>6}{unchanged[kind]:>11}{unchanged[kind] / sent[kind]:>8.3f}"
                f"{records_with[kind]:>9}{records_restored[kind] / records_with[kind]:>10.3f}"
            )
    print("restored: PERSON counts any seen form of the name, not the exact case.")
    if not args.model_dir:
        print("No PERSON rows: names were not redacted.")


if __name__ == "__main__":
    main()
