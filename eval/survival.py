"""Measures how many placeholders a local Ollama model returns unchanged when it rewrites text.

Takes a seeded sample of synthetic_test.jsonl in --data-dir, default data/ (python
eval/make_synthetic.py --seed 7 --split test --count 2000), redacts each sentence, asks the model
to rewrite it politely, then counts the placeholders still in the reply and the records whose
originals come back through restore. The "restored" column covers all kinds: for the ID kinds the
original string must appear as a whole word; for PERSON any form of the name seen in the input
counts, not the exact case.
The request uses temperature 0 and the --seed value, but Ollama's replies can still differ
slightly from run to run.
With --model-dir the PERSON tagger is used too; without it only IDs are redacted, so PERSON
placeholders are absent from the numbers.

Also reports a round trip of the redacted text itself, which does not depend on the model.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from ollama_chat import ask_model
from records import (
    MODEL_DIR_HELP,
    SYNTHETIC_TEST,
    load_records,
    require_model_dir,
    seeded_sample,
)

from greek_pii_redactor.redact import KINDS, redact, restore

# No example token here: a model given one copies it as its whole answer.
INSTRUCTION = (
    "Rewrite the user's message politely, in the same language as the message. "
    "Copy every token in square brackets exactly, character for character. "
    "Reply with the rewritten message only."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, help=MODEL_DIR_HELP)
    parser.add_argument("--url", default="http://localhost:11434")
    args = parser.parse_args()
    file_name, command = SYNTHETIC_TEST
    records = load_records(args.data_dir, file_name, command)

    detect_names = None
    if args.model_dir:
        require_model_dir(args.model_dir)
        # Imported here so the script runs without torch and transformers installed.
        from greek_pii_redactor.tagger import load_tagger

        detect_names = load_tagger(args.model_dir)

    sample = seeded_sample(records, args.count, args.seed, args.data_dir / file_name)

    sent: Counter[str] = Counter()
    unchanged: Counter[str] = Counter()
    records_with: Counter[str] = Counter()
    records_restored: Counter[str] = Counter()
    round_trips = 0
    for record in sample:
        redacted, placeholders = redact(record["text"], detect_names)
        round_trips += restore(redacted, placeholders) == record["text"]
        reply = ask_model(args.url, args.model, args.seed, INSTRUCTION, redacted)
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
