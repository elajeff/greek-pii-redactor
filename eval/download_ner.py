"""Downloads the Greek WikiANN NER dataset from Hugging Face and keeps only the PERSON spans.

Licence, as stated on the dataset's Hugging Face page (unimelb-nlp/wikiann): "unknown". The labels
are machine-generated from Wikipedia.

Output is one JSON object per line: {"text": ..., "spans": [[start, end, "PERSON"], ...]}
with character offsets, end exclusive. Tokens are joined with single spaces.
WikiANN test sentences that also occur in its train or validation split are dropped.
"""

import argparse
import io
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SPLITS = ("train", "validation", "test")
WIKIANN_PERSON_BEGIN, WIKIANN_PERSON_INSIDE = 1, 2  # B-PER, I-PER in the dataset's tag list


def download(repo: str, path: str) -> bytes:
    with urlopen(HF_RESOLVE.format(repo=repo, path=path)) as response:
        return response.read()


def to_record(tokens: list[str], person_marks: list[str]) -> dict:
    """person_marks holds "B", "I" or "O" for each token."""
    text = ""
    spans: list[list] = []
    previous = "O"
    for token, mark in zip(tokens, person_marks, strict=True):
        if text:
            text += " "
        start = len(text)
        text += token
        if mark == "B" or (mark == "I" and previous == "O"):
            spans.append([start, len(text), "PERSON"])
        elif mark == "I":
            spans[-1][1] = len(text)
        previous = mark
    return {"text": text, "spans": spans}


def wikiann_records(split: str) -> list[dict]:
    parquet = download("unimelb-nlp/wikiann", f"el/{split}-00000-of-00001.parquet")
    table = pq.read_table(io.BytesIO(parquet))
    records = []
    for row in table.to_pylist():
        marks = [
            "B" if tag == WIKIANN_PERSON_BEGIN else "I" if tag == WIKIANN_PERSON_INSIDE else "O"
            for tag in row["ner_tags"]
        ]
        records.append(to_record(row["tokens"], marks))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wikiann = {split: wikiann_records(split) for split in SPLITS}
    # WikiANN repeats sentences across its splits; a test sentence the model trained on would
    # inflate its score.
    seen = {record["text"] for split in ("train", "validation") for record in wikiann[split]}
    wikiann["test"] = [record for record in wikiann["test"] if record["text"] not in seen]
    for split in SPLITS:
        write_jsonl(args.out_dir / f"wikiann_el_{split}.jsonl", wikiann[split])


if __name__ == "__main__":
    main()
