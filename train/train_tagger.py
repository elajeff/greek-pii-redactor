"""Fine-tunes XLM-R base to tag PERSON spans (labels O, B-PERSON, I-PERSON).

Base model: FacebookAI/xlm-roberta-base, MIT licence (stated on its Hugging Face model page).
Training data: WikiANN el train and data/synthetic_train.jsonl. The synthetic ID spans (AFM, AMKA,
IBAN, PHONE, EMAIL) stay label O: the rules find those. WikiANN el validation picks the epoch, by
exact character-span F1. Needs a CUDA GPU.

Setup: pip install -e ".[train]", then the data scripts named in eval/evaluate.py.
Run (the weights directory must be outside the repo, nothing from it is committed):
    python train/train_tagger.py --out C:/Users/me/greek-pii-redactor-weights/xlmr-person
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)

from greek_pii_redactor.rules import Span
from greek_pii_redactor.tagger import (
    BEGIN,
    INSIDE,
    LABELS,
    MAX_LENGTH,
    OUTSIDE,
    decode_spans,
    tag_texts,
)

BASE_MODEL = "FacebookAI/xlm-roberta-base"
SEED = 42
IGNORE = -100  # label id that the loss skips (special tokens)
BATCH_SIZE = 32  # fits in 12 GB at this length with bf16
LEARNING_RATE = 2e-5  # a common starting value for fine-tuning an encoder like this
EPOCHS = 3  # the best epoch on validation is kept, so a later one can't make the weights worse
WARMUP_FRACTION = 0.1  # the first tenth of the steps ramp the rate up, then it decays linearly
MAX_GRAD_NORM = 1.0  # clips rare exploding gradients
EVAL_BATCH_SIZE = 64  # no gradients are stored, so a larger batch than in training is fine

TRAIN_FILES = ("wikiann_el_train.jsonl", "synthetic_train.jsonl")
VALIDATION_FILE = "wikiann_el_validation.jsonl"


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def person_spans(record: dict) -> set[Span]:
    return {Span(*span) for span in record["spans"] if span[2] == "PERSON"}


def label_tokens(offsets: list[tuple[int, int]], spans: set[Span]) -> list[int]:
    """B for the first token overlapping a gold span, I for the other overlapping tokens."""
    labels = []
    opened: set[Span] = set()
    for token_start, token_end in offsets:
        if token_start == token_end:
            labels.append(IGNORE)
            continue
        label = OUTSIDE
        for span in spans:
            if token_start < span.end and span.start < token_end:
                label = INSIDE if span in opened else BEGIN
                opened.add(span)
                break
        labels.append(label)
    return labels


def encode(
    tokenizer: PreTrainedTokenizerFast, records: list[dict]
) -> list[tuple[list[int], list[int]]]:
    """(token ids, labels) per record, cut at MAX_LENGTH tokens."""
    encoded = tokenizer(
        [record["text"] for record in records],
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    return [
        (ids, label_tokens(offsets, person_spans(record)))
        for ids, offsets, record in zip(
            encoded["input_ids"], encoded["offset_mapping"], records, strict=True
        )
    ]


def print_unreproducible_share(
    name: str, tokenizer: PreTrainedTokenizerFast, records: list[dict]
) -> None:
    """Gold spans that labelling tokens and decoding them does not give back exactly."""
    encoded = tokenizer(
        [record["text"] for record in records],
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    total = lost = cut = 0
    for record, offsets in zip(records, encoded["offset_mapping"], strict=True):
        gold = person_spans(record)
        rebuilt = set(decode_spans(offsets, label_tokens(offsets, gold)))
        total += len(gold)
        lost += len(gold - rebuilt)
        cut += any(span.end > offsets[-2][1] for span in gold)  # past the last real token
    print(
        f"{name}: {lost} of {total} gold spans ({lost / total:.2%}) cannot be rebuilt exactly "
        f"from token boundaries; {cut} sentences have a name past the {MAX_LENGTH}-token cut"
    )


def make_batch(
    examples: list[tuple[list[int], list[int]]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    longest = max(len(ids) for ids, _ in examples)
    input_ids = torch.full((len(examples), longest), pad_token_id)
    labels = torch.full((len(examples), longest), IGNORE)
    attention_mask = torch.zeros((len(examples), longest), dtype=torch.long)
    for row, (ids, tags) in enumerate(examples):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        labels[row, : len(tags)] = torch.tensor(tags)
        attention_mask[row, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def score(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, records: list[dict]
) -> tuple[float, ...]:
    """Exact character-span precision, recall and F1, decoded the way inference does."""
    model.eval()
    true_pos = false_pos = false_neg = 0
    for first in range(0, len(records), EVAL_BATCH_SIZE):
        chunk = records[first : first + EVAL_BATCH_SIZE]
        predictions = tag_texts(model, tokenizer, [record["text"] for record in chunk])
        for record, predicted in zip(chunk, predictions, strict=True):
            gold = person_spans(record)
            true_pos += len(gold & set(predicted))
            false_pos += len(set(predicted) - gold)
            false_neg += len(gold - set(predicted))
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for the weights")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Training on the CPU is not supported.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_records = [
        record for name in TRAIN_FILES for record in load_records(args.data_dir / name)
    ]
    validation_records = load_records(args.data_dir / VALIDATION_FILE)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print_unreproducible_share("train", tokenizer, train_records)
    print_unreproducible_share("validation", tokenizer, validation_records)

    examples = encode(tokenizer, train_records)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS))
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = EPOCHS * -(-len(examples) // BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_FRACTION * total_steps), total_steps
    )

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(examples)
        loss_sum = 0.0
        batches = 0
        for first in range(0, len(examples), BATCH_SIZE):
            batch = make_batch(examples[first : first + BATCH_SIZE], tokenizer.pad_token_id)
            batch = {name: tensor.to("cuda") for name, tensor in batch.items()}
            # bf16 is native on the RTX 5070, halves activation memory and needs no loss scaling
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
            batches += 1

        precision, recall, f1 = score(model, tokenizer, validation_records)
        print(
            f"epoch {epoch}: train loss {loss_sum / batches:.4f}, validation "
            f"precision {precision:.3f} recall {recall:.3f} F1 {f1:.3f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"saved epoch {epoch} to {args.out}")


if __name__ == "__main__":
    main()
