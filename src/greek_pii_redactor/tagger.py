"""PERSON tagger: a fine-tuned XLM-R token classifier (see train/train_tagger.py)."""

from collections.abc import Callable
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from greek_pii_redactor.rules import Span

LABELS = ("O", "B-PERSON", "I-PERSON")
OUTSIDE, BEGIN, INSIDE = range(len(LABELS))
# 99% of WikiANN el train sentences are 60 tokens or fewer; 128 leaves room for longer ones.
MAX_LENGTH = 128
# A longer text is cut into windows that share this many tokens, so a name cut at the edge of one
# window is whole in the next. Truncating instead would silently drop the names after the cut.
WINDOW_OVERLAP = 32
# A long text has many windows; running them all at once ran out of 12 GB on a 1 MB text.
# 64 windows of 128 tokens add about 0.3 GB to the model's 1.1 GB.
WINDOW_BATCH_SIZE = 64


def decode_spans(offsets: list[tuple[int, int]], labels: list[int]) -> list[Span]:
    """Turns token labels into character spans, from the first token's start to the last's end."""
    spans: list[Span] = []
    start = end = None
    for (token_start, token_end), label in zip(offsets, labels, strict=True):
        if token_start == token_end:  # special or padding token
            continue
        # An I with no open span still starts one: missing a name leaks it, an extra one does not.
        if label == BEGIN or (label == INSIDE and start is None):
            if start is not None:
                spans.append(Span(start, end, "PERSON"))
            start, end = token_start, token_end
        elif label == INSIDE:
            end = token_end
        elif start is not None:
            spans.append(Span(start, end, "PERSON"))
            start = end = None
    if start is not None:
        spans.append(Span(start, end, "PERSON"))
    return spans


def tag_texts(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, texts: list[str]
) -> list[list[Span]]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=WINDOW_OVERLAP,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping").tolist()
    owners = encoded.pop("overflow_to_sample_mapping").tolist()
    encoded = encoded.to(model.device)
    labels: list[list[int]] = []
    with torch.no_grad():
        for first in range(0, len(owners), WINDOW_BATCH_SIZE):
            window_slice = slice(first, first + WINDOW_BATCH_SIZE)
            batch = {name: tensor[window_slice] for name, tensor in encoded.items()}
            labels += model(**batch).logits.argmax(dim=-1).tolist()

    found: list[list[Span]] = [[] for _ in texts]
    for owner, window_offsets, window_labels in zip(owners, offsets, labels, strict=True):
        found[owner] += decode_spans(window_offsets, window_labels)

    merged_per_text = []
    for spans in found:
        merged: list[Span] = []
        for span in sorted(spans):
            # The same name seen in two windows, or cut in one and whole in the other.
            if merged and span.start < merged[-1].end:
                merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end), "PERSON")
            else:
                merged.append(span)
        merged_per_text.append(merged)
    return merged_per_text


def load_tagger(model_dir: Path) -> Callable[[str], list[Span]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def tag(text: str) -> list[Span]:
        return tag_texts(model, tokenizer, [text])[0]

    return tag
