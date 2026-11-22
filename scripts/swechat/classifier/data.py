"""Turn file-level inputs into fixed-length token chunks with sentence spans, for training and evaluation.

Each file version is tokenized once. Its labeled sentences are then grouped into chunks of at most `max_tokens`
tokens: a chunk is centered on the first sentence not yet covered, and takes every other sentence that fits entirely
inside it. A file shorter than `max_tokens` becomes a single chunk holding all of its sentences.

A chunk is a dict:
    input_ids   [CLS] + the chunk's tokens + [SEP]
    spans       list of (first_token, last_token_exclusive) per sentence, positions within input_ids
    labels      1 for slop, 0 for not_slop, per sentence
    comment_ids, repo_ids, hard (the sentence is a hard negative: flagged by the screen, rejected by the judge)

Chunks are cached in data/swechat/classifier/cache/ so each (split, max_tokens, model) is tokenized only once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split import OUT  # noqa: E402

DATA = Path(__file__).resolve().parents[3] / "data/swechat/classifier"


def token_span(offsets: np.ndarray, start: int, end: int) -> tuple[int, int]:
    """Indices of the tokens overlapping characters [start, end), as (first, last_exclusive)."""
    first = int(np.searchsorted(offsets[:, 1], start, side="right"))
    last = int(np.searchsorted(offsets[:, 0], end, side="left"))
    return first, max(last, first + 1)


def chunk_file(token_ids: list[int], spans: list[tuple[int, int]], max_tokens: int) -> list[tuple[int, list[int]]]:
    """Group sentence spans into windows. Returns (window_start, indices of the spans inside it) per chunk."""
    room = max_tokens - 2  # leave space for [CLS] and [SEP]
    n = len(token_ids)
    if n <= room:
        return [(0, list(range(len(spans))))]
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    covered, chunks = set(), []
    for i in order:
        if i in covered:
            continue
        center = (spans[i][0] + spans[i][1]) // 2
        start = min(max(center - room // 2, 0), n - room)
        inside = [j for j in order if j not in covered and spans[j][0] >= start and spans[j][1] <= start + room]
        inside = inside or [i]  # a sentence longer than the window is kept, truncated to it
        covered.update(inside)
        chunks.append((start, inside))
    return chunks


def build_chunks(split: str, max_tokens: int, tokenizer) -> list[dict]:
    """All chunks of one split ("train", "val", "test" or "pilot_holdout")."""
    files = pd.read_parquet(DATA / "inputs.parquet")
    meta = pd.read_parquet(OUT / "split.parquet", columns=["comment_id", "repo_id", "source"])
    repo_of = dict(zip(meta.comment_id, meta.repo_id))
    hard_ids = set(meta.comment_id[meta.source == "judge"])
    room = max_tokens - 2

    chunks = []
    for text, file_spans in zip(files.text, files.spans):
        sentences = [s for s in file_spans if s["split"] == split]
        if not sentences:
            continue
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = encoded["input_ids"]
        offsets = np.array(encoded["offset_mapping"]).reshape(-1, 2)
        spans = [token_span(offsets, s["start"], s["end"]) for s in sentences]

        for start, members in chunk_file(token_ids, spans, max_tokens):
            window = token_ids[start:start + room]
            chunk_spans = []
            for j in members:  # shift into the window, +1 for [CLS]; clip sentences longer than the window
                first = max(spans[j][0] - start, 0) + 1
                last = min(spans[j][1] - start, len(window)) + 1
                chunk_spans.append((first, last))
            chunks.append({
                "input_ids": [tokenizer.cls_token_id, *window, tokenizer.sep_token_id],
                "spans": chunk_spans,
                "labels": [int(sentences[j]["label"] == "slop") for j in members],
                "comment_ids": [int(sentences[j]["comment_id"]) for j in members],
                "repo_ids": [repo_of[sentences[j]["comment_id"]] for j in members],
                "hard": [sentences[j]["comment_id"] in hard_ids for j in members],
            })
    return chunks


def load_chunks(split: str, max_tokens: int, tokenizer, model_name: str) -> list[dict]:
    """build_chunks with an on-disk cache."""
    cache = DATA / "cache" / f"{model_name.replace('/', '_')}_{max_tokens}_{split}.pt"
    if cache.exists():
        return torch.load(cache)
    chunks = build_chunks(split, max_tokens, tokenizer)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(chunks, cache)
    return chunks


def collate(chunks: list[dict], pad_id: int) -> dict:
    """Pad a batch of chunks. Spans become rows of (chunk index in batch, first, last)."""
    longest = max(len(c["input_ids"]) for c in chunks)
    input_ids = torch.full((len(chunks), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(chunks), longest), dtype=torch.long)
    spans, labels = [], []
    for b, c in enumerate(chunks):
        input_ids[b, :len(c["input_ids"])] = torch.tensor(c["input_ids"])
        attention_mask[b, :len(c["input_ids"])] = 1
        spans += [(b, first, last) for first, last in c["spans"]]
        labels += c["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "spans": torch.tensor(spans), "labels": torch.tensor(labels, dtype=torch.float)}
