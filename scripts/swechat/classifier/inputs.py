"""Collect the classifier's inputs: each file version's full text and the spans of its labeled sentences.

    python scripts/swechat/classifier/inputs.py

The model encodes a whole file once and scores every labeled comment sentence in it by pooling over that sentence's
tokens, so an input is just the file text plus a list of character spans. Nothing is inserted into the text. Splitting
long files into chunks happens at training time, because it depends on the tokenizer and the chosen context length.

A sentence's span is found inside its source lines (comment.line_start..line_end), allowing the line breaks and
comment prefixes (//, #, *, --) of multi-line comments. If that fails (about 0.3%, mostly doc comments whose markup
was stripped), the span covers the sentence's whole lines instead.

Writes data/swechat/classifier/inputs.parquet, one row per file version:
    source_id, split, text, spans
where spans is a list of {comment_id, start, end, label, category, split}; text[start:end] is the sentence.
A span's split differs from its file's only for pilot sentences ("pilot_holdout"), which training must skip.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split import DB, OUT  # noqa: E402

WORD_GAP = r"\s+(?:(?:/{2,}!?|#+|\*|--|;+)[ \t]*)?"  # whitespace, possibly a line break plus a comment prefix


def locate(sentence: str, text: str, line_offsets: np.ndarray, line_start: int, line_end: int) -> tuple[int, int, bool]:
    """Character span (start, end) of the sentence in the file, and whether it was matched exactly."""
    region_start = int(line_offsets[line_start - 1])
    region_end = int(line_offsets[line_end]) - 1  # end of the last line, without its newline
    pattern = WORD_GAP.join(re.escape(word) for word in sentence.split())
    match = re.search(pattern, text[region_start:region_end])
    if match is None:
        return region_start, region_end, False
    return region_start + match.start(), region_start + match.end(), True


def main() -> None:
    con = duckdb.connect()
    sentences = con.execute(f"""
        SELECT c.source_id, sp.comment_id, sp.split, sp.label, sp.category,
               c.sentence, c.line_start, c.line_end
        FROM '{OUT / "split.parquet"}' sp JOIN '{DB / "comment.parquet"}' c ON c.id = sp.comment_id
        ORDER BY c.source_id, c.line_start, sp.comment_id""").df()
    texts = dict(con.execute(f"""
        SELECT id, content FROM '{DB / "source_file.parquet"}'
        WHERE id IN (SELECT DISTINCT source_id FROM sentences)""").fetchall())

    files, exact = [], []
    for source_id, group in sentences.groupby("source_id", sort=True):
        text = texts[source_id]
        line_lengths = [len(line) + 1 for line in text.split("\n")]
        line_offsets = np.concatenate([[0], np.cumsum(line_lengths)])
        spans = []
        for s in group.itertuples():
            start, end, found = locate(s.sentence, text, line_offsets, s.line_start, s.line_end)
            exact.append(found)
            spans.append({"comment_id": s.comment_id, "start": start, "end": end,
                          "label": s.label, "category": s.category, "split": s.split})
        # Every sentence of a file shares its repository's split; pilot sentences are the only exception.
        file_split = next((sp["split"] for sp in spans if sp["split"] != "pilot_holdout"), "pilot_holdout")
        files.append({"source_id": source_id, "split": file_split, "text": text, "spans": spans})

    out = pd.DataFrame(files)
    out.to_parquet(OUT / "inputs.parquet", index=False, compression="zstd")

    spans_per_file = out.spans.map(len)
    lengths = out.text.str.len()
    print(f"{len(out):,} file versions, {spans_per_file.sum():,} sentence spans "
          f"(median {spans_per_file.median():.0f} per file); located exactly: {np.mean(exact):.1%}")
    print(f"file length in characters: median {lengths.median():,.0f}, p90 {lengths.quantile(0.9):,.0f}, "
          f"total {lengths.sum() / 1e6:,.0f}M")
    print(out.split.value_counts().to_string())


if __name__ == "__main__":
    main()
