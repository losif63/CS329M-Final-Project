"""Two baselines for slop detection, evaluated on the val and test splits.

    python scripts/swechat/classifier/baselines.py

    majority  never predicts slop (the most common class)
    keyword   predicts slop when the sentence matches the keyword regex in scripts/swechat/markers.py

Writes data/swechat/classifier/baselines.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from markers import MARKER_RE  # noqa: E402
from metrics import evaluate  # noqa: E402
from split import DB, OUT  # noqa: E402


def main() -> None:
    df = duckdb.connect().execute(f"""
        SELECT sp.split, sp.label = 'slop' AS is_slop, c.sentence
        FROM '{OUT / "split.parquet"}' sp JOIN '{DB / "comment.parquet"}' c ON c.id = sp.comment_id
        WHERE sp.split IN ('val', 'test')""").df()

    results = {}
    for split in ("val", "test"):
        rows = df[df.split == split]
        y = rows.is_slop.to_numpy()
        keyword_match = np.array([bool(MARKER_RE.search(s)) for s in rows.sentence])
        results[split] = {
            "majority": evaluate(y, np.zeros(len(y), dtype=bool)),
            "keyword": evaluate(y, keyword_match),
        }

    (OUT / "baselines.json").write_text(json.dumps(results, indent=2) + "\n")
    for split, by_baseline in results.items():
        print(f"\n{split}")
        for name, r in by_baseline.items():
            precision = "n/a" if r["precision"] is None else r["precision"]
            print(f"  {name:9} flagged {r['flagged']:>5}  correct {r['correct']:>4} / {r['slop']} slop"
                  f"  precision {precision}  recall {r['recall']}  F1 {r['f1']}")


if __name__ == "__main__":
    main()
