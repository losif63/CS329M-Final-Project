"""Split the labeled comment sentences into train / val / test by repository.

    python scripts/swechat/classifier/split.py [--labels v1.2] [--train 0.70 --val 0.15 --test 0.15]

Every sentence of a repository lands in the same split, so the classifier is always tested on repositories it has
never seen. Slop is concentrated in a few repositories (the top one holds about a fifth of it), so repositories are
assigned greedily, largest slop count first, to whichever split is furthest below its target share of slop. This
keeps the slop share of each split close to its target; repositories without slop are then balanced by sentence count.

Two kinds of rows are kept out of training:
    unsure          sentences the judge could not decide; dropped entirely.
    pilot_holdout   the 400 labeling-pilot sentences (100 of them hand-labeled), when their repository is in train or
                    val. Pilot sentences in test repositories simply stay in test.

Writes data/swechat/classifier/split.parquet (one row per sentence) and split_summary.json next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data/swechat/db"
OUT = ROOT / "data/swechat/classifier"

SPLITS = ("train", "val", "test")


def load_sentences(label_tag: str) -> pd.DataFrame:
    """One row per labeled sentence with its repository and label details."""
    query = f"""
        SELECT c.id AS comment_id, s.repo_id, s.session_id, s.language,
               l.label, l.category, l.source, l.pilot_item, l.hand_label
        FROM '{DB / "comment.parquet"}' c
        JOIN '{DB / "source_file.parquet"}' s ON s.id = c.source_id
        JOIN '{DB / f"label_{label_tag}.parquet"}' l ON l.comment_id = c.id
        WHERE l.label <> 'unsure'
        ORDER BY c.id
    """
    return duckdb.connect().execute(query).df()


def assign_repositories(sentences: pd.DataFrame, targets: dict[str, float]) -> dict[str, str]:
    """Map each repository to a split, balancing slop first and sentence count second."""
    repos = (sentences.assign(is_slop=sentences.label == "slop")
             .groupby("repo_id")
             .agg(slop=("is_slop", "sum"), sentences=("comment_id", "count"))
             .reset_index()
             .sort_values(["slop", "sentences", "repo_id"], ascending=[False, False, True]))
    total_slop = repos.slop.sum()
    total_sentences = repos.sentences.sum()
    slop_in = dict.fromkeys(SPLITS, 0)
    sentences_in = dict.fromkeys(SPLITS, 0)

    assignment = {}
    for repo_id, slop, n_sentences in zip(repos.repo_id, repos.slop, repos.sentences):
        if slop > 0:
            deficit = {s: targets[s] * total_slop - slop_in[s] for s in SPLITS}
        else:
            deficit = {s: targets[s] * total_sentences - sentences_in[s] for s in SPLITS}
        split = max(SPLITS, key=lambda s: deficit[s])
        assignment[repo_id] = split
        slop_in[split] += slop
        sentences_in[split] += n_sentences
    return assignment


def summarize(table: pd.DataFrame) -> dict:
    """Per-split counts: repositories, sentences, slop and its rate, and slop by category."""
    summary = {}
    for split, rows in table.groupby("split"):
        slop = rows[rows.label == "slop"]
        summary[split] = {
            "repositories": int(rows.repo_id.nunique()),
            "sentences": len(rows),
            "slop": len(slop),
            "slop_rate": round(len(slop) / len(rows), 4),
            "hard_negatives": int(((rows.label == "not_slop") & (rows.source == "judge")).sum()),
            "hand_labeled": int(rows.hand_label.notna().sum()),
            "top_repo_slop_share": round(slop.repo_id.value_counts(normalize=True).max(), 3) if len(slop) else 0.0,
            "slop_by_category": slop.category.value_counts().to_dict(),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="v1.2", help="label tag: reads data/swechat/db/label_<tag>.parquet")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()
    targets = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(targets.values()) - 1) > 1e-9:
        raise SystemExit(f"split fractions must sum to 1, got {sum(targets.values())}")

    sentences = load_sentences(args.labels)
    assignment = assign_repositories(sentences, targets)
    sentences["split"] = sentences.repo_id.map(assignment)

    in_pilot = sentences.pilot_item.notna()
    sentences.loc[in_pilot & (sentences.split != "test"), "split"] = "pilot_holdout"

    OUT.mkdir(parents=True, exist_ok=True)
    columns = ["comment_id", "split", "repo_id", "session_id", "language",
               "label", "category", "source", "pilot_item", "hand_label"]
    sentences[columns].to_parquet(OUT / "split.parquet", index=False, compression="zstd")

    summary = {"labels": args.labels, "targets": targets, "splits": summarize(sentences)}
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
