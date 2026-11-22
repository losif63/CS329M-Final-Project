"""Score a trained checkpoint on the test split (and the pilot holdout), using the threshold chosen on val.

    python scripts/swechat/classifier/evaluate.py --run runs/n512

Reads the run's config.json, history.json (for the val threshold of the best checkpoint) and best.pt. Nothing is
tuned on test: the threshold is the one the saved checkpoint got on val.

For each split it reports PR-AUC, precision / recall / F1 at the val threshold, recall at precision 0.5 and 0.8, and
recall per slop category, with 95% intervals from resampling repositories. It also compares the model with the 100
hand labels (24 in test, 76 in pilot_holdout). Writes to the run folder:
    eval.json                     all metrics
    predictions_<split>.parquet   comment_id, repo_id, category, label, score, predicted, sentence
    errors_<split>.parquet        false positives (highest score first), then false negatives (lowest score first)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, cohen_kappa_score, precision_recall_curve
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DATA, load_chunks  # noqa: E402
from metrics import evaluate  # noqa: E402
from train import SpanClassifier, pick_device, predict  # noqa: E402


def recall_at_precision(y: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    """The highest recall reachable with precision at least min_precision, at any threshold."""
    precision, recall, _ = precision_recall_curve(y, scores)
    reachable = recall[precision >= min_precision]
    return round(float(reachable.max()), 3) if len(reachable) else 0.0


def summarize(df: pd.DataFrame, threshold: float) -> dict:
    y, scores = df.label.to_numpy(), df.score.to_numpy()
    return {"pr_auc": round(float(average_precision_score(y, scores)), 4),
            **evaluate(y, scores >= threshold),
            "recall_at_precision_0.5": recall_at_precision(y, scores, 0.5),
            "recall_at_precision_0.8": recall_at_precision(y, scores, 0.8)}


def bootstrap(df: pd.DataFrame, threshold: float, rounds: int, seed: int = 0) -> dict:
    """95% intervals for PR-AUC, precision, recall and F1, resampling whole repositories."""
    rng = np.random.default_rng(seed)
    rows_of = {repo: idx for repo, idx in df.groupby("repo_id").indices.items()}
    repos = np.array(list(rows_of))
    samples = {"pr_auc": [], "precision": [], "recall": [], "f1": []}
    for _ in range(rounds):
        idx = np.concatenate([rows_of[r] for r in rng.choice(repos, len(repos))])
        part = df.iloc[idx]
        if part.label.sum() == 0:
            continue
        result = summarize(part, threshold)
        for key in samples:
            if result[key] is not None:
                samples[key].append(result[key])
    return {key: [round(float(v), 3) for v in np.percentile(values, [2.5, 97.5])] for key, values in samples.items()}


def sentence_texts(comment_ids: set[int]) -> dict[int, str]:
    """The sentence text of each comment id, cut from its file in inputs.parquet."""
    texts = {}
    for text, spans in pd.read_parquet(DATA / "inputs.parquet", columns=["text", "spans"]).itertuples(index=False):
        for s in spans:
            if s["comment_id"] in comment_ids:
                texts[s["comment_id"]] = text[s["start"]:s["end"]]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="a train.py output folder")
    parser.add_argument("--splits", nargs="+", default=["test", "pilot_holdout"])
    parser.add_argument("--bootstrap", type=int, default=1000, help="resampling rounds for the intervals")
    args = parser.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    history = json.loads((args.run / "history.json").read_text())
    best = max(history, key=lambda e: e["pr_auc"])  # the evaluation at which best.pt was saved
    threshold = best["threshold"]
    print(f"{args.run}: checkpoint from epoch {best['epoch']} (val PR-AUC {best['pr_auc']}), threshold {threshold}")

    device = pick_device()
    fp16 = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = SpanClassifier(config["model"]).to(device)
    model.load_state_dict(torch.load(args.run / "best.pt", map_location=device))
    meta = pd.read_parquet(DATA / "split.parquet", columns=["comment_id", "repo_id", "category", "hand_label"])

    results = {"run": str(args.run), "checkpoint_epoch": best["epoch"], "threshold": threshold}
    scored = []
    for split in args.splits:
        chunks = load_chunks(split, config["max_tokens"], tokenizer, config["model"])
        df = predict(model, chunks, device, 2 * config["batch_size"], tokenizer.pad_token_id, fp16)
        df = df.merge(meta, on="comment_id")
        df["predicted"] = df.score >= threshold
        df["sentence"] = df.comment_id.map(sentence_texts(set(df.comment_id)))
        scored.append(df)

        slop = df[df.label == 1]
        results[split] = {
            **summarize(df, threshold),
            "intervals_95": bootstrap(df, threshold, args.bootstrap),
            "recall_by_category": {c: {"recall": round(float(g.predicted.mean()), 3), "slop": len(g)}
                                   for c, g in slop.groupby("category")},
        }
        print(split, json.dumps({k: v for k, v in results[split].items() if k != "recall_by_category"}))
        df.to_parquet(args.run / f"predictions_{split}.parquet", index=False)
        false_pos = df[df.predicted & (df.label == 0)].sort_values("score", ascending=False)
        false_neg = df[~df.predicted & (df.label == 1)].sort_values("score")
        pd.concat([false_pos.assign(error="false_positive"), false_neg.assign(error="false_negative")]).to_parquet(
            args.run / f"errors_{split}.parquet", index=False)

    hand = pd.concat(scored)
    hand = hand[hand.hand_label.isin(["slop", "not_slop"])]
    if len(hand):
        human = (hand.hand_label == "slop").to_numpy()
        results["hand_labels"] = {
            "sentences": len(hand), "hand_slop": int(human.sum()),
            "model_agrees": int((hand.predicted.to_numpy() == human).sum()),
            "model_kappa": round(float(cohen_kappa_score(human, hand.predicted)), 3),
            "judge_agrees": int(((hand.label == 1).to_numpy() == human).sum()),
            "judge_kappa": round(float(cohen_kappa_score(human, hand.label == 1)), 3),
        }
        print("hand labels", json.dumps(results["hand_labels"]))

    (args.run / "eval.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
