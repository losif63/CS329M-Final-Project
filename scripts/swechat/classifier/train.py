"""Fine-tune an encoder to score comment sentences as slop, pooling over each sentence's tokens.

    python scripts/swechat/classifier/train.py --max-tokens 512 --out runs/n512
    python scripts/swechat/classifier/train.py --max-tokens 512 --limit-train 400 --limit-val 400 --epochs 1  # timing

Model: encoder → one vector per token → for each sentence, [mean of its tokens ; the [CLS] vector] → small MLP → logit.

Training data, re-drawn every epoch: every chunk holding a slop sentence or a hard negative, plus a random
--easy-fraction of the chunks that hold only easy negatives. The loss is a weighted binary cross-entropy:
    - a repository holding more than --repo-cap of a class's training sentences is scaled down to that share;
    - slop is weighted up so both classes carry the same total weight per epoch.

Evaluates on val every half epoch; keeps the checkpoint with the best val PR-AUC. Writes to --out:
    best.pt, val_predictions.parquet (comment_id, label, score), history.json, config.json
With --wandb, also logs training loss and val metrics to Weights & Biases (run `wandb login` once first).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import collate, load_chunks  # noqa: E402
from metrics import best_f1_threshold, evaluate  # noqa: E402


class SpanClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, input_ids, attention_mask, spans):
        """spans: rows of (chunk index in batch, first token, last token exclusive). Returns one logit per span."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        # Mean over a span from a running sum: sum of tokens first..last-1 = running[last] - running[first].
        running = nn.functional.pad(hidden.cumsum(dim=1), (0, 0, 1, 0))
        b, first, last = spans.unbind(dim=1)
        span_mean = (running[b, last] - running[b, first]) / (last - first).unsqueeze(1)
        return self.head(torch.cat([span_mean, hidden[b, 0]], dim=1)).squeeze(1)


def is_important(chunk: dict) -> bool:
    return any(chunk["labels"]) or any(chunk["hard"])


def attach_weights(chunks: list[dict], repo_cap: float, easy_fraction: float) -> float:
    """Store a per-sentence weight in each chunk (the repository cap). Returns the slop weight that balances classes."""
    per_repo = {0: Counter(), 1: Counter()}
    for c in chunks:
        for repo, y in zip(c["repo_ids"], c["labels"]):
            per_repo[y][repo] += 1
    totals = {y: sum(counts.values()) for y, counts in per_repo.items()}
    cap = {y: {repo: min(1.0, repo_cap * totals[y] / n) for repo, n in counts.items()} for y, counts in per_repo.items()}

    expected = {0: 0.0, 1: 0.0}  # total weight per class in one epoch, given that easy chunks are subsampled
    for c in chunks:
        c["weights"] = [cap[y][repo] for repo, y in zip(c["repo_ids"], c["labels"])]
        share = 1.0 if is_important(c) else easy_fraction
        for w, y in zip(c["weights"], c["labels"]):
            expected[y] += w * share
    return expected[0] / expected[1]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model, chunks, device, batch_size, pad_id, fp16) -> pd.DataFrame:
    """Score every sentence in the chunks. Chunks are sorted by length so batches need little padding."""
    model.eval()
    ids, labels, scores = [], [], []
    ordered = sorted(chunks, key=lambda c: len(c["input_ids"]))
    for i in range(0, len(ordered), batch_size):
        part = ordered[i:i + batch_size]
        batch = collate(part, pad_id)
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["spans"].to(device))
        scores += torch.sigmoid(logits.float()).cpu().tolist()
        labels += batch["labels"].int().tolist()
        ids += [cid for c in part for cid in c["comment_ids"]]
    model.train()
    return pd.DataFrame({"comment_id": ids, "label": labels, "score": scores})


def report(predictions: pd.DataFrame) -> dict:
    y, scores = predictions.label.to_numpy(), predictions.score.to_numpy()
    threshold = best_f1_threshold(y, scores)
    return {"pr_auc": round(float(average_precision_score(y, scores)), 4), "threshold": round(threshold, 4),
            **evaluate(y, scores >= threshold)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2, help="batches per optimizer step")
    parser.add_argument("--lr", type=float, default=3e-5, help="encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--easy-fraction", type=float, default=0.2)
    parser.add_argument("--repo-cap", type=float, default=0.05)
    parser.add_argument("--grad-checkpointing", action="store_true", help="save GPU memory at ~30%% more time")
    parser.add_argument("--limit-train", type=int, help="use only this many training chunks (timing tests)")
    parser.add_argument("--limit-val", type=int, help="evaluate on only this many val chunks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/default"))
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", default="slop-classifier")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    fp16 = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "out": str(args.out)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.out.name, config=config)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id
    train_chunks = load_chunks("train", args.max_tokens, tokenizer, args.model)[:args.limit_train]
    val_chunks = load_chunks("val", args.max_tokens, tokenizer, args.model)[:args.limit_val]
    slop_weight = attach_weights(train_chunks, args.repo_cap, args.easy_fraction)
    print(f"device {device}, fp16 {fp16}; {len(train_chunks):,} train chunks, {len(val_chunks):,} val chunks; "
          f"slop weight {slop_weight:.1f}", flush=True)

    model = SpanClassifier(args.model).to(device)
    if args.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                                   {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    chunks_per_epoch = sum(1.0 if is_important(c) else args.easy_fraction for c in train_chunks)
    total_steps = int(args.epochs * chunks_per_epoch / (args.batch_size * args.grad_accum)) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=fp16)

    def make_batch(chunks):
        batch = collate(chunks, pad_id)
        batch["weights"] = torch.tensor([w for c in chunks for w in c["weights"]])
        return batch

    history, best_auc, step, tokens, started = [], -1.0, 0, 0, time.time()
    for epoch in range(args.epochs):
        sample = [c for c in train_chunks if is_important(c) or random.random() < args.easy_fraction]
        loader = DataLoader(sample, batch_size=args.batch_size, shuffle=True, collate_fn=make_batch)
        eval_points = {len(loader) // 2, len(loader)}
        for i, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                logits = model(batch["input_ids"], batch["attention_mask"], batch["spans"])
            weights = batch["weights"] * torch.where(batch["labels"] == 1, slop_weight, 1.0)
            loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), batch["labels"], weight=weights)
            scaler.scale(loss / args.grad_accum).backward()
            tokens += int(batch["attention_mask"].sum())

            if i % args.grad_accum == 0 or i == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                if step % 25 == 0:
                    elapsed = time.time() - started
                    print(f"epoch {epoch + 1} batch {i}/{len(loader)} step {step}/{total_steps} "
                          f"loss {loss.item():.4f} {tokens / elapsed:,.0f} tokens/s {elapsed / 60:.1f} min", flush=True)
                    if args.wandb:
                        wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0],
                                   "train/tokens_per_s": tokens / elapsed}, step=step)

            if i in eval_points:
                predictions = predict(model, val_chunks, device, 2 * args.batch_size, pad_id, fp16)
                result = {"epoch": round(epoch + i / len(loader), 2), "step": step, **report(predictions)}
                history.append(result)
                print("val", json.dumps(result), flush=True)
                if args.wandb:
                    wandb.log({f"val/{k}": v for k, v in result.items() if isinstance(v, (int, float))}, step=step)
                if result["pr_auc"] > best_auc:
                    best_auc = result["pr_auc"]
                    torch.save(model.state_dict(), args.out / "best.pt")
                    predictions.to_parquet(args.out / "val_predictions.parquet", index=False)
                (args.out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"done in {(time.time() - started) / 60:.1f} min; best val PR-AUC {best_auc}", flush=True)
    if args.wandb:
        wandb.summary["best_val_pr_auc"] = best_auc
        wandb.finish()


if __name__ == "__main__":
    main()
