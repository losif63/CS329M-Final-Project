"""Cheap, high-recall slop screening: many sentences per LLM call, comment text only (no session, no code).

Its job is to shrink the pool that the full judge (pilot.py label, with session context) has to read. It is not a
label: anything it flags still goes to the judge.

    python scripts/swechat/labeling/screen.py pilot [--run pilot_v1]   # evaluate on the labeled pilot items
    python scripts/swechat/labeling/screen.py all   [--out screen_all] # screen every sentence (resumable)

Options: --backend codex|claude, --model, --effort, --batch 40, --workers 4, --seed 0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot  # noqa: E402

INSTRUCTIONS = """You screen code-comment sentences written by an AI coding agent, to find candidates for a later,
careful review. A sentence is a candidate if it MIGHT depend on the session in which the agent wrote it, rather than
describing the code as it stands. Flag a sentence if it might:
- narrate how the code changed ("now", "no longer", "used to", "previously", "the old …", "replaces …", "was removed");
- refer to the conversation or the person the agent worked for ("as requested", "the user reported", "per our discussion");
- state facts about the author's machine, paths, accounts, or one particular run (home directories, "on this machine",
  versions found installed, files "found in ~/Downloads");
- report measurements or results from runs ("measured at 3.2 MB", "5.7x faster", "tests pass");
- point to a plan, phase, step, story, or task number that may live outside the repository ("slice 2.4", "Phase 5", "T098");
- think aloud: options considered or abandoned, self-corrections, plans to try something ("Actually, let's…", "OR: …").
Do not flag sentences that only explain what the code does or why it is designed that way, list parameters, or
describe tests, even if they are wordy. When unsure, flag: missing a candidate is worse than flagging an extra one.

You get numbered entries, each with the sentence, the full comment it comes from (for context), and the file.
Return the ids of the candidate sentences."""

SCHEMA = {"type": "object", "properties": {"flagged": {"type": "array", "items": {"type": "string"}}},
          "required": ["flagged"], "additionalProperties": False}
COMMENT_CHARS = 600


def batch_prompt(entries: list[dict]) -> tuple[str, dict[str, int]]:
    """entries: {key, sentence, comment, file, language, kind}. Returns the prompt and neutral id -> key."""
    lines, ids = [], {}
    for n, e in enumerate(entries, 1):
        sid = f"s{n}"
        ids[sid] = e["key"]
        comment = " ".join((e["comment"] or "").split())
        if len(comment) > COMMENT_CHARS:
            comment = comment[:COMMENT_CHARS] + " …"
        lines += [f"[{sid}] ({e['language']} · {e['file']} · {e['kind']})", f"SENTENCE: {e['sentence']}"]
        if comment and comment != " ".join(e["sentence"].split()):
            lines.append(f"COMMENT: {comment}")
        lines.append("")
    return "\n".join(lines), ids


def load_entries(where: str = "", params: list | None = None) -> list[dict]:
    rows = duckdb.connect().execute(f"""
        SELECT c.id, c.sentence, c.full_comment, c.comment_kind, s.file_path, s.language
        FROM '{pilot.DB / "comment.parquet"}' c JOIN '{pilot.DB / "source_file.parquet"}' s ON s.id = c.source_id
        {where} ORDER BY c.id""", params or []).fetchall()
    return [{"key": cid, "sentence": sent, "comment": full, "kind": kind,
             "file": fp.replace("\\", "/").rsplit("/", 1)[-1], "language": lang}
            for cid, sent, full, kind, fp, lang in rows]


def run_batches(entries: list[dict], out: Path, args) -> set[int]:
    """Screen entries in shuffled batches; append one JSON line per batch to `out` (resumable). Returns flagged keys."""
    rng = random.Random(args.seed)
    entries = entries[:]
    rng.shuffle(entries)
    batches = [entries[i:i + args.batch] for i in range(0, len(entries), args.batch)]
    done, flagged = set(), set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add(r["batch"])
            flagged.update(r["flagged"])
    todo = [b for b in range(len(batches)) if b not in done]
    print(f"{len(entries)} sentences in {len(batches)} batches of {args.batch}; {len(todo)} to run "
          f"with {args.backend} ({args.workers} workers)", flush=True)
    call = pilot.call_claude if args.backend == "claude" else pilot.call_codex
    workdir = tempfile.mkdtemp(prefix="slop_screen_")
    lock, stats, t0 = threading.Lock(), Counter(), time.time()

    def work(b: int) -> None:
        prompt, ids = batch_prompt(batches[b])
        err: Exception | None = None
        for attempt in range(3):
            try:
                r = call(prompt, args.model, args.effort, workdir, instructions=INSTRUCTIONS, schema=SCHEMA)
                keys = sorted({ids[i] for i in r["output"].get("flagged", []) if i in ids})
                with lock:
                    with out.open("a") as f:
                        f.write(json.dumps({"batch": b, "n": len(ids), "flagged": keys, "keys": sorted(ids.values()),
                                            "cost_usd": r["cost_usd"], "backend": args.backend}) + "\n")
                    flagged.update(keys)
                    stats["sentences"] += len(ids)
                    stats["flagged"] += len(keys)
                    stats["cost"] += r["cost_usd"] or 0
                return
            except Exception as e:  # noqa: BLE001 — retry, then report; one bad batch must not stop the run
                err = e
                time.sleep(2 * (attempt + 1))
        with lock:
            stats["failed"] += 1
            print(f"  batch {b} failed: {err}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, b) for b in todo]
        for n, _ in enumerate(as_completed(futs), 1):
            if n % 10 == 0 or n == len(futs):
                with lock:
                    rate = stats["flagged"] / max(1, stats["sentences"])
                    print(f"  {n}/{len(futs)} batches, flagged {stats['flagged']}/{stats['sentences']} "
                          f"({100 * rate:.1f}%), failed {stats['failed']}, ${stats['cost']:.2f}, "
                          f"{time.time() - t0:.0f}s", flush=True)
    return flagged


def cmd_pilot(args) -> None:
    d = pilot.run_dir(args.run)
    items = {int(it["comment_id"]): it for it in pilot.load_sample(d)}
    entries = load_entries("WHERE c.id IN (SELECT unnest(?::BIGINT[]))", [list(items)])
    out = d / f"screen_{args.backend}_b{args.batch}.jsonl"
    flagged = run_batches(entries, out, args)
    by_item = {items[k]["item_id"]: k in flagged for k in items}
    llm = pilot.load_llm(d)
    hand = pilot.load_hand(d).get("you", {})
    lines = [f"# Screening test ({args.backend}, batch {args.batch}) on {len(by_item)} pilot items", ""]
    for stratum in ("random", "marker"):
        ids = [i for i in by_item if i.startswith(stratum[0])]
        lines.append(f"- {stratum}: flagged {sum(by_item[i] for i in ids)}/{len(ids)} "
                     f"({100 * sum(by_item[i] for i in ids) / len(ids):.1f}%)")
    meta = json.loads((d / "sample_meta.json").read_text())
    w = meta["population_marker"] / meta["population_sentences"]
    rnd = [i for i in by_item if i.startswith("r")]
    unmarked = [i for i in rnd if not items[next(k for k in items if items[k]["item_id"] == i)]["marker"]]
    marked = [i for i in by_item if i.startswith("m")]
    fr = (w * sum(by_item[i] for i in marked) / len(marked)
          + (1 - w) * sum(by_item[i] for i in unmarked) / max(1, len(unmarked)))
    lines.append(f"- estimated share of all sentences flagged: {100 * fr:.1f}% "
                 f"(about {round(fr * meta['population_sentences']):,} of {meta['population_sentences']:,} to judge)")
    refs = {f"llm:{k}": [i for i, o in v.items() if o["label"] == "slop"] for k, v in llm.items()}
    refs["human:you"] = [i for i, v in hand.items() if v["label"] == "slop"]
    lines.append("")
    for name, pos in refs.items():
        kept = [i for i in pos if by_item.get(i)]
        lines.append(f"- recall of {name} slop: {len(kept)}/{len(pos)}"
                     + (f"; missed: {', '.join(sorted(set(pos) - set(kept)))}" if len(kept) < len(pos) else ""))
    report = "\n".join(lines)
    (d / f"screen_{args.backend}_b{args.batch}_report.md").write_text(report + "\n")
    print(report)


def cmd_all(args) -> None:
    out_dir = pilot.run_dir(args.out)
    entries = load_entries()
    flagged = run_batches(entries, out_dir / f"screen_{args.backend}_b{args.batch}.jsonl", args)
    print(f"flagged {len(flagged):,} of {len(entries):,} sentences ({100 * len(flagged) / len(entries):.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", choices=["codex", "claude"], default="codex")
    common.add_argument("--model", default=None)
    common.add_argument("--effort", default=None)
    common.add_argument("--batch", type=int, default=40)
    common.add_argument("--workers", type=int, default=4)
    common.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pilot", parents=[common])
    s.add_argument("--run", default="pilot_v1")
    s.set_defaults(func=cmd_pilot)
    s = sub.add_parser("all", parents=[common])
    s.add_argument("--out", default="screen_all")
    s.set_defaults(func=cmd_all)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
