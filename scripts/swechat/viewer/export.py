"""Export the db tables to a static, sharded bundle for the comment viewer (index.html).

    index.js      window.SWECHAT_INDEX = {sessions: [session_id...], stats,
                    comments: [[comment_id, session_idx, language, kind, label, marker_hit], ...]}
    s/<k>.js      window.SWECHAT_SHARD(k, {comments: {comment_id: {...}}, sources: {...}, turns: [...]})

One shard per session, loaded on demand by the page (script tags, so it also works from disk).
Each shard holds the session's compact conversation (conversation.build_items: text turns in full,
agent work as one-line stubs). A comment's conversation is items[:item_end], with a work stub that
spans its write turn clipped to the part before it, so every comment sees its whole session prefix.

Usage: python scripts/swechat/viewer/export.py [--db DIR] [--out DIR] [--max-text-chars 20000]
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import sys

import duckdb
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversation import build_items  # noqa: E402
from markers import MARKER_RE  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data/swechat/db"
HERE = Path(__file__).resolve().parent
LABEL_CHARS = 90


def sentence_span(full: str, sent: str, start: int) -> tuple[int, int] | None:
    """Locate a sentence inside the full comment text, tolerating re-wrapped whitespace."""
    words = sent.split()
    if not words:
        return None
    pat = re.compile(r"\s+".join(re.escape(w) for w in words))
    m = pat.search(full, start) or pat.search(full)
    return (m.start(), m.end()) if m else None


def js_payload(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB, help="directory with the parquet tables")
    ap.add_argument("--out", type=Path, default=ROOT / "data/swechat/viewer_site")  # gitignored
    ap.add_argument("--max-text-chars", type=int, default=20000,
                    help="cap for a single text turn in the viewer (pasted logs can be huge)")
    ap.add_argument("--labels", type=Path, default=None,
                    help="label table from labeling/finalize.py (default: the newest db/label_*.parquet, if any)")
    args = ap.parse_args()
    db, con = args.db, duckdb.connect()
    label_path = args.labels or max(db.glob("label_v*.parquet"), default=None, key=lambda p: p.stat().st_mtime)
    labels: dict[int, dict] = {}
    if label_path:
        for cid, lab, cat, why, src in con.execute(f"""
                SELECT comment_id, label, category, rationale, source FROM '{label_path}'""").fetchall():
            labels[cid] = {"lab": lab, "cat": cat, "why": why, "src": src}
        print(f"labels: {label_path.name} ({sum(v['lab'] == 'slop' for v in labels.values())} slop)")

    # ---- comments (grouped from sentences) and their sources
    sources = {}
    for r in con.execute(f"""
            SELECT id, session_id, turn_number, repo_id, file_path, language, tool_name, content_flag, certified_by, content
            FROM '{db / "source_file.parquet"}'
            WHERE id IN (SELECT DISTINCT source_id FROM '{db / "comment.parquet"}')""").fetchall():
        sources[r[0]] = dict(zip(["session_id", "turn_number", "repo_id", "file_path", "language", "tool_name",
                                  "content_flag", "certified_by", "content"], r[1:]))
    comments: dict[tuple, dict] = {}
    for cid, sid, sent, full, kind, l0, l1 in con.execute(f"""
            SELECT id, source_id, sentence, full_comment, comment_kind, line_start, line_end
            FROM '{db / "comment.parquet"}' ORDER BY id""").fetchall():
        c = comments.setdefault((sid, full, kind), {"source_id": sid, "kind": kind, "full_comment": full,
                                                     "sentences": [], "_cursor": 0})
        span = sentence_span(full, sent, c["_cursor"])
        if span:
            c["_cursor"] = span[1]
        c["sentences"].append({"id": cid, "text": sent, "line_start": l0, "line_end": l1,
                               "span": list(span) if span else None, **labels.get(cid, {})})
    by_session: dict[str, list[dict]] = defaultdict(list)
    for c in comments.values():
        c.pop("_cursor")
        by_session[sources[c["source_id"]]["session_id"]].append(c)
    session_ids = sorted(by_session)
    sidx = {s: i for i, s in enumerate(session_ids)}

    # stable comment ids: session order, then write order, then position in file
    ordered = sorted(comments.values(), key=lambda c: (sources[c["source_id"]]["session_id"],
                                                       sources[c["source_id"]]["turn_number"],
                                                       c["sentences"][0]["line_start"]))
    for i, c in enumerate(ordered):
        c["id"] = i

    # ---- stream turns once (turn.parquet is sorted by session) and write one shard per session
    shutil.rmtree(args.out / "s", ignore_errors=True)
    for stale in ("data.js",):
        (args.out / stale).unlink(missing_ok=True)
    (args.out / "s").mkdir(parents=True, exist_ok=True)
    pending = set(session_ids)
    n_unplaced = 0

    def flush(sid: str, turns: list) -> None:
        nonlocal n_unplaced
        cs = by_session[sid]
        last_write = max(sources[c["source_id"]]["turn_number"] for c in cs)
        items = build_items([t for t in turns if t[0] < last_write])
        for it in items:
            if "x" in it and len(it["x"]) > args.max_text_chars:
                it["x"] = it["x"][:args.max_text_chars] + f"\n…[truncated {len(it['x']) - args.max_text_chars} chars]"
        starts = [it["t"] for it in items]
        shard_comments = {}
        for c in cs:
            write = sources[c["source_id"]]["turn_number"]
            shard_comments[c["id"]] = {
                "source_id": c["source_id"], "kind": c["kind"], "full_comment": c["full_comment"],
                "sentences": c["sentences"], "item_end": bisect.bisect_left(starts, write)}
            n_unplaced += sum(x["span"] is None for x in c["sentences"])
        shard = {"comments": shard_comments,
                 "sources": {str(c["source_id"]): sources[c["source_id"]] for c in cs},
                 "items": items}
        k = sidx[sid]
        (args.out / "s" / f"{k}.js").write_text(f"window.SWECHAT_SHARD({k}," + js_payload(shard) + ");\n")
        pending.discard(sid)

    cur_sid, cur_turns = None, []
    pf = pq.ParquetFile(db / "turn.parquet")
    cols_needed = ["session_id", "turn_number", "turn_type", "tool_name", "content", "file_path", "command"]
    for batch in pf.iter_batches(batch_size=50_000, columns=cols_needed):
        cols = batch.to_pydict()
        for sid, tn, tt, tool, content, fpath, cmd in zip(*(cols[c] for c in cols_needed)):
            if sid != cur_sid:
                if cur_sid in pending:
                    flush(cur_sid, cur_turns)
                cur_sid, cur_turns = sid, []
            if sid in pending:
                # tool results are never shown: drop their content early
                cur_turns.append((tn, tt, tool, None if tt == "tool_result" else content, fpath, cmd))
    if cur_sid in pending:
        flush(cur_sid, cur_turns)
    for sid in list(pending):  # sessions with no turn rows at all
        flush(sid, [])

    # ---- index
    index = []
    for c in ordered:
        s = sources[c["source_id"]]
        first = c["sentences"][0]
        text = re.sub(r"\s+", " ", c["full_comment"])[:LABEL_CHARS]
        label = f"{s['file_path'].rsplit('/', 1)[-1]}:{first['line_start']} · {text}"
        marker = int(any(MARKER_RE.search(x["text"]) for x in c["sentences"]))
        slop_cats = sorted({x.get("cat") or "?" for x in c["sentences"] if x.get("lab") == "slop"})
        index.append([c["id"], sidx[s["session_id"]], s["language"] or "?", c["kind"], label, marker, ",".join(slop_cats)])
    stats = json.loads((db / "stats.json").read_text())
    (args.out / "index.js").write_text("window.SWECHAT_INDEX = " + js_payload({
        "sessions": session_ids, "comments": index,
        "n_sentences": sum(len(c["sentences"]) for c in ordered), "stats": stats}) + ";\n")
    shutil.copy(HERE / "index.html", args.out / "index.html")
    shutil.copy(HERE / "space_README.md", args.out / "README.md")
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e6
    print(f"{len(ordered)} comments, {sum(len(c['sentences']) for c in ordered)} sentences ({n_unplaced} not located), "
          f"{len(sources)} sources, {len(session_ids)} session shards, index.js "
          f"{(args.out / 'index.js').stat().st_size / 1e6:.1f} MB, total {size:.0f} MB -> {args.out}")


if __name__ == "__main__":
    main()
