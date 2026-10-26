"""Export the db tables to a static, sharded bundle for the comment viewer (index.html).

    index.js      window.SWECHAT_INDEX = {sessions: [session_id...], stats,
                    comments: [[comment_id, session_idx, language, kind, label, marker_hit], ...]}
    s/<k>.js      window.SWECHAT_SHARD(k, {comments: {comment_id: {...}}, sources: {...}, turns: [...]})

One shard per session, loaded on demand by the page (script tags, so it also works from disk).
A comment's conversation is turns[turn_end - turn_count : turn_end] of its session shard: the last
`max_turns` turns before the write that produced its source version. Shards only keep turns inside
some comment's window; `turns_before_window` counts the earlier ones.

Usage: python scripts/swechat/viewer/export.py [--db DIR] [--out DIR] [--max-turns 50] [--max-turn-chars 1500]
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data/swechat/db"
HERE = Path(__file__).resolve().parent
TURN_TYPES = {"user_prompt", "system_injected", "assistant_response", "assistant_thinking",
              "tool_use", "tool_result", "summary"}
LABEL_CHARS = 90
# surface markers of session context (a rough proxy for browsing, not a label)
MARKER_RE = re.compile("|".join([
    r"\b(now|no longer|previously|used to|instead of|originally|anymore|changed (to|from)|updated to|switched (to|from)"
    r"|the old|the new (approach|logic|version|behavior|implementation)|unlike before|this (fix|change)"
    r"|(was|were) (removed|replaced|changed))\b",
    r"\b(as requested|per (the )?(user|your|request)|requested by|the user (asked|wants|requested)|you asked"
    r"|as discussed|as mentioned|we (discussed|agreed|decided)|per our|as (you|we) )\b",
    r"(/Users/|/home/\w|on (my|this) machine|\blocally\b|in this (environment|session|run)|the (test )?run (above|earlier)"
    r"|output above|tested (it|this|locally)|verified (it|this|that))",
    r"\b(fix(es|ed)? (the|a|an) (bug|issue|error|crash|problem)|bug fix|workaround for the|to avoid the (bug|issue|error))\b",
]), re.I)


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
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--max-turn-chars", type=int, default=1500)
    args = ap.parse_args()
    db, con = args.db, duckdb.connect()

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
                               "span": list(span) if span else None})
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
        nums = [t[0] for t in turns]
        keep: set[int] = set()
        ends = {}
        for c in cs:
            end = bisect.bisect_left(nums, sources[c["source_id"]]["turn_number"])
            ends[c["id"]] = end
            keep.update(range(max(0, end - args.max_turns), end))
        kept_idx = sorted(keep)
        shard_comments = {}
        for c in cs:
            end = ends[c["id"]]
            n = min(args.max_turns, end)
            shard_comments[c["id"]] = {
                "source_id": c["source_id"], "kind": c["kind"], "full_comment": c["full_comment"],
                "sentences": c["sentences"], "turn_end": bisect.bisect_left(kept_idx, end),
                "turn_count": n, "turns_before_window": end - n}
            n_unplaced += sum(x["span"] is None for x in c["sentences"])
        shard = {"comments": shard_comments,
                 "sources": {str(c["source_id"]): sources[c["source_id"]] for c in cs},
                 "turns": [turns[j] for j in kept_idx]}
        k = sidx[sid]
        (args.out / "s" / f"{k}.js").write_text(f"window.SWECHAT_SHARD({k}," + js_payload(shard) + ");\n")
        pending.discard(sid)

    def clip(text: str | None) -> str:
        text = text or ""
        if len(text) > args.max_turn_chars:
            return text[:args.max_turn_chars] + f"\n…[truncated {len(text) - args.max_turn_chars} chars]"
        return text

    cur_sid, cur_turns = None, []
    pf = pq.ParquetFile(db / "turn.parquet")
    for batch in pf.iter_batches(batch_size=50_000, columns=["session_id", "turn_number", "turn_type", "tool_name", "content"]):
        cols = batch.to_pydict()
        for sid, tn, tt, tool, content in zip(cols["session_id"], cols["turn_number"], cols["turn_type"],
                                             cols["tool_name"], cols["content"]):
            if sid != cur_sid:
                if cur_sid in pending:
                    flush(cur_sid, cur_turns)
                cur_sid, cur_turns = sid, []
            if sid in pending and tt in TURN_TYPES:
                cur_turns.append([tn, tt, tool, clip(content)])
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
        index.append([c["id"], sidx[s["session_id"]], s["language"] or "?", c["kind"], label, marker])
    stats = json.loads((db / "stats.json").read_text())
    (args.out / "index.js").write_text("window.SWECHAT_INDEX = " + js_payload({
        "sessions": session_ids, "max_turns": args.max_turns, "comments": index,
        "n_sentences": sum(len(c["sentences"]) for c in ordered), "stats": stats}) + ";\n")
    shutil.copy(HERE / "index.html", args.out / "index.html")
    shutil.copy(HERE / "space_README.md", args.out / "README.md")
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e6
    print(f"{len(ordered)} comments, {sum(len(c['sentences']) for c in ordered)} sentences ({n_unplaced} not located), "
          f"{len(sources)} sources, {len(session_ids)} session shards, index.js "
          f"{(args.out / 'index.js').stat().st_size / 1e6:.1f} MB, total {size:.0f} MB -> {args.out}")


if __name__ == "__main__":
    main()
