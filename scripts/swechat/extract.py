"""Build the SWE-chat slop-comment tables: turn, source_file, comment.

Usage:
    python scripts/swechat/extract.py [--limit-sessions N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comments import drop_reason, extract_comments, new_comments, prose_lines  # noqa: E402
from commit_route import CommitStats  # noqa: E402
from languages import is_code_path, is_scratch_path, language_of  # noqa: E402
from reconstruct import ReconStats, reconstruct_session  # noqa: E402
from sentences import norm_sentence, split_comment  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/swechat/raw"
DB = ROOT / "data/swechat/db"
DROP_TURN_TYPES = ("progress", "file_snapshot")
CHUNK_SESSIONS = 400
DROPPED_SAMPLE_SIZE = 200


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- step 1

def select_sessions(limit: int | None, seed: int, stats: dict) -> pd.DataFrame:
    s = pd.read_parquet(RAW / "sessions.parquet",
                        columns=["session_id", "repo_id", "agent", "kind", "files_touched", "checkpoint_ids"])
    steps = []

    def record(name: str, df: pd.DataFrame) -> None:
        steps.append({"step": name, "sessions": len(df), "repos": int(df["repo_id"].nunique())})
        log(f"  {name}: {len(df)} sessions, {df['repo_id'].nunique()} repos")

    record("all", s)
    s = s[s["agent"] == "Claude Code"]
    record("agent == 'Claude Code'", s)
    s = s[s["kind"].fillna("") == ""]  # drops imported, agent_review, agent_investigate
    record("kind == '' (normal)", s)

    def has_code(ft: str | None) -> bool:
        try:
            return any(isinstance(p, str) and is_code_path(p) for p in json.loads(ft or "[]"))
        except json.JSONDecodeError:
            return False

    s = s[s["files_touched"].map(has_code)]
    record("files_touched has a code file", s)
    s = s.sort_values("session_id").reset_index(drop=True)
    if limit is not None and limit < len(s):
        ids = sorted(random.Random(seed).sample(list(s["session_id"]), limit))
        s = s[s["session_id"].isin(ids)].reset_index(drop=True)
        record(f"random sample (limit={limit}, seed={seed})", s)
    stats["session_filters"] = steps
    return s[["session_id", "repo_id", "checkpoint_ids"]]


# ---------------------------------------------------------------- turn table

def write_turn_table(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    con.execute(f"""
        COPY (
          SELECT c.session_id, c.turn_number, c.turn_type, c.tool_name, c.tool_call_id,
                 c.file_path, c.command, c.content
          FROM read_parquet('{RAW / "conversations.parquet"}') c
          JOIN sel USING (session_id)
          WHERE c.turn_type NOT IN {DROP_TURN_TYPES}
          ORDER BY c.session_id, c.turn_number
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 100000)""")


def load_tool_rows(con: duckdb.DuckDBPyConnection, session_ids: list[str]) -> pd.DataFrame:
    con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT unnest(?::VARCHAR[]) AS session_id", [session_ids])
    df = con.execute(f"""
        SELECT c.session_id, c.turn_number, c.turn_type, c.tool_name, c.tool_call_id,
               c.file_path, c.command, c.tool_input_json, c.content, CAST(c.timestamp AS VARCHAR) AS timestamp
        FROM read_parquet('{RAW / "conversations.parquet"}') c
        JOIN chunk USING (session_id)
        WHERE (c.turn_type IN ('tool_use', 'tool_result') AND c.tool_name IN ('Write', 'Edit', 'MultiEdit', 'Read'))
           OR (c.turn_type = 'tool_use' AND c.tool_name IN ('Bash', 'Task', 'Agent'))
        ORDER BY c.session_id, c.turn_number""").df()
    return df.astype(object).where(df.notna(), None)  # NaN -> None


def load_commit_rows(con: duckdb.DuckDBPyConnection, checkpoint_ids: dict[str, list[str]]) -> dict[str, list[tuple]]:
    """session_id -> linked commits as (commit_date, author_date, patch, agent_changes, file_attribution)."""
    cps = sorted({c for v in checkpoint_ids.values() for c in v})
    if not cps:
        return {}
    con.execute("CREATE OR REPLACE TEMP TABLE cps AS SELECT unnest(?::VARCHAR[]) AS checkpoint_pk", [cps])
    rows = con.execute(f"""
        SELECT checkpoint_pk, CAST(commit_date AS VARCHAR), CAST(author_date AS VARCHAR), patch, agent_changes,
               file_attribution
        FROM read_parquet('{RAW / "commits.parquet"}') JOIN cps USING (checkpoint_pk)
        WHERE status = 'ok' AND agent_changes IS NOT NULL AND agent_changes NOT IN ('', '[]')
        ORDER BY checkpoint_pk, commit_date""").fetchall()
    by_cp: dict[str, list[tuple]] = {}
    for cp, *rest in rows:
        by_cp.setdefault(cp, []).append(tuple(rest))
    return {sid: [r for cp in cpl for r in by_cp.get(cp, [])] for sid, cpl in checkpoint_ids.items()}


# ---------------------------------------------------------------- steps 3-4

class Extractor:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.drop_counts: Counter = Counter()
        self.line_drop_counts: Counter = Counter()
        self.dropped_seen = 0
        self.dropped_sample: list[dict] = []
        self.version_skips: Counter = Counter()  # versions whose comments were not attributable
        self.unattributable_comments = 0
        self._cache: dict[str, tuple] = {}

    def _comments(self, src: str, lang: str):
        hit = self._cache.get((src, lang))
        if hit is None:
            hit = extract_comments(src, lang)
            self._cache[(src, lang)] = hit
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return hit

    def _sample_drop(self, row: dict) -> None:
        self.dropped_seen += 1
        if len(self.dropped_sample) < DROPPED_SAMPLE_SIZE:
            self.dropped_sample.append(row)
        else:
            j = self.rng.randrange(self.dropped_seen)
            if j < DROPPED_SAMPLE_SIZE:
                self.dropped_sample[j] = row

    def sentences_for(self, v) -> list[dict]:
        if v.skip_reason:
            self.version_skips[v.skip_reason] += 1
            return []
        lang = language_of(v.file_path)
        after, fb_after = self._comments(v.content, lang)
        before, fb_before = self._comments(v.before, lang) if v.before is not None else (None, False)
        if fb_after or fb_before:
            # parse errors: comment boundaries (and docstrings) are guesses
            self.version_skips["unparseable_after" if fb_after else "unparseable_before"] += 1
            return []
        added, unsure = new_comments(after, before, v.before, v.content)
        self.unattributable_comments += unsure
        if not added:
            return []
        pre_sents: Counter = Counter()
        if before is not None:
            for c in before:
                if drop_reason(c, lang) is None:
                    pre_sents.update(norm_sentence(s) for s, _, _ in split_comment(prose_lines(c, lang)[0]))
        rows = []
        for c in added:
            if c.near_error:
                self.drop_counts["near_parse_error"] += 1
                continue
            reason = drop_reason(c, lang)
            if reason:
                self.drop_counts[reason] += 1
                self._sample_drop({"session_id": v.session_id, "turn_number": v.turn_number, "language": lang,
                                   "file_path": v.file_path, "comment_kind": c.kind, "reason": reason,
                                   "full_comment": c.text, "line_start": c.line_start})
                continue
            lines, removed = prose_lines(c, lang)
            self.line_drop_counts.update(removed)
            sents = split_comment(lines)
            if not sents:
                self.drop_counts["no_prose_after_line_filter"] += 1
                continue
            for sent, l0, l1 in sents:
                key = norm_sentence(sent)
                if pre_sents.get(key, 0) > 0:
                    pre_sents[key] -= 1
                    self.line_drop_counts["sentence_preexisting_in_before_state"] += 1
                    continue
                rows.append({"sentence": sent, "norm": key, "full_comment": c.text,
                             "comment_kind": c.kind, "line_start": l0, "line_end": l1})
        return rows


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-sessions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DB)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats: dict = {"args": {"limit_sessions": args.limit_sessions, "seed": args.seed}}

    log("step 1: selecting sessions")
    sel = select_sessions(args.limit_sessions, args.seed, stats)
    repo_of = dict(zip(sel["session_id"], sel["repo_id"]))
    cps_of = {sid: json.loads(c or "[]") for sid, c in zip(sel["session_id"], sel["checkpoint_ids"])}
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.register("sel_df", sel[["session_id"]])
    con.execute("CREATE TEMP TABLE sel AS SELECT * FROM sel_df")

    log("writing turn.parquet")
    write_turn_table(con, args.out / "turn.parquet")
    stats["turn_rows"] = con.execute(f"SELECT count(*) FROM '{args.out / 'turn.parquet'}'").fetchall()[0][0]

    recon = ReconStats()
    cstats = CommitStats()
    ext = Extractor(args.seed)
    src_rows: list[dict] = []
    com_rows: list[dict] = []
    sents_per_session: dict[str, int] = {}
    dedup_dropped = 0
    scratch_dropped = 0
    ids = list(sel["session_id"])
    for k in range(0, len(ids), CHUNK_SESSIONS):
        chunk = ids[k : k + CHUNK_SESSIONS]
        log(f"steps 2-4: sessions {k}-{k + len(chunk)} of {len(ids)}")
        df = load_tool_rows(con, chunk)
        commits = load_commit_rows(con, {sid: cps_of.get(sid, []) for sid in chunk})
        for sid, g in df.groupby("session_id", sort=True):
            versions = reconstruct_session(g.to_dict("records"), recon, commits.get(sid), cstats)
            seen: set[tuple[str, str]] = set()
            n = 0
            for v in versions:
                if is_scratch_path(v.file_path):
                    scratch_dropped += 1  # throwaway script in a temp dir, not a repository file
                    continue
                sid_src = len(src_rows)
                src_rows.append({"id": sid_src, "session_id": v.session_id, "turn_number": v.turn_number,
                                 "repo_id": repo_of.get(v.session_id), "file_path": v.file_path,
                                 "language": language_of(v.file_path), "tool_name": v.tool_name,
                                 "content": v.content, "content_flag": v.content_flag,
                                 "certified_by": v.certified_by})
                for r in ext.sentences_for(v):
                    key = (v.file_path, r.pop("norm"))
                    if key in seen:
                        dedup_dropped += 1
                        continue
                    seen.add(key)
                    com_rows.append({"id": len(com_rows), "source_id": sid_src, **r, "label": None})
                    n += 1
            sents_per_session[sid] = n
        for sid in chunk:
            sents_per_session.setdefault(sid, 0)

    log("writing source_file.parquet, comment.parquet")
    src_schema = pa.schema([("id", pa.int64()), ("session_id", pa.string()), ("turn_number", pa.int64()),
                            ("repo_id", pa.string()), ("file_path", pa.string()), ("language", pa.string()),
                            ("tool_name", pa.string()), ("content", pa.string()), ("content_flag", pa.string()),
                            ("certified_by", pa.string())])
    com_schema = pa.schema([("id", pa.int64()), ("source_id", pa.int64()), ("sentence", pa.string()),
                            ("full_comment", pa.string()), ("comment_kind", pa.string()),
                            ("line_start", pa.int64()), ("line_end", pa.int64()), ("label", pa.string())])
    pq.write_table(pa.Table.from_pylist(src_rows, schema=src_schema), args.out / "source_file.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(com_rows, schema=com_schema), args.out / "comment.parquet", compression="zstd")
    drop_schema = pa.schema([("session_id", pa.string()), ("turn_number", pa.int64()), ("language", pa.string()),
                             ("file_path", pa.string()),
                             ("comment_kind", pa.string()), ("reason", pa.string()), ("full_comment", pa.string()),
                             ("line_start", pa.int64())])
    sample = sorted(ext.dropped_sample, key=lambda r: (r["session_id"], r["turn_number"], r["line_start"]))
    pq.write_table(pa.Table.from_pylist(sample, schema=drop_schema), args.out / "dropped_sample.parquet")

    counts = sorted(sents_per_session.values())
    stats.update({
        "reconstruction": {
            "code_write_tool_uses": recon.tool_uses,
            "no_paired_result": recon.no_result,
            "failed_result": recon.failed,
            "unparseable_input": recon.bad_input,
            "versions_dropped_no_reliable_full_file": recon.unreliable,
            "tracked_state_discarded": recon.state_resets,
            "versions_retracted_after_drift": recon.retracted,
            "versions_dropped_scratch_file": scratch_dropped,
            "reliable_versions": len(src_rows),
        },
        "commit_route": {"commits": cstats.commits, "files": cstats.files,
                         **dict(sorted(cstats.reasons.items(), key=lambda kv: -kv[1]))},
        "certified_by": dict(Counter(r["certified_by"] for r in src_rows).most_common()),
        "versions_by_language": dict(Counter(r["language"] for r in src_rows).most_common()),
        "sentences_by_language": dict(Counter(src_rows[r["source_id"]]["language"] for r in com_rows).most_common()),
        "sentences_by_certified_by": dict(Counter(src_rows[r["source_id"]]["certified_by"] for r in com_rows).most_common()),
        "versions_with_comments_skipped": dict(ext.version_skips.most_common()),
        "comments_skipped_unattributable_occurrence": ext.unattributable_comments,
        "content_flag": dict(Counter(r["content_flag"] for r in src_rows).most_common()),
        "tool_name": dict(Counter(r["tool_name"] for r in src_rows).most_common()),
        "comments_dropped_by_reason": dict(ext.drop_counts.most_common()),
        "lines_or_sentences_removed_within_kept_comments": dict(ext.line_drop_counts.most_common()),
        "sentences_dropped_duplicate_in_session": dedup_dropped,
        "sentences": len(com_rows),
        "sentences_by_kind": dict(Counter(r["comment_kind"] for r in com_rows).most_common()),
        "sentences_by_content_flag": dict(Counter(src_rows[r["source_id"]]["content_flag"] for r in com_rows).most_common()),
        "sentences_per_session": {
            "sessions": len(counts),
            "sessions_with_zero": sum(c == 0 for c in counts),
            "mean": round(statistics.mean(counts), 2) if counts else 0,
            "median": statistics.median(counts) if counts else 0,
            "p90": counts[int(0.9 * (len(counts) - 1))] if counts else 0,
            "max": counts[-1] if counts else 0,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    shutil.copy(Path(__file__).resolve().parent / "SCHEMA.md", args.out / "README.md")
    log(f"done: {len(src_rows)} source versions, {len(com_rows)} sentences in {stats['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
