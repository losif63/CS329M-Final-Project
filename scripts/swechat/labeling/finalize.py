"""Combine screen + judge results into final labels for every comment sentence.

    python scripts/swechat/labeling/finalize.py [--screen screen_all] [--judge judge_all] [--tag v1.2]

Writes, in data/swechat/db/:
    label_<tag>.parquet   one row per sentence: comment_id, label, category, rationale, evidence_turns, source,
                          model, rulebook, pilot_item, pilot_stratum, hand_label, hand_category
    comment.parquet       its `label` column filled with the final label (rerunning extract.py resets it to null;
                          rerun this script afterwards)
    label_<tag>_summary.json

`source` says where a label comes from:
    judge            the full judge (session context, rulebook) labeled it
    screen_negative  the screen did not flag it, so it counts as not_slop without a judge call. The pilot measured
                     that the screen misses about 1 in 20 slop sentences, so a few of these are really slop.
Sentences in the pilot sample keep their ids in `pilot_item` so they can be held out of training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", default="screen_all")
    ap.add_argument("--judge", default="judge_all")
    ap.add_argument("--tag", default="v1.2")
    ap.add_argument("--backend", default="claude")
    ap.add_argument("--pilot", default="pilot_v1")
    args = ap.parse_args()
    db, labels = pilot.DB, pilot.LABELS
    con = duckdb.connect()

    screened, flagged = set(), set()
    for p in sorted((labels / args.screen).glob("screen_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            screened.update(r["keys"])
            flagged.update(r["flagged"])
    jitems = {it["item_id"]: int(it["comment_id"]) for it in pilot.load_sample(labels / args.judge)}
    judged = {}
    for line in (labels / args.judge / f"llm_{args.backend}_{args.tag}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if pilot.valid(r.get("output")):
            judged[jitems[r["item_id"]]] = r  # last valid record wins
    missing = sorted(set(jitems.values()) - set(judged))
    if missing:
        sys.exit(f"{len(missing)} flagged sentences have no judge label yet (e.g. {missing[:3]}); finish the judge run")

    pd_dir = labels / args.pilot
    pilot_items = {int(it["comment_id"]): it for it in pilot.load_sample(pd_dir)} if (pd_dir / "sample.parquet").exists() else {}
    hand = {}
    if (pd_dir / "handsheet_key.json").exists():
        for item_id, v in pilot.load_hand(pd_dir).get("you", {}).items():
            cid = next(c for c, it in pilot_items.items() if it["item_id"] == item_id)
            hand[cid] = v

    ids = [r[0] for r in con.execute(f"SELECT id FROM '{db / 'comment.parquet'}' ORDER BY id").fetchall()]
    unscreened = [c for c in ids if c not in screened]
    if unscreened:
        sys.exit(f"{len(unscreened)} sentences were never screened (e.g. {unscreened[:3]}); finish the screen run")
    rows = []
    for cid in ids:
        j = judged.get(cid)
        o = j["output"] if j else None
        p = pilot_items.get(cid)
        h = hand.get(cid)
        rows.append({
            "comment_id": cid,
            "label": o["label"] if o else "not_slop",
            "category": (o["category"] if o and o["label"] == "slop" else None),
            "rationale": o["rationale"] if o else None,
            "evidence_turns": json.dumps(o["evidence_turns"]) if o else None,
            "source": "judge" if j else "screen_negative",
            "model": j["model"] if j else None,
            "rulebook": j["rulebook"] if j else None,
            "pilot_item": p["item_id"] if p else None,
            "pilot_stratum": p["stratum"] if p else None,
            "hand_label": h["label"] if h else None,
            "hand_category": (h["category"] or None) if h else None,
        })
    out = db / f"label_{args.tag}.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), out, compression="zstd")

    # fill comment.parquet's label column (atomic replace)
    lab = pd.DataFrame({"comment_id": [r["comment_id"] for r in rows], "final": [r["label"] for r in rows]})
    con.register("lab", lab)
    tmp = db / "comment.parquet.tmp"
    con.execute(f"""COPY (SELECT c.* EXCLUDE (label), l.final AS label FROM '{db / 'comment.parquet'}' c
                          JOIN lab l ON l.comment_id = c.id ORDER BY c.id) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)""")
    tmp.replace(db / "comment.parquet")

    meta = con.execute(f"""SELECT c.id, s.language, s.tool_name, s.repo_id, s.certified_by
        FROM '{db / 'comment.parquet'}' c JOIN '{db / 'source_file.parquet'}' s ON s.id = c.source_id""").fetchall()
    by = {cid: (lang, tool, repo, cert) for cid, lang, tool, repo, cert in meta}
    slop = [r for r in rows if r["label"] == "slop"]
    def rate(key_idx: int, top: int = 12) -> dict:
        n, k = Counter(), Counter()
        for r in rows:
            g = by[r["comment_id"]][key_idx]
            n[g] += 1
            k[g] += r["label"] == "slop"
        return {g: {"sentences": n[g], "slop": k[g], "rate": round(k[g] / n[g], 4)} for g, _ in n.most_common(top)}
    summary = {
        "sentences": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "slop_rate": round(len(slop) / len(rows), 4),
        "slop_by_category": dict(Counter(r["category"] for r in slop).most_common()),
        "by_language": rate(0),
        "by_tool": rate(1),
        "by_certified_by": rate(3),
        "repos_with_slop": len({by[r["comment_id"]][2] for r in slop}),
        "sessions_with_slop": con.execute(f"""SELECT count(DISTINCT s.session_id) FROM '{db / 'comment.parquet'}' c
            JOIN '{db / 'source_file.parquet'}' s ON s.id = c.source_id WHERE c.label = 'slop'""").fetchall()[0][0],
        "pilot_items": sum(r["pilot_item"] is not None for r in rows),
        "screen": args.screen, "judge": args.judge, "rulebook": args.tag,
    }
    (db / f"label_{args.tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
