"""SWE-chat slop-labeling pilot: sample sentences, label them with an LLM CLI, compare with hand labels.

    python scripts/swechat/labeling/pilot.py sample  [--random 200] [--marker 200] [--seed 0] [--run pilot_v1]
    python scripts/swechat/labeling/pilot.py prompts [--run pilot_v1]
    python scripts/swechat/labeling/pilot.py label   [--backend claude|codex] [--model M] [--workers 4] [--limit N]
    python scripts/swechat/labeling/pilot.py handsheet [--n 100]
    python scripts/swechat/labeling/pilot.py report
    python scripts/swechat/labeling/pilot.py serve [--port 8000]   # local web UI for the hand labels

Outputs go to data/swechat/labels/<run>/ (gitignored):
    sample.parquet       items: item_id, comment_id, stratum (random | marker), marker, language, ...
    prompts/<item>.md    exactly what the labeler sees (rulebook is sent as the system prompt)
    llm_<backend>.jsonl  one JSON line per labeled item (resumable: done items are skipped)
    handsheet.csv        blind sheet for human labels (no stratum, no LLM label)
    report.md / .json    prevalence with confidence intervals, proxy precision, agreement

Strata: `random` is a uniform sample of all sentences (unbiased prevalence). `marker` samples sentences that match
the keyword proxy (markers.py), at most 2 per session, to measure the proxy's precision and find positives faster.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from example import build_example  # noqa: E402
from markers import MARKER_RE  # noqa: E402

ROOT = HERE.parents[2]
DB = ROOT / "data/swechat/db"
LABELS = ROOT / "data/swechat/labels"
RULEBOOK = HERE / "rulebook.md"
SPACE = "https://huggingface.co/spaces/CS329M/Data-Visualization"
CATEGORIES = ["conversation", "session_history", "local_environment", "unverifiable_result", "plan_reference",
              "other_session", "none"]
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["slop", "not_slop", "unsure"]},
        "category": {"type": "string", "enum": CATEGORIES},
        "evidence_turns": {"type": "array", "items": {"type": "integer"}},
        "rationale": {"type": "string"},
    },
    "required": ["label", "category", "evidence_turns", "rationale"],
    "additionalProperties": False,
}
def rulebook_version(path: Path = RULEBOOK) -> str:
    m = re.match(r"# .*\((v[\d.]+)", path.read_text())
    return m.group(1) if m else "unknown"


SOURCE_WINDOW = 60          # lines of code shown above and below the target
MAX_CONVERSATION = 120_000  # characters (~30k tokens); longer sessions keep the most recent context


def run_dir(name: str) -> Path:
    d = LABELS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_sample(d: Path) -> list[dict]:
    p = d / "sample.parquet"
    if not p.exists():
        sys.exit(f"{p} not found; run `pilot.py sample` first")
    return duckdb.connect().execute(f"SELECT * FROM '{p}' ORDER BY item_id").df().to_dict("records")


# ---------------------------------------------------------------- sample

def cmd_sample(args) -> None:
    d = run_dir(args.run)
    rows = duckdb.connect().execute(f"""
        SELECT c.id AS comment_id, c.sentence, c.comment_kind, s.session_id, s.repo_id, s.language,
               s.tool_name, s.certified_by, s.file_path
        FROM '{DB / "comment.parquet"}' c JOIN '{DB / "source_file.parquet"}' s ON s.id = c.source_id
        ORDER BY c.id""").df().to_dict("records")
    for r in rows:
        r["marker"] = bool(MARKER_RE.search(r["sentence"]))
    rng = random.Random(args.seed)
    rand = rng.sample(rows, args.random)
    taken = {r["comment_id"] for r in rand}
    pool = [r for r in rows if r["marker"] and r["comment_id"] not in taken]
    rng.shuffle(pool)
    per_session, mark = Counter(), []
    for r in pool:
        if len(mark) == args.marker:
            break
        if per_session[r["session_id"]] < 2:
            per_session[r["session_id"]] += 1
            mark.append(r)
    items = ([{**r, "stratum": "random", "item_id": f"r{i:04d}"} for i, r in enumerate(rand)]
             + [{**r, "stratum": "marker", "item_id": f"m{i:04d}"} for i, r in enumerate(mark)])
    con = duckdb.connect()
    con.register("items", pd.DataFrame(items))
    con.execute(f"COPY (SELECT * FROM items ORDER BY item_id) TO '{d / 'sample.parquet'}' (FORMAT parquet)")
    meta = {"seed": args.seed, "population_sentences": len(rows),
            "population_marker": sum(r["marker"] for r in rows), "random": len(rand), "marker": len(mark)}
    (d / "sample_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"sampled {len(rand)} random + {len(mark)} marker items -> {d / 'sample.parquet'}")
    print(f"population: {meta['population_sentences']} sentences, {meta['population_marker']} marker hits "
          f"({100 * meta['population_marker'] / meta['population_sentences']:.2f}%)")


def cmd_from_screen(args) -> None:
    """Make a judging run from a screen run: every flagged sentence becomes an item (stratum `screened`)."""
    src = LABELS / args.screen
    flagged, seen = set(), set()
    for p in sorted(src.glob("screen_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            flagged.update(r["flagged"])
            seen.update(r["keys"])
    if not seen:
        sys.exit(f"no screen results in {src}")
    d = run_dir(args.run)
    if (d / "sample.parquet").exists() and not args.force:
        sys.exit(f"{d / 'sample.parquet'} exists; pass --force to overwrite")
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE f AS SELECT unnest(?::BIGINT[]) AS comment_id", [sorted(flagged)])
    rows = con.execute(f"""
        SELECT c.id AS comment_id, c.sentence, c.comment_kind, s.session_id, s.repo_id, s.language,
               s.tool_name, s.certified_by, s.file_path
        FROM f JOIN '{DB / "comment.parquet"}' c ON c.id = f.comment_id
        JOIN '{DB / "source_file.parquet"}' s ON s.id = c.source_id ORDER BY c.id""").df().to_dict("records")
    for n, r in enumerate(rows):
        r["marker"] = bool(MARKER_RE.search(r["sentence"]))
        r["stratum"] = "screened"
        r["item_id"] = f"s{n:06d}"
    con.register("items", pd.DataFrame(rows))
    con.execute(f"COPY (SELECT * FROM items ORDER BY item_id) TO '{d / 'sample.parquet'}' (FORMAT parquet)")
    meta = {"screen": args.screen, "screened_sentences": len(seen), "flagged": len(rows)}
    (d / "sample_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(rows):,} flagged of {len(seen):,} screened sentences -> {d / 'sample.parquet'}")


# ---------------------------------------------------------------- prompts

def source_excerpt(content: str, l0: int, l1: int, window: int = SOURCE_WINDOW) -> str:
    lines = content.split("\n")
    a, b = max(1, l0 - window), min(len(lines), l1 + window)
    out = [f"(lines {a}–{b} of {len(lines)})"] if (a > 1 or b < len(lines)) else []
    out += [f"{'>>' if l0 <= i <= l1 else '  '} {i:5d} | {lines[i - 1]}" for i in range(a, b + 1)]
    return "\n".join(out)


def build_prompt(item: dict, con: duckdb.DuckDBPyConnection) -> str:
    sent, src, conv = build_example(int(item["comment_id"]), max_total_chars=MAX_CONVERSATION, con=con,
                                    show_turns=True)
    full, l0, l1, write_turn, tool = con.execute(f"""
        SELECT c.full_comment, c.line_start, c.line_end, s.turn_number, s.tool_name
        FROM '{DB / "comment.parquet"}' c JOIN '{DB / "source_file.parquet"}' s ON s.id = c.source_id
        WHERE c.id = ?""", [int(item["comment_id"])]).fetchall()[0]
    return "\n".join([
        "# Target sentence",
        sent,
        "",
        "# Full comment",
        full,
        "",
        f"# Source file after the agent's {tool} at turn {write_turn}: {item['file_path']} ({item['language']})",
        "```",
        source_excerpt(src, l0, l1),
        "```",
        "",
        "# Conversation before the write",
        conv or "(no conversation before this write)",
        "",
        "Label the target sentence according to the rulebook and return the JSON object.",
    ])


def cmd_prompts(args) -> None:
    d = run_dir(args.run)
    out = d / "prompts"
    out.mkdir(exist_ok=True)
    con = duckdb.connect()
    items = load_sample(d)
    total = 0
    for i, it in enumerate(items, 1):
        p = out / f"{it['item_id']}.md"
        if not p.exists():
            p.write_text(build_prompt(it, con))
        total += p.stat().st_size
        if i % 50 == 0:
            print(f"  {i}/{len(items)} prompts", flush=True)
    rb = RULEBOOK.stat().st_size
    tokens = (total + rb * len(items)) / 4
    print(f"{len(items)} prompts in {out}; about {tokens / 1e6:.1f}M input tokens in total "
          f"({tokens / len(items) / 1e3:.0f}k per item, rulebook included)")


# ---------------------------------------------------------------- label

def call_claude(prompt: str, model: str | None, effort: str | None, workdir: str,
                instructions: str | None = None, schema: dict | None = None) -> dict:
    cmd = ["claude", "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
           "--json-schema", json.dumps(schema or SCHEMA), "--system-prompt", instructions or RULEBOOK.read_text()]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd=workdir)
    if p.returncode:
        raise RuntimeError(f"claude exited {p.returncode}: {p.stderr[-500:] or p.stdout[-500:]}")
    d = json.loads(p.stdout)
    if d.get("is_error") or not isinstance(d.get("structured_output"), dict):
        raise RuntimeError(f"claude returned no structured output: {str(d.get('result'))[:300]}")
    return {"output": d["structured_output"], "cost_usd": d.get("total_cost_usd"),
            "model": ",".join(d.get("modelUsage", {})) or model}


def call_codex(prompt: str, model: str | None, effort: str | None, workdir: str,
               instructions: str | None = None, schema: dict | None = None) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".schema.json", dir=workdir, delete=False) as f:
        f.write(json.dumps(schema or SCHEMA))
        schema_path = Path(f.name)
    with tempfile.NamedTemporaryFile("r", suffix=".json", dir=workdir, delete=False) as f:
        out = Path(f.name)
    cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
           "--output-schema", str(schema_path), "-o", str(out)]
    if model:
        cmd += ["-m", model]
    if effort:
        cmd += ["-c", f"model_reasoning_effort={effort}"]
    cmd.append("-")
    full = (instructions or RULEBOOK.read_text()) + "\n\n---\n\n" + prompt
    p = subprocess.run(cmd, input=full, capture_output=True, text=True, timeout=900, cwd=workdir)
    try:
        if p.returncode:
            raise RuntimeError(f"codex exited {p.returncode}: {p.stderr[-500:]}")
        return {"output": json.loads(out.read_text()), "cost_usd": None, "model": model or "codex-default"}
    finally:
        out.unlink(missing_ok=True)
        schema_path.unlink(missing_ok=True)


def valid(o: dict) -> bool:
    return (isinstance(o, dict) and o.get("label") in ("slop", "not_slop", "unsure")
            and o.get("category") in CATEGORIES)


def cmd_label(args) -> None:
    d = run_dir(args.run)
    items = {it["item_id"]: it for it in load_sample(d)}
    prompts = d / "prompts"
    out = d / f"llm_{args.backend}{'_' + args.tag if args.tag else ''}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if valid(r.get("output")):
                done.add(r["item_id"])
    todo = [i for i in sorted(items) if i not in done and (prompts / f"{i}.md").exists()]
    missing = [i for i in items if not (prompts / f"{i}.md").exists()]
    if missing:
        print(f"warning: {len(missing)} items have no prompt yet; run `pilot.py prompts`")
    if args.only:
        todo = [i for i in todo if i in set(args.only.split(","))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} already labeled, {len(todo)} to label with {args.backend} ({args.workers} workers)", flush=True)
    call = call_claude if args.backend == "claude" else call_codex
    lock = threading.Lock()
    workdir = tempfile.mkdtemp(prefix="slop_label_")  # empty dir: no project instructions get loaded
    stats = Counter()
    t0 = time.time()

    def work(item_id: str) -> None:
        prompt = (prompts / f"{item_id}.md").read_text()
        err = None
        for attempt in range(1 + args.retries):
            try:
                t = time.time()
                r = call(prompt, args.model, args.effort, workdir)
                if not valid(r["output"]):
                    raise RuntimeError(f"invalid output: {r['output']}")
                rec = {"item_id": item_id, "comment_id": int(items[item_id]["comment_id"]), "backend": args.backend,
                       **r, "seconds": round(time.time() - t, 1), "rulebook": rulebook_version()}
                with lock:
                    with out.open("a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats[r["output"]["label"]] += 1
                    stats["cost"] += r["cost_usd"] or 0
                return
            except Exception as e:  # noqa: BLE001 — record and retry; one bad item must not stop the run
                err = e
                time.sleep(2 * (attempt + 1))
        with lock:
            stats["failed"] += 1
            print(f"  {item_id} failed: {err}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, i) for i in todo]
        for n, _ in enumerate(as_completed(futs), 1):
            if n % 10 == 0 or n == len(futs):
                with lock:
                    print(f"  {n}/{len(futs)}  slop={stats['slop']} not_slop={stats['not_slop']} "
                          f"unsure={stats['unsure']} failed={stats['failed']}  "
                          f"${stats['cost']:.2f}  {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- hand-label sheet

def viewer_ids(db: Path = DB) -> dict[int, tuple[int, int]]:
    """sentence id -> (viewer comment id, sentence index), mirroring viewer/export.py's grouping and order."""
    rows = duckdb.connect().execute(f"""
        SELECT c.id, c.source_id, c.full_comment, c.comment_kind, c.line_start, s.session_id, s.turn_number
        FROM '{db / "comment.parquet"}' c JOIN '{db / "source_file.parquet"}' s ON s.id = c.source_id
        ORDER BY c.id""").fetchall()
    groups: dict[tuple, dict] = {}
    for cid, src, full, kind, l0, sess, turn in rows:
        g = groups.setdefault((src, full, kind), {"ids": [], "key": (sess, turn, l0)})
        g["ids"].append(cid)
    out = {}
    for vid, g in enumerate(sorted(groups.values(), key=lambda g: g["key"])):
        for k, cid in enumerate(g["ids"]):
            out[cid] = (vid, k)
    return out


def cmd_handsheet(args) -> None:
    d = run_dir(args.run)
    items = load_sample(d)
    rng = random.Random(args.seed + 1)
    by = defaultdict(list)
    for it in items:
        by[it["stratum"]].append(it)
    pick = []
    for _stratum, its in sorted(by.items()):
        pick += rng.sample(its, min(len(its), args.n // len(by)))
    rng.shuffle(pick)  # blind: strata are mixed and not shown
    vids = viewer_ids()
    p = d / "handsheet.csv"
    if p.exists() and not args.force:
        sys.exit(f"{p} exists (it may hold your labels); pass --force to overwrite")
    # opaque ids so the sheet doesn't reveal the stratum (item ids start with r/m)
    ctx = d / "handsheet_contexts"
    ctx.mkdir(exist_ok=True)
    key = {}
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hand_id", "sentence", "file", "context_file", "viewer_link", "label", "category", "notes"])
        for n, it in enumerate(pick, 1):
            hid = f"h{n:03d}"
            key[hid] = it["item_id"]
            (ctx / f"{hid}.md").write_text((d / "prompts" / f"{it['item_id']}.md").read_text())
            vid, k = vids[int(it["comment_id"])]
            w.writerow([hid, it["sentence"], it["file_path"].replace("\\", "/").rsplit("/", 1)[-1],
                        f"handsheet_contexts/{hid}.md", f"{SPACE} (#c={vid}&s={k})", "", "", ""])
    (d / "handsheet_key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"{len(pick)} items -> {p}\nlabel (required): slop / not_slop / unsure\n"
          f"category (optional, for slop): {', '.join(c for c in CATEGORIES if c != 'none')}\n"
          f"(handsheet_key.json maps hand ids to items; don't open it before labeling)")


# ---------------------------------------------------------------- report

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, mid - half), min(1.0, mid + half)


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def load_llm(d: Path) -> dict[str, dict[str, dict]]:
    out = {}
    for p in sorted(d.glob("llm_*.jsonl")):
        recs = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if valid(r.get("output")):
                recs[r["item_id"]] = r["output"]  # last valid record wins
        out[p.stem[4:]] = recs
    return out


def _norm(v: str | None) -> str:
    return (v or "").strip().lower().replace("-", "_").replace(" ", "_")


def load_hand(d: Path) -> dict[str, dict[str, dict[str, str]]]:
    """rater -> item_id -> {"label", "category"}. One labeler uses `label`/`category`; the older two-labeler
    sheet's `label_a`/`label_b` columns are still read."""
    p = d / "handsheet.csv"
    if not p.exists():
        return {}
    key = json.loads((d / "handsheet_key.json").read_text())
    out: dict = defaultdict(dict)
    with p.open() as f:
        for r in csv.DictReader(f):
            for who, suffix in (("you", ""), ("a", "_a"), ("b", "_b")):
                lab = _norm(r.get(f"label{suffix}"))
                if lab in ("slop", "not_slop", "unsure"):
                    cat = _norm(r.get(f"category{suffix}"))
                    out[who][key[r["hand_id"]]] = {"label": lab, "category": cat if cat in CATEGORIES else ""}
    return dict(out)


def pct(p: tuple[float, float, float]) -> str:
    return f"{100 * p[0]:.1f}% [{100 * p[1]:.1f}, {100 * p[2]:.1f}]"


def cmd_report(args) -> None:
    d = run_dir(args.run)
    items = {it["item_id"]: it for it in load_sample(d)}
    meta = json.loads((d / "sample_meta.json").read_text())
    llm, hand = load_llm(d), load_hand(d) if (d / "handsheet_key.json").exists() else {}
    N, M = meta.get("population_sentences", 0), meta.get("population_marker", 0)
    md, js = [f"# Labeling pilot `{args.run}`", ""], {"meta": meta, "backends": {}}
    for backend, recs in llm.items():
        md += [f"## {backend}: {len(recs)} of {len(items)} items labeled", ""]
        b: dict = {"labeled": len(recs)}
        for stratum in sorted({it["stratum"] for it in items.values()}):
            ids = [i for i in recs if items[i]["stratum"] == stratum]
            labs = Counter(recs[i]["label"] for i in ids)
            decided = labs["slop"] + labs["not_slop"]
            r = wilson(labs["slop"], decided)
            b[stratum] = {"n": len(ids), **labs, "slop_rate_decided": r}
            md.append(f"- **{stratum}** (n={len(ids)}): slop {labs['slop']}, not_slop {labs['not_slop']}, "
                      f"unsure {labs['unsure']}; slop rate among decided {pct(r)}")
        # stratified prevalence: P(slop) = P(slop|marker) P(marker) + P(slop|no marker) P(no marker)
        rnd_nm = [i for i in recs if items[i]["stratum"] == "random" and not items[i]["marker"]]
        all_m = [i for i in recs if items[i]["marker"]]
        k_nm = sum(recs[i]["label"] == "slop" for i in rnd_nm)
        n_nm = sum(recs[i]["label"] in ("slop", "not_slop") for i in rnd_nm)
        k_m = sum(recs[i]["label"] == "slop" for i in all_m)
        n_m = sum(recs[i]["label"] in ("slop", "not_slop") for i in all_m)
        if n_nm and n_m and N:
            p_nm, p_m = wilson(k_nm, n_nm), wilson(k_m, n_m)
            w = M / N
            est = tuple(w * p_m[j] + (1 - w) * p_nm[j] for j in range(3))
            b["prevalence"] = {"estimate": est, "implied_slop_sentences": [round(x * N) for x in est],
                               "proxy_precision": p_m, "slop_rate_without_marker": p_nm,
                               "proxy_recall": (w * p_m[0] / est[0]) if est[0] else None}
            md += ["", f"- **Estimated prevalence** (stratified by marker): {pct(est)}, "
                       f"about {round(est[0] * N):,} of {N:,} sentences "
                       f"(range {round(est[1] * N):,}–{round(est[2] * N):,})",
                   f"- **Keyword proxy precision** (marker hits judged slop): {pct(p_m)}",
                   f"- **Slop rate without a marker**: {pct(p_nm)}"
                   + (f"; the proxy catches about {100 * b['prevalence']['proxy_recall']:.0f}% of slop"
                      if b["prevalence"]["proxy_recall"] else "")]
        cats = Counter(recs[i]["category"] for i in recs if recs[i]["label"] == "slop")
        b["categories"] = dict(cats.most_common())
        md += ["", "Slop by category: " + ", ".join(f"{c} {n}" for c, n in cats.most_common())]
        for key in ("language", "tool_name"):
            groups = defaultdict(Counter)
            for i in recs:
                if items[i]["stratum"] == "random":
                    groups[items[i][key]][recs[i]["label"]] += 1
            rows = sorted(groups.items(), key=lambda kv: -sum(kv[1].values()))[:8]
            md.append(f"Random stratum by {key}: " + "; ".join(
                f"{g} {c['slop']}/{c['slop'] + c['not_slop']}" for g, c in rows))
        js["backends"][backend] = b
        md.append("")
    # agreement
    raters = {f"llm:{k}": {i: v["label"] for i, v in recs.items()} for k, recs in llm.items()}
    raters.update({f"human:{k}": {i: v["label"] for i, v in recs.items()} for k, recs in hand.items()})
    names = sorted(raters)
    if len(names) > 1:
        md += ["## Agreement (Cohen's kappa on slop / not_slop; items marked unsure by either rater excluded)", ""]
        js["agreement"] = {}
        for i, a in enumerate(names):
            for b_ in names[i + 1:]:
                common = [x for x in raters[a] if x in raters[b_]
                          and raters[a][x] != "unsure" and raters[b_][x] != "unsure"]
                la, lb = [raters[a][x] for x in common], [raters[b_][x] for x in common]
                k = kappa(la, lb)
                acc = sum(x == y for x, y in zip(la, lb)) / len(common) if common else None
                js["agreement"][f"{a} vs {b_}"] = {"n": len(common), "kappa": k, "agreement": acc}
                if common:
                    md.append(f"- {a} vs {b_}: n={len(common)}, agreement {100 * acc:.0f}%, kappa "
                              + (f"{k:.2f}" if k is not None else "n/a"))
        for h, hrecs in hand.items():
            for backend, recs in llm.items():
                both = [i for i, v in hrecs.items() if v["label"] == "slop" and v["category"] and i in recs
                        and recs[i]["label"] == "slop"]
                if both:
                    same = sum(hrecs[i]["category"] == recs[i]["category"] for i in both)
                    md.append(f"- category match, human:{h} vs llm:{backend} (both slop, category given): "
                              f"{same}/{len(both)}")
        disagree = []
        for x in sorted(items):
            labs = {n: raters[n][x] for n in names if x in raters[n]}
            if len(set(labs.values()) - {"unsure"}) > 1:
                disagree.append((x, labs))
        if disagree:
            md += ["", f"Disagreements ({len(disagree)}), worth reading together to refine the rulebook:"]
            md += [f"- {x}: {items[x]['sentence'][:140]!r}  {labs}" for x, labs in disagree[:40]]
    (d / "report.md").write_text("\n".join(md) + "\n")
    (d / "report.json").write_text(json.dumps(js, indent=2, default=str) + "\n")
    print("\n".join(md))


def cmd_serve(args) -> None:
    from labelui import serve
    d = run_dir(args.run)
    if not (d / "handsheet.csv").exists():
        sys.exit("no handsheet.csv yet; run `pilot.py handsheet` first")
    serve(d, DB, args.port)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", default="pilot_v1", help="output folder under data/swechat/labels/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", parents=[common])
    s.add_argument("--random", type=int, default=200)
    s.add_argument("--marker", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)
    s = sub.add_parser("from-screen", parents=[common], help="judging run from a screen run's flagged sentences")
    s.add_argument("--screen", default="screen_all")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_from_screen)
    s = sub.add_parser("prompts", parents=[common])
    s.set_defaults(func=cmd_prompts)
    s = sub.add_parser("label", parents=[common])
    s.add_argument("--backend", choices=["claude", "codex"], default="claude")
    s.add_argument("--model", default=None)
    s.add_argument("--effort", default=None, help="claude: low…max; codex: model_reasoning_effort")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--only", default=None, help="comma-separated item ids")
    s.add_argument("--retries", type=int, default=2)
    s.add_argument("--tag", default=None, help="suffix for the output file, e.g. v1.1 -> llm_claude_v1.1.jsonl")
    s.set_defaults(func=cmd_label)
    s = sub.add_parser("handsheet", parents=[common])
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_handsheet)
    s = sub.add_parser("report", parents=[common])
    s.set_defaults(func=cmd_report)
    s = sub.add_parser("serve", parents=[common], help="local web UI for hand labels (writes handsheet.csv)")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
