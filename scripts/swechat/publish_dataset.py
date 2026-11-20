"""Upload the labeled dataset to the team's private Hugging Face dataset repository.

    python scripts/swechat/publish_dataset.py [--repo CS329M/SWEChat-AISlopComments] [--tag v1.2] [--dry-run]

Stages data/swechat/db tables, the rulebooks and the labeling-run outputs into a folder with a dataset card, then
uploads it. Refuses to upload unless the repository is private: the conversations include tool output (file
contents, command output) that the source dataset did not redact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data/swechat/db"
LABELS = ROOT / "data/swechat/labels"
HERE = Path(__file__).resolve().parent

CARD = """---
license: odc-by
pretty_name: SWE-chat AI Slop Comments
configs:
  - config_name: comment
    default: true
    data_files: data/comment.parquet
  - config_name: label
    data_files: data/label_{tag}.parquet
  - config_name: source_file
    data_files: data/source_file.parquet
  - config_name: turn
    data_files: data/turn.parquet
---

# SWE-chat AI Slop Comments

Code-comment sentences written by Claude Code in real sessions from
[SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) (v2, revision `202b071f`), each tied to the exact
file version it was written into and the conversation that preceded it, and labeled for **slop**: a sentence whose
meaning or truth depends on the authoring session (the conversation or the agent's machine) rather than the code.

**Private.** The conversations include tool output (file contents, command output) that the source dataset did not
redact. Do not make this repository public.

## Contents

| Config / path | Rows | Description |
|---|---|---|
| `comment` (`data/comment.parquet`) | {sentences:,} | One row per agent-written comment sentence; `label` is `slop`, `not_slop` or `unsure`. |
| `label` (`data/label_{tag}.parquet`) | {sentences:,} | Label details: `category`, `rationale`, `evidence_turns`, `source` (`judge` or `screen_negative`), model, rulebook, pilot and hand labels. Join on `comment_id` = `comment.id`. |
| `source_file` (`data/source_file.parquet`) | {versions:,} | The exact file text after the write that added each comment (`comment.source_id` = `source_file.id`). |
| `turn` (`data/turn.parquet`) | {turns:,} | The sessions' conversations. A comment's context is every turn of its session with `turn_number < source_file.turn_number`. |
| `data/dropped_sample.parquet`, `data/stats.json` | | Extraction audit sample and per-step counts. |
| `labeling/` | | Rulebooks, the pilot (400 sentences: judge labels per rulebook version, 100 hand labels), and the raw screen and judge outputs. |

Schema details: `data/SCHEMA.md`. Code: `scripts/swechat/` in the project repository.

## Labels ({tag})

- **{slop:,} slop** ({rate:.1%}), {not_slop:,} not slop, {unsure} unsure; in {sessions} sessions across {repos} repositories.
- By category: {categories}.
- How: every sentence was screened by Claude Opus 5.5 (40 sentences per call, comment text only, tuned for recall).
  The {judged:,} flagged sentences were judged one at a time by Claude Opus 5.5 with rulebook {tag}, the code around
  the comment and the whole conversation. Unflagged sentences are labeled `not_slop` (`source = screen_negative`).

**Known limitations**
- The screen misses about 1 in 20 slop sentences (34 of 36 kept on the pilot), so some `screen_negative` rows are slop.
- Borderline labels are noisy: the same judge and rulebook run twice agree on 74 of 80 pilot items (κ 0.85).
- Human validation is limited to 100 sentences (one annotator, labeled before rulebook v1.2's history rule).
- Slop is concentrated: the top repository holds about a fifth of it, the top ten about half. Split by repository.
- The 400 pilot sentences (`label.pilot_item` not null) should be held out of training.

## Attribution

Derived from SWE-chat by Baumann et al., "SWE-chat: Coding Agent Interactions from Real Users in the Wild" (2026),
released under the Open Data Commons Attribution License (ODC-By 1.0).
"""


def stage(tag: str, dest: Path) -> dict:
    (dest / "data").mkdir(parents=True)
    for name in ["comment.parquet", f"label_{tag}.parquet", "source_file.parquet", "turn.parquet",
                 "dropped_sample.parquet", "stats.json", f"label_{tag}_summary.json"]:
        shutil.copy2(DB / name, dest / "data" / name)
    shutil.copy2(HERE / "SCHEMA.md", dest / "data" / "SCHEMA.md")
    lab = dest / "labeling"
    lab.mkdir()
    for rb in ["rulebook.md", "rulebook_v2.md", "README.md"]:
        shutil.copy2(HERE / "labeling" / rb, lab / rb)
    pilot = LABELS / "pilot_v1"
    (lab / "pilot_v1").mkdir()
    for p in pilot.iterdir():
        if p.is_file() and not p.name.endswith((".log", ".backup.csv")) and "backup" not in p.name:
            shutil.copy2(p, lab / "pilot_v1" / p.name)
    for run, keep in (("screen_all", "screen_*.jsonl"), ("judge_all", "llm_*_v*.jsonl")):
        (lab / run).mkdir()
        for p in (LABELS / run).glob(keep):
            if "backup" not in p.name:
                shutil.copy2(p, lab / run / p.name)
    shutil.copy2(LABELS / "judge_all" / "sample.parquet", lab / "judge_all" / "sample.parquet")
    s = json.loads((DB / f"label_{tag}_summary.json").read_text())
    import duckdb
    con = duckdb.connect()
    versions = con.execute(f"SELECT count(*) FROM '{DB / 'source_file.parquet'}'").fetchall()[0][0]
    turns = con.execute(f"SELECT count(*) FROM '{DB / 'turn.parquet'}'").fetchall()[0][0]
    cats = ", ".join(f"{k} {v:,}" for k, v in s["slop_by_category"].items())
    (dest / "README.md").write_text(CARD.format(
        tag=tag, sentences=s["sentences"], versions=versions, turns=turns, slop=s["labels"]["slop"],
        not_slop=s["labels"]["not_slop"], unsure=s["labels"].get("unsure", 0), rate=s["slop_rate"],
        sessions=s["sessions_with_slop"], repos=s["repos_with_slop"], categories=cats,
        judged=s["sources"]["judge"]))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="CS329M/SWEChat-AISlopComments")
    ap.add_argument("--tag", default="v1.2")
    ap.add_argument("--dry-run", action="store_true", help="stage the folder and print it, without uploading")
    args = ap.parse_args()
    api = HfApi()
    if not args.dry_run and not api.dataset_info(args.repo).private:
        raise SystemExit(f"refusing to upload: {args.repo} is not private")
    dest = Path(tempfile.mkdtemp(prefix="slop_dataset_"))
    stage(args.tag, dest)
    files = sorted(p for p in dest.rglob("*") if p.is_file())
    print(f"staged {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB in {dest}")
    for p in files:
        print(f"  {p.relative_to(dest)}  {p.stat().st_size / 1e6:.1f} MB")
    if args.dry_run:
        return
    try:
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=dest,
                          commit_message=f"SWE-chat slop comments, labels {args.tag}")
    except Exception as e:  # the commit endpoint can time out after the commit has landed; verify below
        print(f"upload call raised {type(e).__name__}: {e}")
    remote = set(api.list_repo_files(args.repo, repo_type="dataset"))
    missing = [str(p.relative_to(dest)) for p in files if str(p.relative_to(dest)) not in remote]
    if missing:
        raise SystemExit(f"{len(missing)} files missing on the Hub, e.g. {missing[:3]}; re-run to retry")
    print(f"done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
