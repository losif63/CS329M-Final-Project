"""Upload the exported viewer (data/swechat/viewer_site) to the team's Hugging Face Space.

Usage: python scripts/swechat/viewer/publish.py [--space CS329M/Data-Visualization] [--site DIR]

Requires `huggingface-cli login` with write access to the Space. Refuses to upload unless the
Space is private: the viewer includes conversation tool output that the dataset did not redact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="CS329M/Data-Visualization")
    ap.add_argument("--site", type=Path, default=ROOT / "data/swechat/viewer_site")
    args = ap.parse_args()
    if not (args.site / "index.html").exists() or not (args.site / "index.js").exists():
        raise SystemExit(f"{args.site} has no exported viewer; run scripts/swechat/viewer/export.py first")
    api = HfApi()
    if not api.space_info(args.space).private:
        raise SystemExit(f"refusing to upload: {args.space} is not private")
    n = sum(1 for p in args.site.rglob("*") if p.is_file())
    print(f"uploading {n} files from {args.site} to {args.space} …", flush=True)
    try:
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=args.site,
                          commit_message="Update slop comment viewer")
    except Exception as e:  # the commit endpoint can time out (504) after the commit has landed
        print(f"upload call raised {type(e).__name__}: {e}")
    remote = set(api.list_repo_files(args.space, repo_type="space"))
    local = {p.relative_to(args.site).as_posix() for p in args.site.rglob("*") if p.is_file()}
    missing = sorted(local - remote)
    if missing:
        raise SystemExit(f"{len(missing)} files missing on the Space, e.g. {missing[:3]}; re-run to retry")
    print(f"done: all {len(local)} files present on https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
