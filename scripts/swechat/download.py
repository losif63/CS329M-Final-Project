"""Download the SWE-chat raw tables at a pinned revision, verify them, and prefetch grammars.

Usage: python scripts/swechat/download.py [--revision SHA] [--no-verify]

Files land in data/swechat/raw/. Existing files whose size and SHA-256 already match are kept.
The tree-sitter grammars used by the extractor are downloaded into the package cache (keyed to
the pinned tree-sitter-language-pack version), so extraction can run offline afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from languages import EXT_LANG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/swechat/raw"
REPO = "SALT-NLP/SWE-chat"
# dataset revision the published tables were built from, with the files' SHA-256 at that revision
REVISION = "202b071f18e03c79df5a7287565ce2ebc5ee7756"
FILES = {
    "sessions.parquet": (36562091, "28e7d4c6015fcb895c2bb6c1cb1c3cf5861b4b9876dba439284617e15d2d38fe"),
    "repositories.parquet": (900468, "b2983efca46267c05e77045ddf5f8efc281ccd188ef3b54b5b7314ebf3d2e4ad"),
    "checkpoints.parquet": (25471283, "5a0e8fd1d52d182921680b45d23d0b8202671db7b533362d184d76d52875240f"),
    "commits.parquet": (5102596005, "8d344cf14731c58d85025f9d16ae1ae5379d256447b32ccbfb5884504a3ff81c"),
    "conversations.parquet": (6424352043, "b6731767fb89f4e613cd7876ba76aebcf845fa1b75fcc198b97204238faf995d"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    pinned = args.revision == REVISION

    for name, (size, digest) in FILES.items():
        path = RAW / name
        if path.exists() and pinned and path.stat().st_size == size and (args.no_verify or sha256(path) == digest):
            print(f"ok       {name}")
            continue
        print(f"download {name} @ {args.revision[:8]} …", flush=True)
        hf_hub_download(REPO, name, repo_type="dataset", revision=args.revision, local_dir=RAW)
        if pinned and not args.no_verify:
            got = sha256(path)
            if got != digest:
                sys.exit(f"SHA-256 mismatch for {name}: {got} != {digest}")
        print(f"ok       {name}")

    import tree_sitter_language_pack as tslp
    langs = sorted({lang for lang in EXT_LANG.values() if lang != "python"} | {"javascript", "typescript"})
    tslp.download(langs)
    print(f"grammars cached ({tslp.__version__}): {', '.join(sorted(tslp.downloaded_languages()))}")


if __name__ == "__main__":
    main()
