# SWE-chat slop-comment pipeline

Extracts agent-written code comment sentences from [SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat)
(Claude Code sessions, all code languages). Each sentence comes with the exact file version it was written into
and the conversation that preceded it, as training data for a "slop comment" classifier: comments whose meaning
depends on session context that isn't in the repository.

The table schema and every certification and filtering rule are documented in [`SCHEMA.md`](SCHEMA.md).

## Reproduce

```bash
conda create -n cs329 python=3.12 -y && conda activate cs329
pip install -r scripts/swechat/requirements.txt

scripts/swechat/run_all.sh                    # download → tests → extract → viewer export
LIMIT=50 SEED=0 scripts/swechat/run_all.sh    # quick sample of 50 sessions
PUBLISH=1 scripts/swechat/run_all.sh          # also upload the viewer (needs `huggingface-cli login`)
```

Step by step:

| Step | Command | Output |
|---|---|---|
| 1. Download | `python scripts/swechat/download.py` | `data/swechat/raw/*.parquet` (~11 GB) at the pinned dataset revision `202b071f`, SHA-256-verified, plus tree-sitter grammars cached for offline use |
| 2. Test | `python -m pytest scripts/swechat/tests` | 73 unit tests on synthetic cases |
| 3. Extract | `python scripts/swechat/extract.py [--limit-sessions N] [--seed S] [--out DIR]` | `data/swechat/db/`: `turn`, `source_file`, `comment`, `dropped_sample` parquet files, plus `stats.json` and `README.md` (copied from `SCHEMA.md`). The full run takes about 15 min |
| 4. Viewer | `python scripts/swechat/viewer/export.py [--db DIR]` | `data/swechat/viewer_site/`: `index.html`, `index.js`, and `s/<k>.js` session shards. Open `index.html` directly from disk |
| 5. Publish | `python scripts/swechat/viewer/publish.py` | uploads the viewer to the **private** Space `CS329M/Data-Visualization`, and refuses if the Space is public |

**Determinism.** Given the pinned data revision and package versions, the same `--limit-sessions`/`--seed` produce
byte-identical parquet files. Session sampling uses `random.Random(seed)` over sorted session IDs, and every output is
written in a fixed order. `data/` is gitignored, so outputs are rebuilt rather than committed.

The full build (all 7,661 eligible sessions) produced 80,471 certified file versions and 162,584 comment sentences
across 31 languages. `stats.json` records the counts at every filter step.

## Code map

| File | Role |
|---|---|
| `extract.py` | CLI: session selection, chunked loading with DuckDB, writing tables and stats |
| `languages.py` | extension → language, comment delimiters, directive patterns, scratch-path rule |
| `reconstruct.py` | session route: pairing tool calls with results, tracking file state, Bash/subagent guards, retraction, merging with the commit route |
| `commit_route.py` | commit route: diff parsing, explaining a commit through the agent's changes, rebase/base-move guards, redaction-tolerant comparison |
| `comments.py` | comment extraction (Python `tokenize`/`ast`, tree-sitter for other languages), non-prose filters, diffing for newly added comments |
| `sentences.py` | pysbd sentence splitting with protected spans, and line mapping |
| `conversation.py` | compact conversation view (full text turns, one-line stubs for thinking/tool calls) shared by the viewer and `build_example` |
| `example.py` | `build_example(comment_id, ...)` → (sentence, source file, whole-session conversation text) for model inputs |
| `download.py`, `viewer/export.py`, `viewer/publish.py`, `run_all.sh` | reproducibility tooling |
| `tests/` | unit tests |

The viewer's "slop-marker hits" filter is a regex proxy (wording like "now", "previously", "instead of", "as requested", "locally"),
for browsing only. It is not a label.
