"""Assemble a model input for one comment sentence.

The conversation for a comment is every turn of the same session with
turn_number < source_file.turn_number, rendered compactly (conversation.py): user prompts,
assistant responses and summaries in full; agent thinking and tool calls as one-line stubs.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conversation import build_items, items_before, render_text  # noqa: E402

DB = Path(__file__).resolve().parents[2] / "data/swechat/db"
# ~75k tokens: covers all but ~1% of sessions in full; longer ones keep the most recent context
DEFAULT_MAX_TOTAL_CHARS = 300_000


@lru_cache(maxsize=64)
def _session_items(db: str, session_id: str) -> tuple[dict, ...]:
    rows = duckdb.connect().execute(f"""
        SELECT turn_number, turn_type, tool_name, content, file_path, command
        FROM read_parquet('{Path(db) / "turn.parquet"}') WHERE session_id = ? ORDER BY turn_number""",
                                    [session_id]).fetchall()
    return tuple(build_items(rows))


def build_example(comment_id: int, max_total_chars: int | None = DEFAULT_MAX_TOTAL_CHARS,
                  max_turn_chars: int | None = None, db: Path = DB,
                  con: duckdb.DuckDBPyConnection | None = None, show_turns: bool = False) -> tuple[str, str, str]:
    """Return (sentence, source_file_content, conversation_text) for a comment id.

    The whole session before the write is included. Text turns can be capped with
    `max_turn_chars`; if the result exceeds `max_total_chars`, the oldest items are dropped.
    """
    con = con or duckdb.connect()
    row = con.execute(f"""
        SELECT c.sentence, s.content, s.session_id, s.turn_number
        FROM read_parquet('{db / "comment.parquet"}') c
        JOIN read_parquet('{db / "source_file.parquet"}') s ON s.id = c.source_id
        WHERE c.id = ?""", [comment_id]).fetchall()
    if not row:
        raise KeyError(f"comment id {comment_id} not found")
    sentence, source, session_id, turn_number = row[0]
    items = items_before(list(_session_items(str(db), session_id)), turn_number)
    return sentence, source, render_text(items, max_total_chars, max_turn_chars, show_turns)
