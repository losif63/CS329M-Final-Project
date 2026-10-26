"""Assemble a model input for one comment sentence.

The conversation for a comment is every turn row of the same session with
turn_number < source_file.turn_number; it is not stored, only derived here.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DB = Path(__file__).resolve().parents[2] / "data/swechat/db"
CONVERSATION_TURN_TYPES = ("user_prompt", "system_injected", "assistant_response", "assistant_thinking",
                           "tool_use", "tool_result", "summary")


def format_turn(turn_type: str, tool_name: str | None, content: str | None, max_turn_chars: int) -> str:
    head = f"[{turn_type} {tool_name}]" if turn_type == "tool_use" and tool_name else f"[{turn_type}]"
    text = (content or "").strip()
    if len(text) > max_turn_chars:
        text = text[:max_turn_chars] + f" …[+{len(text) - max_turn_chars} chars]"
    return f"{head} {text}"


def build_example(comment_id: int, max_turn_chars: int = 500, max_total_chars: int = 20000,
                  db: Path = DB, con: duckdb.DuckDBPyConnection | None = None) -> tuple[str, str, str]:
    """Return (sentence, source_file_content, conversation_text) for a comment id.

    Turns are formatted in order, each truncated to max_turn_chars; when the
    whole conversation exceeds max_total_chars the oldest turns are dropped.
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
    turns = con.execute(f"""
        SELECT turn_type, tool_name, content FROM read_parquet('{db / "turn.parquet"}')
        WHERE session_id = ? AND turn_number < ? AND turn_type IN {CONVERSATION_TURN_TYPES}
        ORDER BY turn_number""", [session_id, turn_number]).fetchall()
    kept: list[str] = []
    total = 0
    for tt, tn, content in reversed(turns):  # keep the most recent context
        s = format_turn(tt, tn, content, max_turn_chars)
        if total + len(s) + 1 > max_total_chars:
            break
        kept.append(s)
        total += len(s) + 1
    return sentence, source, "\n".join(reversed(kept))
