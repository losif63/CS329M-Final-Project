"""Compact conversation view shared by the viewer export and build_example.

Text turns (user prompts, assistant responses, compaction summaries) are kept in full. Every run
of agent work between them (thinking, tool calls, their results, injected system messages) is
collapsed into one stub: how many thinking turns, and each tool call as `Tool target` (file
path, short command or search pattern). Tool results are never shown. Text turns carry the
session context (the agent reports findings and the user replies to them), while the stubs keep
the provenance a labeler needs, e.g. whether the agent read a repository file or ran something.
"""

from __future__ import annotations

import json
import re

TEXT_KINDS = {"user_prompt": "user", "assistant_response": "assistant", "summary": "summary"}
WORK_TYPES = {"assistant_thinking", "tool_use", "tool_result", "system_injected"}
TARGET_CHARS = 80


def _short(s: str, n: int = TARGET_CHARS) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_path(p: str) -> str:
    parts = re.split(r"[\\/]", p.rstrip("/\\"))
    return p if len(parts) <= 3 else "…/" + "/".join(parts[-3:])


def tool_target(tool: str | None, content: str | None, file_path: str | None = None, command: str | None = None) -> str:
    """Short description of what a tool call touched, without its output. (`tool` is kept for
    call-site symmetry; targets are read from the input itself.)"""
    if command:
        return _short(command.split("\n", 1)[0] + (" …" if "\n" in command.strip() else ""))
    if file_path:
        return _short_path(file_path)
    try:
        d = json.loads(content or "")
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(d, dict):
        return ""
    if isinstance(d.get("pattern"), str):  # Grep / Glob: what was searched for, and where
        where = f" in {_short_path(d['path'])}" if isinstance(d.get("path"), str) else ""
        return _short(f'"{d["pattern"]}"{where}')
    for key in ("file_path", "filePath", "notebook_path", "path"):
        if isinstance(d.get(key), str) and d[key]:
            return _short_path(d[key])
    for key in ("query", "url", "description", "skill", "subagent_type", "title", "prompt"):
        if isinstance(d.get(key), str) and d[key]:
            return _short(d[key])
    if isinstance(d.get("todos"), list):
        return f"{len(d['todos'])} todos"
    return ""


def short_tool_name(tool: str | None) -> str:
    tool = tool or "tool"
    m = re.match(r"mcp__([^_]+(?:_[^_]+)*)__(.+)$", tool)
    return f"mcp:{m.group(1)}/{m.group(2)}" if m else tool


def build_items(turns: list[tuple]) -> list[dict]:
    """turns: (turn_number, turn_type, tool_name, content, file_path, command), in order.

    Returns items: {"k": "user"|"assistant"|"summary", "t": turn, "x": text} or
    {"k": "work", "t": first_turn, "h": [thinking turns], "c": [[tool, target, turn], ...], "s": n_system}.
    Other turn types (attachments, progress, queue operations, ...) are ignored.
    """
    items: list[dict] = []
    work: dict | None = None
    for tn, tt, tool, content, file_path, command in turns:
        if tt in TEXT_KINDS:
            work = None
            text = (content or "").strip()
            if text:
                items.append({"k": TEXT_KINDS[tt], "t": tn, "x": text})
            continue
        if tt not in WORK_TYPES:
            continue
        if work is None:
            work = {"k": "work", "t": tn, "h": [], "c": [], "s": 0}
            items.append(work)
        if tt == "assistant_thinking":
            work["h"].append(tn)
        elif tt == "tool_use":
            work["c"].append([short_tool_name(tool), tool_target(tool, content, file_path, command), tn])
        elif tt == "system_injected":
            work["s"] += 1
    return items


def items_before(items: list[dict], turn: int) -> list[dict]:
    """Items strictly before `turn`; a work stub spanning it is clipped to its earlier part."""
    out = []
    for it in items:
        if it["t"] >= turn:
            break
        if it["k"] == "work":
            it = {**it, "h": [t for t in it["h"] if t < turn], "c": [c for c in it["c"] if c[2] < turn]}
            if not it["h"] and not it["c"] and not it["s"]:
                continue
        out.append(it)
    return out


def work_line(it: dict) -> str:
    bits = []
    if it["h"]:
        bits.append("thinking" + (f" ×{len(it['h'])}" if len(it["h"]) > 1 else ""))
    if it["c"]:
        calls = " · ".join(f"{name} {target}".strip() for name, target, _ in it["c"])
        bits.append(f"{len(it['c'])} tool call{'s' if len(it['c']) > 1 else ''}: {calls}")
    if it["s"] and not bits:
        bits.append("system message" + (f" ×{it['s']}" if it["s"] > 1 else ""))
    return "[agent: " + " · ".join(bits) + "]"


def render_text(items: list[dict], max_total_chars: int | None = None, max_turn_chars: int | None = None) -> str:
    """Plain-text conversation for model inputs. Over `max_total_chars`, the oldest items are dropped."""
    lines = []
    for it in items:
        if it["k"] == "work":
            lines.append(work_line(it))
        else:
            text = it["x"]
            if max_turn_chars and len(text) > max_turn_chars:
                text = text[:max_turn_chars] + f" …[+{len(text) - max_turn_chars} chars]"
            lines.append(f"[{it['k']}] {text}")
    if max_total_chars is None or sum(len(s) + 1 for s in lines) <= max_total_chars:
        return "\n".join(lines)
    kept, total = [], 0
    for s in reversed(lines):  # keep the most recent context
        if total + len(s) + 1 > max_total_chars:
            break
        kept.append(s)
        total += len(s) + 1
    dropped = len(lines) - len(kept)
    return "\n".join([f"[… {dropped} earlier items omitted]"] + list(reversed(kept)))
