"""Local web UI for hand-labeling the pilot sheet (handsheet.csv).

    python scripts/swechat/labeling/pilot.py serve [--run pilot_v1] [--port 8000]

Shows, for each sheet item: the target sentence inside its full comment, the full source file with the sentence
highlighted, the other files the agent wrote earlier in the session (latest version before this write), and the whole
conversation before the write, with every tool call expandable to its input and output. Labels are written back to
handsheet.csv on every change. The UI never shows the LLM labels or the sampling stratum.
Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import duckdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from conversation import build_items, items_before  # noqa: E402

LABELS = ("slop", "not_slop", "unsure")
CATEGORIES = ("conversation", "session_history", "local_environment", "unverifiable_result", "plan_reference",
              "other_session")
MAX_TOOL_CHARS = 40_000


class Store:
    def __init__(self, run_dir: Path, db: Path):
        self.d, self.db = run_dir, db
        self.sheet = run_dir / "handsheet.csv"
        self.key = json.loads((run_dir / "handsheet_key.json").read_text())
        self.lock = threading.Lock()
        items = duckdb.connect().execute(f"SELECT item_id, comment_id FROM '{run_dir / 'sample.parquet'}'").fetchall()
        self.comment_of = {i: int(c) for i, c in items}

    # ---- sheet
    def rows(self) -> tuple[list[str], list[dict]]:
        with self.sheet.open(newline="") as f:
            r = csv.DictReader(f)
            return list(r.fieldnames or []), list(r)

    def save(self, hid: str, label: str, category: str, notes: str) -> None:
        with self.lock:
            fields, rows = self.rows()
            for r in rows:
                if r["hand_id"] == hid:
                    r["label"], r["category"], r["notes"] = label, category, notes
                    break
            else:
                raise KeyError(hid)
            fd, tmp = tempfile.mkstemp(dir=self.d, suffix=".csv")
            with os.fdopen(fd, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, self.sheet)  # atomic: a crash never leaves a half-written sheet

    # ---- item context
    def con(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    @lru_cache(maxsize=32)
    def session_turns(self, session_id: str) -> tuple:
        return tuple(self.con().execute(f"""
            SELECT turn_number, turn_type, tool_name, tool_call_id, content, file_path, command
            FROM '{self.db / "turn.parquet"}' WHERE session_id = ? ORDER BY turn_number""", [session_id]).fetchall())

    def item(self, hid: str) -> dict:
        cid = self.comment_of[self.key[hid]]
        con = self.con()
        (sent, full, kind, l0, l1, sid, turn, fp, lang, tool, content, repo) = con.execute(f"""
            SELECT c.sentence, c.full_comment, c.comment_kind, c.line_start, c.line_end, s.session_id,
                   s.turn_number, s.file_path, s.language, s.tool_name, s.content, s.repo_id
            FROM '{self.db / "comment.parquet"}' c JOIN '{self.db / "source_file.parquet"}' s ON s.id = c.source_id
            WHERE c.id = ?""", [cid]).fetchall()[0]
        turns = self.session_turns(sid)
        items = items_before(build_items([(t[0], t[1], t[2], t[4], t[5], t[6]) for t in turns]), turn)
        others = con.execute(f"""
            SELECT file_path, language, turn_number, content FROM (
              SELECT *, row_number() OVER (PARTITION BY file_path ORDER BY turn_number DESC) AS k
              FROM '{self.db / "source_file.parquet"}' WHERE session_id = ? AND turn_number < ? AND file_path <> ?)
            WHERE k = 1 ORDER BY file_path""", [sid, turn, fp]).fetchall()
        _, rows = self.rows()
        mine = next(r for r in rows if r["hand_id"] == hid)
        return {"hand_id": hid, "sentence": sent, "full_comment": full, "kind": kind, "line_start": l0,
                "line_end": l1, "file_path": fp, "language": lang, "tool": tool, "write_turn": turn,
                "repo": repo, "content": content, "conversation": items,
                "other_files": [{"path": p, "language": lg, "turn": t, "content": c} for p, lg, t, c in others],
                "label": mine.get("label", ""), "category": mine.get("category", ""), "notes": mine.get("notes", "")}

    def tool_call(self, hid: str, turn: int) -> dict:
        cid = self.comment_of[self.key[hid]]
        sid = self.con().execute(f"""
            SELECT s.session_id FROM '{self.db / "comment.parquet"}' c
            JOIN '{self.db / "source_file.parquet"}' s ON s.id = c.source_id WHERE c.id = ?""", [cid]).fetchall()[0][0]
        turns = self.session_turns(sid)
        use = next((t for t in turns if t[0] == turn and t[1] == "tool_use"), None)
        if use is None:
            return {"error": "not a tool call"}
        result = None
        if use[3]:  # nearest result with the same call id (ids can repeat within a session)
            cands = [t for t in turns if t[1] == "tool_result" and t[3] == use[3]]
            if cands:
                result = min(cands, key=lambda t: (abs(t[0] - turn), t[0] < turn))
        def clip(s: str | None) -> str:
            s = s or ""
            return s if len(s) <= MAX_TOOL_CHARS else s[:MAX_TOOL_CHARS] + f"\n…[{len(s) - MAX_TOOL_CHARS} more chars]"
        return {"tool": use[2], "turn": turn, "input": clip(use[4]),
                "output": clip(result[4]) if result else "(no result recorded)"}


def make_handler(store: Store):
    page = (HERE / "labelui.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 — quiet; signature matches the base class
            pass

        def send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code: int = 200) -> None:
            self.send(code, json.dumps(obj, ensure_ascii=False, default=str).encode())

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path in ("/", "/index.html"):
                    return self.send(200, page, "text/html; charset=utf-8")
                if path == "/rulebook":
                    return self.send(200, (HERE / "rulebook.md").read_bytes(), "text/plain; charset=utf-8")
                if path == "/api/items":
                    _, rows = store.rows()
                    return self.json([{"hand_id": r["hand_id"], "sentence": r["sentence"], "file": r["file"],
                                       "label": r.get("label", ""), "category": r.get("category", "")} for r in rows])
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[:2] == ["api", "item"]:
                    return self.json(store.item(parts[2]))
                if len(parts) == 4 and parts[:2] == ["api", "tool"]:
                    return self.json(store.tool_call(parts[2], int(parts[3])))
                self.json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001 — report to the page instead of dropping the connection
                self.json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                return self.json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                label = body.get("label", "")
                category = body.get("category", "")
                if label not in LABELS + ("",) or category not in CATEGORIES + ("",):
                    return self.json({"error": "bad label or category"}, 400)
                store.save(body["hand_id"], label, category, (body.get("notes") or "").replace("\r", ""))
                self.json({"ok": True})
            except Exception as e:  # noqa: BLE001
                self.json({"error": f"{type(e).__name__}: {e}"}, 500)

    return H


def serve(run_dir: Path, db: Path, port: int) -> None:
    store = Store(run_dir, db)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    print(f"labeling UI for {run_dir.name}: http://localhost:{port}  (Ctrl+C to stop; labels save to {store.sheet})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
