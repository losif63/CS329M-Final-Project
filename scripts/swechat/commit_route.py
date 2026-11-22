"""Certify file versions from linked git commits (commits.parquet).

For a file in a commit we know the committed text (file_attribution[path].committed_version)
and the commit's own diff (patch), hence the pre-commit text. The agent's changes
(agent_changes: Write content / Edit old_string+new_string) are undone newest-first from the
committed text; if that lands exactly on the pre-commit text (or on a Write's content), the
commit's change to that file is fully explained by the agent's visible changes, so every
intermediate version is known exactly. Anything else (human edits, formatters, subagents,
partially captured changes) breaks the match and nothing is certified.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from languages import is_code_path

UNKNOWN = object()  # before-state sentinel: a Write over content we never saw
_CLOCK_SLACK = timedelta(seconds=5)
_REBASE_SLACK = timedelta(minutes=2)  # author vs commit date: larger gaps mean rebase/amend/cherry-pick
_AFTER_SESSION_SLACK = timedelta(minutes=10)
# commands that can bring other code into the working tree under the agent's edits
_GIT = r"\bgit\s+(?:-C\s+\S+\s+)?"
BASE_MOVE_RE = re.compile(
    _GIT + r"(?:pull|merge|rebase|cherry-pick|am|apply|revert)\b"
    + r"|" + _GIT + r"stash\s+(?:pop|apply)\b"
    + r"|" + _GIT + r"reset\b[^;&|\n]*--hard\b"
    # switching to another ref (not creating a branch at HEAD, not restoring files with "--")
    + r"|" + _GIT + r"(?:checkout|switch)\b(?![^;&|\n]*\s(?:-[bBcC]|-qb|--orphan)\b)(?![^;&|\n]*\s--(?:\s|$))"
    + r"|\bgh\s+pr\s+checkout\b")


@dataclass
class CommitVersion:
    turn: int
    file_path: str
    tool: str
    content: str
    before: object  # str | None (file created) | UNKNOWN


@dataclass
class CommitStats:
    commits: int = 0
    files: int = 0
    reasons: dict = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.reasons[key] = self.reasons.get(key, 0) + n


# ---------------------------------------------------------------- unified diff

def file_patches(patch: str | None) -> dict[str, dict]:
    """Per-file hunks of a `git show` patch, using hunk header counts to delimit hunks."""
    out: dict[str, dict] = {}
    lines = (patch or "").split("\n")
    cur = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            m = re.match(r'diff --git "?a/(.*?)"? "?b/(.*?)"?$', line)
            cur = {"hunks": [], "new_file": False, "deleted": False, "binary": False}
            if m:
                out[m.group(2)] = cur
        elif cur is None:
            pass
        elif line.startswith("new file mode"):
            cur["new_file"] = True
        elif line.startswith("deleted file mode"):
            cur["deleted"] = True
        elif line.startswith(("Binary files", "GIT binary patch")):
            cur["binary"] = True
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not m:
                cur["binary"] = True  # unparseable: refuse this file
                i += 1
                continue
            oc = int(m.group(2)) if m.group(2) is not None else 1
            nc = int(m.group(4)) if m.group(4) is not None else 1
            h = {"ns": int(m.group(3)), "old": [], "new": []}
            i += 1
            while (oc > 0 or nc > 0) and i < len(lines):
                ln = lines[i]
                if ln.startswith("\\"):  # "\ No newline at end of file"
                    i += 1
                    continue
                tag, body = (ln[0], ln[1:]) if ln else (" ", "")
                if tag == " ":
                    h["old"].append(body); h["new"].append(body); oc -= 1; nc -= 1  # noqa: E702
                elif tag == "-":
                    h["old"].append(body); oc -= 1  # noqa: E702
                elif tag == "+":
                    h["new"].append(body); nc -= 1  # noqa: E702
                else:
                    break
                i += 1
            if oc != 0 or nc != 0:
                cur["binary"] = True  # truncated or malformed hunk
            cur["hunks"].append(h)
            continue
        i += 1
    return out


def reverse_apply(text: str, fp: dict) -> str | None:
    """Pre-commit text of a file from its committed text and its diff (None if it doesn't apply)."""
    if fp["binary"]:
        return None
    if fp["new_file"]:
        return ""
    lines = text.split("\n")
    trailing_nl = bool(lines) and lines[-1] == ""
    if trailing_nl:
        lines.pop()
    for h in sorted(fp["hunks"], key=lambda h: -h["ns"]):
        start = h["ns"] - 1 if h["new"] else h["ns"]
        if lines[start:start + len(h["new"])] != h["new"]:
            return None
        lines[start:start + len(h["new"])] = h["old"]
    return "\n".join(lines) + ("\n" if trailing_nl else "")


def same_text(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a.rstrip("\n") == b.rstrip("\n")


_REDACTED = "REDACTED"


def same_modulo_redaction(redacted: str | None, real: str | None) -> bool:
    """True if `redacted` (conversation-derived, PII-redacted by the dataset) equals `real`
    (from git) except where it says REDACTED. Each placeholder may stand for any short span."""
    if redacted is None or real is None:
        return False
    if same_text(redacted, real):
        return True
    a, b = redacted.rstrip("\n"), real.rstrip("\n")
    if _REDACTED not in a or a.count(_REDACTED) > 50:
        return False
    parts = a.split(_REDACTED)
    pattern = "".join(re.escape(p) + (r"[\s\S]{0,300}?" if i < len(parts) - 1 else "") for i, p in enumerate(parts))
    try:
        return re.fullmatch(pattern, b) is not None
    except re.error:
        return False


# ---------------------------------------------------------------- certification

def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    if re.search(r"[+-]\d\d$", s):
        s += ":00"
    try:
        return datetime.fromisoformat(s.replace(" ", "T", 1))
    except ValueError:
        return None


def _is_write(change: dict) -> bool:
    return change.get("tool_name", "").endswith("Write")


def _explain(committed: str, parent: str, chs: list[dict]) -> tuple[list[dict], str] | None:
    """Undo changes newest-first until reaching `parent` (anchor 'parent') or a Write whose
    content equals the current text (anchor 'write'). Returns (chain oldest-first, anchor)."""
    state = committed
    for k in range(len(chs) - 1, -1, -1):
        c = chs[k]
        if _is_write(c):
            return (chs[k:], "write") if same_text(state, c.get("content")) else None
        old, new = c.get("old_string"), c.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str) or not new or state.count(new) != 1:
            return None
        state = state.replace(new, old, 1)
        if same_text(state, parent):
            return chs[k:], "parent"
    return None


def _replay(chain: list[dict], anchor: str, parent: str, new_file: bool) -> list[tuple[object, str]] | None:
    """(before, after) text for each change of an explained chain."""
    out: list[tuple[object, str]] = []
    state: object = parent
    for i, c in enumerate(chain):
        if _is_write(c):
            before = state if i > 0 else (None if new_file else (UNKNOWN if anchor == "write" else parent))
            state = c["content"]
            out.append((before, state))
            continue
        old = c.get("old_string")
        if not isinstance(state, str) or not old or state.count(old) != 1:
            return None
        after = state.replace(old, c["new_string"], 1)
        out.append((state, after))
        state = after
    return out


def _signature(file_path: str, tool: str, inp: dict) -> tuple | None:
    if tool == "Write":
        return (file_path, "W", inp.get("content"))
    edits = inp.get("edits") or []
    if len(edits) != 1:
        return None  # MultiEdit is recorded differently; skip rather than guess
    return (file_path, "E", edits[0]["old_string"], edits[0]["new_string"])


def certify(session_id: str, uses: list[dict], commit_rows: list[tuple], stats: CommitStats,
            session_end: str | None = None, base_moves: list[str] | None = None) -> dict[int, CommitVersion]:
    """uses: successful write-like tool calls of this session: {turn, tool, inp, ts}.
    commit_rows: (commit_date, author_date, patch, agent_changes_json, file_attribution_json)
    for the session's linked commits. session_end: timestamp of the session's last turn.
    base_moves: timestamps of session Bash commands that may move the git base.

    The explanation assumes the agent's edits were applied on the commit's parent. That only
    holds if the base did not move between the edits and the commit, so commits that were
    rebased/amended, made after the session, or preceded by base-moving git commands are refused.
    Returns turn -> certified version."""
    end_ts = _ts(session_end)
    move_ts = [t for t in (_ts(x) for x in base_moves or []) if t is not None]
    by_sig: dict[tuple, list[dict]] = {}
    for u in uses:
        sig = _signature(u["inp"]["file_path"], u["tool"], u["inp"])
        if sig is not None:
            by_sig.setdefault(sig, []).append(u)
    certified: dict[int, CommitVersion] = {}
    conflicted: set[int] = set()
    for commit_date, author_date, patch, ac, fa in commit_rows:
        stats.commits += 1
        try:
            changes = json.loads(ac or "[]")
            attrs = json.loads(fa or "{}")
        except json.JSONDecodeError:
            stats.bump("bad_json")
            continue
        if not changes or not isinstance(attrs, dict):
            continue
        cts, ats = _ts(commit_date), _ts(author_date)
        if cts is None or ats is None or abs(cts - ats) > _REBASE_SLACK:
            stats.bump("commit_rebased_or_amended")
            continue
        if end_ts is not None and cts > end_ts + _AFTER_SESSION_SLACK:
            stats.bump("committed_after_session")
            continue
        fps = None
        by_file: dict[str, list[dict]] = {}
        for c in changes:
            if isinstance(c, dict) and is_code_path(c.get("file_path")):
                by_file.setdefault(c["file_path"], []).append(c)
        for absf, chs in by_file.items():
            if not any(c.get("session_id") == session_id for c in chs):
                continue
            stats.files += 1
            rel = max((k for k, v in attrs.items() if isinstance(v, dict) and "committed_version" in v
                       and (absf == k or absf.endswith("/" + k))), key=len, default=None)
            if rel is None:
                stats.bump("file_not_in_commit")
                continue
            committed = attrs[rel]["committed_version"]
            if not isinstance(committed, str):
                stats.bump("no_committed_text")
                continue
            if fps is None:
                fps = file_patches(patch)
            if rel not in fps:
                stats.bump("file_unchanged_in_commit")
                continue
            parent = reverse_apply(committed, fps[rel])
            if parent is None:
                stats.bump("patch_reverse_failed")
                continue
            chs = [c for c in chs if cts is None or (_ts(c.get("timestamp")) or cts) <= cts + _CLOCK_SLACK]
            chs.sort(key=lambda c: (c.get("timestamp") or "", int(c.get("change_position") or 0)))
            got = _explain(committed, parent, chs)
            if got is None:
                stats.bump("not_explained_by_agent_changes")
                continue
            chain, anchor = got
            first = _ts(chain[0].get("timestamp"))
            if first is None or any(first <= t <= cts for t in move_ts):
                stats.bump("base_may_have_moved")
                continue
            states = _replay(chain, anchor, parent, fps[rel]["new_file"])
            if states is None or not same_text(states[-1][1], committed):
                stats.bump("replay_mismatch")
                continue
            stats.bump("explained_files")
            for c, (before, after) in zip(chain, states):
                if c.get("session_id") != session_id:
                    continue
                tool = "Write" if _is_write(c) else "Edit"
                sig = (absf, "W", c.get("content")) if tool == "Write" else (absf, "E", c.get("old_string"), c.get("new_string"))
                cands = [u for u in by_sig.get(sig, []) if u["turn"] not in certified or certified[u["turn"]].content == after]
                if len(cands) > 1:
                    cts_c = _ts(c.get("timestamp"))
                    near = [u for u in cands if cts_c and u.get("ts") and abs((_ts(u["ts"]) or cts_c) - cts_c) < timedelta(seconds=60)]
                    cands = near if len(near) == 1 else []
                if len(cands) != 1:
                    stats.bump("ambiguous_or_unmapped_change")
                    continue
                u = cands[0]
                prev = certified.get(u["turn"])
                if prev is not None and not same_text(prev.content, after):
                    conflicted.add(u["turn"])
                    continue
                certified[u["turn"]] = CommitVersion(u["turn"], absf, u["tool"], after, before)
    for t in conflicted:
        certified.pop(t, None)
        stats.bump("conflicting_commits")
    stats.bump("versions_certified", len(certified))
    return certified
