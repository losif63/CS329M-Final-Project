"""Reconstruct .py file versions from Claude Code Write/Edit/MultiEdit tool calls.

Findings from inspecting SWE-chat v2 (see README.md) that shape this module:
  * tool_call_id is NOT unique within a session (some models emit ids like
    "functions.Read:12" that are reused), so a tool_use is paired with the
    nearest following tool_result carrying the same id.
  * Success is decided by whitelist: only results matching SUCCESS_RE count.
    Everything else (<tool_use_error>..., user rejections, hook errors,
    permission prompts, EACCES, ...) is a failure.
  * Read results are line-numbered as "     12→text" (newer CLI) or "12\ttext"
    (older CLI), optionally followed by a <system-reminder> or
    "[result-id: rN]" trailer. Full reads are capped at 2000 lines.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field

from datetime import datetime, timezone

from commit_route import BASE_MOVE_RE, UNKNOWN, CommitStats, certify, same_modulo_redaction, same_text
from commit_route import _ts as _commit_ts
from languages import is_code_path

# results that certainly mean the write did NOT happen; any other non-success is "unknown"
KNOWN_FAILURE_RE = re.compile(
    r"^\s*(<tool_use_error>|The user doesn't want to proceed|PreToolUse:\w+ hook error|Claude requested permissions"
    r"|Permission for this action was denied|Edit denied by user|.{0,80}is temporarily unavailable"
    r"|Refusing to write|BLOCKED:|Cannot edit files outside|EACCES|ENOSPC|EPERM|File has been unexpectedly modified"
    r"|File content has changed since|Source write blocked|Tool permission request failed|InputValidationError)",
    re.S)
SUCCESS_RE = re.compile(
    r"^(The file .+ has been (updated|overwritten)|File created successfully at"
    r"|Applied \d+ edits? to)",
    re.S,
)
READ_LINE_RE = re.compile(r"^\s*(\d+)(?:→|\t)(.*)$")
READ_TRAILER_RE = re.compile(r"\n*(<system-reminder>.*?</system-reminder>\s*|\[result-id: r\d+\]\s*)$", re.S)
READ_MAX_LINES = 2000
WRITE_TOOLS = ("Write", "Edit", "MultiEdit")


def is_success(result_content: str | None) -> bool:
    return bool(result_content) and SUCCESS_RE.match(result_content) is not None


def parse_input(tool_name: str, tool_input_json: str | None) -> dict | None:
    """Normalize the tool input to {file_path, content} or {file_path, edits}.

    A handful of rows use alternative key spellings (filePath, newString,
    new_str, ...); they are accepted. Returns None if unusable.
    """
    try:
        d = json.loads(tool_input_json or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    fp = d.get("file_path") or d.get("filePath") or d.get("path")
    if not isinstance(fp, str):
        return None
    if tool_name == "Write":
        content = d.get("content", d.get("file_content", d.get("contents")))
        return {"file_path": fp, "content": content} if isinstance(content, str) else None
    if tool_name == "Edit":
        edits = [_norm_edit(d)]
    elif tool_name == "MultiEdit":
        raw = d.get("edits")
        if not isinstance(raw, list):
            return None
        edits = [_norm_edit(e) if isinstance(e, dict) else None for e in raw]
    else:
        return None
    if not edits or any(e is None for e in edits):
        return None
    return {"file_path": fp, "edits": edits}


def _norm_edit(d: dict) -> dict | None:
    old = d.get("old_string", d.get("oldString", d.get("old_str")))
    new = d.get("new_string", d.get("newString", d.get("new_str")))
    if not isinstance(old, str) or not isinstance(new, str):
        return None
    return {"old_string": old, "new_string": new, "replace_all": bool(d.get("replace_all", False))}


def apply_edits(state: str, edits: list[dict]) -> str | None:
    """Apply edits in order; None if any old_string is missing (or ambiguous) in the state."""
    for e in edits:
        old, new = e["old_string"], e["new_string"]
        if old == "":
            # Edit with empty old_string only succeeds on an empty/new file.
            if state != "":
                return None
            state = new
            continue
        n = state.count(old)
        # Claude Code rejects a non-replace_all edit unless old_string is unique on disk,
        # so any other count means our state disagrees with the real file.
        if n == 0 or (n > 1 and not e["replace_all"]):
            return None
        state = state.replace(old, new) if e["replace_all"] else state.replace(old, new, 1)
    return state


def _read_lines(content: str | None) -> dict[int, str] | None:
    """Numbered lines of a Read result ({line_no: text}), or None if it is not a plain listing."""
    if not content or content.lstrip().startswith("<tool_use_error>"):
        return None
    body = READ_TRAILER_RE.sub("", content)
    body = READ_TRAILER_RE.sub("", body)  # reminder and result-id can both appear
    lines = body.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out: dict[int, str] = {}
    prev = None
    for line in lines:
        m = READ_LINE_RE.match(line)
        if not m:
            return None  # unnumbered line: a notice or marker, not file content
        n = int(m.group(1))
        if prev is not None and n != prev + 1:
            return None
        if "[truncated]" in m.group(2):
            return None  # possibly a CLI truncation marker; cannot be sure of the text
        out[n] = m.group(2)
        prev = n
    return out or None


def parse_read_result(tool_input_json: str | None, content: str | None) -> str | None:
    """Return the full file text from a Read result, or None if it may be partial."""
    try:
        d = json.loads(tool_input_json or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict) or d.get("limit") is not None or d.get("offset") not in (None, 0, 1):
        return None
    lines = _read_lines(content)
    if not lines or min(lines) != 1 or len(lines) >= READ_MAX_LINES:
        return None
    return "\n".join(lines[i] for i in sorted(lines)) + "\n"


def partial_read_agrees(state: str, content: str | None) -> bool | None:
    """Whether a (partial) Read's lines match the tracked state; None if not comparable."""
    lines = _read_lines(content)
    if not lines:
        return None
    st = state.split("\n")
    return all(n - 1 < len(st) and st[n - 1] == t for n, t in lines.items())


@dataclass
class Version:
    session_id: str
    turn_number: int
    file_path: str
    tool_name: str
    content: str
    content_flag: str  # write | replayed | from_read | commit
    before: str | None  # state before the write (None = file did not exist)
    skip_reason: str | None = None  # set when comments in this version cannot be attributed
    retracted: bool = False  # later evidence showed the tracked state had drifted
    certified_by: str = "session"  # session | commit | both


@dataclass
class _FileState:
    text: str | None = None
    origin: str | None = None  # "write" (chain of writes/edits) or "read" (based on a Read)
    # Edit versions since the text was last confirmed (by a Write or an equal full Read);
    # retracted if drift is detected before the next confirmation.
    pending: list = field(default_factory=list)


@dataclass
class ReconStats:
    tool_uses: int = 0
    no_result: int = 0
    failed: int = 0
    bad_input: int = 0
    non_py: int = 0
    flags: dict = field(default_factory=dict)
    unreliable: dict = field(default_factory=dict)  # write calls yielding no version
    state_resets: dict = field(default_factory=dict)  # why tracked state was discarded
    retracted: dict = field(default_factory=dict)  # emitted versions withdrawn, by drift signal

    def bump(self, d: dict, key: str, n: int = 1) -> None:
        d[key] = d.get(key, 0) + n


DISK_MODIFIED_NOTE = "had been modified on disk"
READ_ONLY_SUBAGENTS = {"Explore", "Plan", "claude-code-guide"}
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
_OPERATORS = {"|", "||", "&&", ";", "&", "(", ")", ";;", "|&"}
_REDIRECTS = {">", ">>", ">|", "&>", "&>>"}
_WRAPPERS = {"uv", "poetry", "pdm", "hatch", "timeout", "time", "nice", "command", "exec", "sudo", "env", "xargs"}
_READ_ONLY_CMDS = {"cat", "head", "tail", "grep", "rg", "egrep", "fgrep", "wc", "ls", "echo", "printf", "diff",
                   "find", "awk", "sort", "uniq", "cut", "file", "stat", "less", "nl", "pwd", "cd", "which", "true",
                   "false", "pytest", "pyright", "mypy", "flake8", "pylint", "sleep", "jq", "tree", "du", "bat",
                   "md5", "sha256sum", "md5sum", "basename", "dirname", "realpath", "date", "test", "[", "[[",
                   "source", ".", "export", "type", "readlink", "column", "tr", "comm", "cmp",
                   "tsc", "jest", "vitest", "mocha", "node", "bun", "deno", "ts-node", "tsx", "go", "cargo",
                   "eslint", "prettier", "biome", "rubocop", "golangci-lint", "swiftc", "swift", "xcodebuild",
                   "gradle", "./gradlew", "gradlew", "mvn", "dotnet", "rspec", "bundle", "rake", "make", "cmake",
                   "ctest", "gcc", "g++", "clang", "clang++", "javac", "java", "kotlinc", "php", "phpunit", "ruby",
                   "npm", "pnpm", "yarn", "npx", "bunx", "playwright", "cypress", "staticcheck", "stylelint"}
_READ_ONLY_GIT = {"diff", "log", "show", "status", "blame", "grep", "ls-files", "rev-parse", "branch", "add",
                  "commit", "push", "fetch", "remote", "config", "tag", "check-ignore", "shortlog", "describe",
                  "worktree", "cat-file", "ls-tree", "reflog", "merge-base", "rev-list", "diff-tree"}
_FORMATTERS = {"black", "isort", "autopep8", "yapf", "autoflake", "pre-commit", "pyupgrade", "docformatter",
               "gofmt", "goimports", "gofumpt", "rustfmt", "swiftformat", "swift-format", "clang-format",
               "ktlint", "ktfmt", "google-java-format", "standardrb", "rufo", "php-cs-fixer", "phpcbf",
               "dprint", "shfmt", "stylua"}
_FMT_SUBCOMMAND_TOOLS = {"cargo", "go", "dart", "mix", "zig", "terraform", "deno", "buf", "swift", "dotnet", "gleam"}
# tools that rewrite files only with a flag (or a subcommand) — treated as mutators when present
_FIX_FLAGS = {"--fix", "-w", "--write", "--apply", "--apply-unsafe", "-a", "-A", "--autocorrect", "-i", "--in-place", "-F"}
_FIXABLE = {"eslint", "prettier", "biome", "rubocop", "golangci-lint", "stylelint", "oxlint", "deno", "cargo", "npx", "bunx", "pnpm", "yarn", "npm"}
_READ_ONLY_PY_MODULES = {"pytest", "pyright", "mypy", "py_compile", "flake8", "pylint", "pip", "unittest"}


def _shell_segments(command: str) -> tuple[list[list[str]], list[str]] | None:
    """Split a command into simple-command word lists plus redirect targets (None if unparseable)."""
    lines, out, delim = command.split("\n"), [], None
    for line in lines:  # drop heredoc bodies; the line with "<<" (and any redirect) stays
        if delim is not None:
            if line.strip() == delim:
                delim = None
            continue
        m = _HEREDOC_RE.search(line)
        if m:
            delim = m.group(1)
        out.append(line)
    try:
        lex = shlex.shlex(" ; ".join(out), posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return None
    segments, cur, targets = [], [], []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in _OPERATORS:
            if cur:
                segments.append(cur)
            cur = []
        elif t in _REDIRECTS:
            if i + 1 < len(tokens):
                targets.append(tokens[i + 1])
            i += 1
        elif t in ("<", "<<", "<<-", "<<<", ">&", "<&"):
            i += 1  # skip the operand
        else:
            cur.append(t)
        i += 1
    if cur:
        segments.append(cur)
    return segments, targets


def _command_words(words: list[str]) -> list[str]:
    """Strip env assignments and wrappers ("uv run --with x", "timeout 60") from a simple command."""
    while words:
        w = words[0]
        if "=" in w and not w.startswith("-"):
            words = words[1:]
        elif w.rsplit("/", 1)[-1] in _WRAPPERS:
            words = words[1:]
            if words and words[0] == "run":
                words = words[1:]
            while words and (words[0].startswith("-") or words[0].isdigit() or words[0][:1].isdigit()):
                # option, or option value of "--with pkg" / "timeout 60s"
                words = words[2:] if words[0] in ("--with", "--python", "-p", "--project", "--directory") else words[1:]
        else:
            break
    return words


def _segment_read_only(words: list[str]) -> bool:
    words = _command_words(words)
    if not words:
        return True
    cmd = words[0].rsplit("/", 1)[-1]
    if cmd == "git":
        rest = words[1:]
        while rest and rest[0] in ("-C", "-c", "--git-dir", "--work-tree"):
            rest = rest[2:]
        if rest and rest[0] == "rm":
            return "--cached" in rest
        return bool(rest) and rest[0] in _READ_ONLY_GIT
    if cmd == "sed":
        return not any(w == "-i" or (w.startswith("-i") or (w.startswith("-") and not w.startswith("--") and "i" in w))
                       for w in words[1:] if w.startswith("-"))
    if cmd in ("python", "python3") or re.fullmatch(r"python3\.\d+", cmd):
        if len(words) > 2 and words[1] == "-m":
            return words[2] in _READ_ONLY_PY_MODULES
        return len(words) > 1 and words[1].endswith(".py")  # running a script does not rewrite itself
    if cmd == "ruff":
        return len(words) > 1 and words[1] == "check" and "--fix" not in words
    if cmd in _FIXABLE and (any(w in _FIX_FLAGS for w in words[1:]) or (len(words) > 1 and words[1] in ("fmt", "fix", "format"))):
        return False
    if cmd in ("npm", "pnpm", "yarn", "bun") and len(words) > 2 and words[1] == "run" and re.search(r"fmt|format|fix|prettier|lint:fix", words[2]):
        return False
    return cmd in _READ_ONLY_CMDS


def bash_may_modify(command: str | None, paths: list[str]) -> set[str] | None:
    """Tracked paths a Bash command might have changed (None = possibly all of them).

    Conservative: a simple command that names a tracked file and is not known to be
    read-only is assumed to modify it; repo-wide mutators (git checkout/reset/stash/
    pull/..., formatters, rm -r) without explicit .py targets affect everything.
    """
    if not command:
        return set()
    parsed = _shell_segments(command)

    def names(text: str) -> set[str]:
        return {p for p in paths if p.rsplit("/", 1)[-1] in text}

    if parsed is None:
        return names(command)
    segments, targets = parsed
    hit = {p for t in targets for p in names(t) if t.endswith(p.rsplit("/", 1)[-1])}
    for seg in segments:
        words = _command_words(seg)
        if not words:
            continue
        cmd = words[0].rsplit("/", 1)[-1]
        rest = words[1:]
        while cmd == "git" and rest and rest[0] in ("-C", "-c", "--git-dir", "--work-tree"):
            rest = rest[2:]
        sub = [w for w in rest if not w.startswith("-")]
        mutator = (cmd in _FORMATTERS
                   or (cmd == "ruff" and sub[:1] == ["format"]) or (cmd == "ruff" and "--fix" in words)
                   or (cmd in _FMT_SUBCOMMAND_TOOLS and sub[:1] and sub[0] in ("fmt", "format", "fix"))
                   or (cmd in _FIXABLE and any(w in _FIX_FLAGS for w in words[1:]))
                   or (cmd == "git" and sub[:1] and sub[0] in {"checkout", "switch", "reset", "restore", "stash",
                       "pull", "merge", "rebase", "apply", "am", "cherry-pick", "revert", "clone", "mv"})
                   or (cmd in ("rm", "mv") and any(w.startswith("-") and "r" in w for w in words[1:])))
        if mutator:
            named = {p for w in sub for p in names(w)}
            if named and all(is_code_path(w) for w in sub[1 if cmd in ("git", "ruff") else 0:]):
                hit |= named
            else:
                return None
        elif not _segment_read_only(seg):
            hit |= {p for w in seg for p in names(w)}
    return hit


def pair_results(rows: list[dict]) -> dict[int, dict]:
    """Map index of each tool_use row to its tool_result row: same tool_call_id, closest in
    turn order. Results normally follow their call, but the dataset occasionally records a
    result just before it, so both directions are allowed (after is preferred on ties)."""
    uses: dict[str, list[int]] = {}
    results: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        cid = r.get("tool_call_id")
        if not cid:
            continue
        if r["turn_type"] == "tool_use":
            uses.setdefault(cid, []).append(i)
        elif r["turn_type"] == "tool_result":
            results.setdefault(cid, []).append(i)
    paired: dict[int, dict] = {}
    for cid, us in uses.items():
        rs = results.get(cid)
        if not rs:
            continue
        cand = sorted((abs(int(rows[j]["turn_number"]) - int(rows[i]["turn_number"])), j < i, i, j) for i in us for j in rs)
        used_u, used_r = set(), set()
        for _, _, i, j in cand:
            if i not in used_u and j not in used_r:
                paired[i] = rows[j]
                used_u.add(i)
                used_r.add(j)
    return paired


def _same_text(a: str, b: str) -> bool:
    return a.rstrip("\n") == b.rstrip("\n")  # Read output always ends in a newline


def reconstruct_session(rows: list[dict], stats: ReconStats, commit_rows: list[tuple] | None = None,
                        cstats: CommitStats | None = None) -> list[Version]:
    """rows: one session's tool_use/tool_result rows (Read/Write/Edit/MultiEdit, plus
    Bash/Task/Agent tool_use rows), sorted by turn_number. commit_rows: the session's
    linked commits (see commit_route.certify), certifying versions independently.

    Only versions whose full file text is known with certainty are returned:
      * Edits are replayed on a tracked state; with no state they yield nothing.
      * The state is discarded whenever the file may have changed outside our view
        (a subagent that can edit, a Bash command that may touch it).
      * If drift is detected later (a Read disagrees, old_string doesn't match, or
        Claude Code reports the file changed on disk), every Edit version since the
        state was last confirmed is retracted.
    """
    paired = pair_results(rows)
    files: dict[str, _FileState] = {}
    versions: list[Version] = []

    def forget(fs: _FileState, why: str) -> None:
        if fs.text is not None:
            stats.bump(stats.state_resets, why)
        fs.text, fs.origin, fs.pending = None, None, []

    def drift(fs: _FileState, why: str) -> None:
        live = [v for v in fs.pending if not v.retracted]
        for v in live:
            v.retracted = True
        if live:
            stats.bump(stats.retracted, why, len(live))
        forget(fs, why)

    for i, r in enumerate(rows):
        if r["turn_type"] != "tool_use":
            continue
        tn = r["tool_name"]
        res = paired.get(i)
        if tn in ("Task", "Agent"):
            try:
                sub = json.loads(r.get("tool_input_json") or "{}").get("subagent_type")
            except (json.JSONDecodeError, AttributeError):
                sub = None
            if sub not in READ_ONLY_SUBAGENTS:
                for fs in files.values():
                    forget(fs, "subagent_may_edit")
            continue
        if tn == "Bash":
            tracked = [p for p, fs in files.items() if fs.text is not None]
            hit = bash_may_modify(r.get("command"), tracked) if tracked else set()
            for p in (tracked if hit is None else hit):
                forget(files[p], "bash_may_modify")
            continue
        if tn == "Read":
            if res is None:
                continue
            try:
                fp = json.loads(r.get("tool_input_json") or "{}").get("file_path")
            except (json.JSONDecodeError, AttributeError):
                fp = None
            if not is_code_path(fp):
                continue
            fs = files.setdefault(fp, _FileState())
            text = parse_read_result(r.get("tool_input_json"), res.get("content"))
            if text is not None:
                if fs.text is not None and not _same_text(fs.text, text):
                    drift(fs, "full_read_differs")
                if fs.text is None:
                    fs.text, fs.origin = text, "read"
                fs.pending = []  # confirmed
            elif fs.text is not None and partial_read_agrees(fs.text, res.get("content")) is False:
                drift(fs, "partial_read_differs")
            continue
        if tn not in WRITE_TOOLS:
            continue
        inp = parse_input(tn, r.get("tool_input_json"))
        if inp is None:
            if is_code_path(str(r.get("file_path") or "")):
                stats.bad_input += 1
            continue
        fp = inp["file_path"]
        if not is_code_path(fp):
            stats.non_py += 1
            continue
        stats.tool_uses += 1
        fs = files.setdefault(fp, _FileState())
        if res is None:
            # outcome unknown: the write may have happened, so the tracked text can't be trusted
            stats.no_result += 1
            forget(fs, "write_outcome_unknown")
            continue
        result = res.get("content") or ""
        if not is_success(result):
            stats.failed += 1
            if not KNOWN_FAILURE_RE.match(result):
                forget(fs, "write_outcome_unknown")
            continue
        before = fs.text
        skip = None
        if tn == "Write":
            after, flag = inp["content"], "write"
            if before is None and not result.startswith("File created successfully"):
                # Overwrote an existing file we never saw in full: its comments may predate the agent.
                skip = "overwrite_of_unseen_file"
            fs.text, fs.origin, fs.pending = after, "write", []
            v = Version(r["session_id"], int(r["turn_number"]), fp, tn, after, flag, before, skip)
        else:
            edits = inp["edits"]
            if DISK_MODIFIED_NOTE in result:
                # Claude Code says the file holds changes not in context.
                stats.bump(stats.unreliable, "disk_modified_note")
                drift(fs, "disk_modified_note")
                continue
            if fs.text is None:
                if all(e["old_string"] == "" for e in edits) and result.startswith("File created"):
                    after, flag = apply_edits("", edits), "write"
                else:
                    stats.bump(stats.unreliable, "edit_without_known_full_file")
                    continue
            else:
                after = apply_edits(fs.text, edits)
                if after is None:
                    stats.bump(stats.unreliable, "replay_mismatch")
                    drift(fs, "replay_mismatch")
                    continue
                flag = "replayed" if fs.origin == "write" else "from_read"
            fs.text = after
            v = Version(r["session_id"], int(r["turn_number"]), fp, tn, after, flag, before, skip)
            fs.pending.append(v)
        versions.append(v)
    kept = [v for v in versions if not v.retracted]
    if commit_rows:
        kept = _merge_commit_route(rows, paired, kept, commit_rows, stats, cstats or CommitStats())
    for v in kept:
        stats.bump(stats.flags, v.content_flag)
    return kept


def _merge_commit_route(rows: list[dict], paired: dict[int, dict], kept: list[Version], commit_rows: list[tuple],
                        stats: ReconStats, cstats: CommitStats) -> list[Version]:
    """Add commit-certified versions; where both routes certify a turn they must agree."""
    uses = []
    for i, r in enumerate(rows):
        if r["turn_type"] != "tool_use" or r["tool_name"] not in WRITE_TOOLS:
            continue
        res = paired.get(i)
        inp = parse_input(r["tool_name"], r.get("tool_input_json"))
        if res is None or inp is None or not is_success(res.get("content")) or not is_code_path(inp["file_path"]):
            continue
        if DISK_MODIFIED_NOTE in (res.get("content") or ""):
            continue
        uses.append({"turn": int(r["turn_number"]), "tool": r["tool_name"], "inp": inp, "ts": r.get("timestamp")})
    sid = rows[0]["session_id"] if rows else ""
    times = [r["timestamp"] for r in rows if r.get("timestamp")]
    session_end = max(times, key=lambda t: _commit_ts(t) or datetime.min.replace(tzinfo=timezone.utc)) if times else None
    base_moves = [r["timestamp"] for r in rows if r["turn_type"] == "tool_use" and r["tool_name"] == "Bash"
                  and r.get("timestamp") and BASE_MOVE_RE.search(r.get("command") or "")]
    cert = certify(sid, uses, commit_rows, cstats, session_end, base_moves)
    out: list[Version] = []
    by_turn = {v.turn_number: v for v in kept}
    for v in kept:
        cv = cert.get(v.turn_number)
        if cv is None:
            out.append(v)
        elif same_modulo_redaction(v.content, cv.content):
            # agree (the session text may carry the dataset's REDACTED placeholders; keep it,
            # so the file stays consistent with the redacted conversation)
            v.certified_by = "both"
            out.append(v)
        else:
            stats.bump(stats.unreliable, "routes_disagree")  # neither is trusted
    for t, cv in cert.items():
        if t in by_turn:
            continue
        before = None if cv.before is None else (None if cv.before is UNKNOWN else cv.before)
        skip = "overwrite_of_unseen_file" if cv.before is UNKNOWN else None
        out.append(Version(sid, t, cv.file_path, cv.tool, cv.content, "commit", before, skip, certified_by="commit"))
    out.sort(key=lambda v: v.turn_number)
    return out
