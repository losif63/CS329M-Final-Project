import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comments import drop_reason, extract_comments, is_code_line, new_comments, prose_lines  # noqa: E402
from reconstruct import (ReconStats, apply_edits, is_success, parse_read_result,  # noqa: E402
                         reconstruct_session)
from sentences import split_comment, split_text  # noqa: E402


# ---------------------------------------------------------------- edit replay

def E(old, new, ra=False):
    return {"old_string": old, "new_string": new, "replace_all": ra}


def test_apply_edit_unique_occurrence():
    assert apply_edits("a = 1\nc = 3\n", [E("a = 1", "b = 2")]) == "b = 2\nc = 3\n"


def test_apply_edit_replace_all():
    assert apply_edits("a = 1\na = 1\n", [E("a = 1", "b = 2", True)]) == "b = 2\nb = 2\n"


def test_apply_edits_in_order():
    assert apply_edits("x\n", [E("x", "y"), E("y", "z")]) == "z\n"


def test_apply_edit_old_string_not_found():
    assert apply_edits("x = 1\n", [E("nope", "y")]) is None


def _use(turn, tool, inp, cid):
    return {"session_id": "s", "turn_number": turn, "turn_type": "tool_use", "tool_name": tool,
            "tool_call_id": cid, "file_path": inp.get("file_path"), "tool_input_json": json.dumps(inp),
            "content": json.dumps(inp)}


def _res(turn, cid, content, tool="Edit"):
    return {"session_id": "s", "turn_number": turn, "turn_type": "tool_result", "tool_name": tool,
            "tool_call_id": cid, "file_path": None, "tool_input_json": None, "content": content}


OK_EDIT = "The file /p/a.py has been updated successfully."


def test_reconstruct_replay_miss_retracts_unconfirmed_and_drops_state():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1\n"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
        _use(3, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 2  # two"}, "c2"),
        _res(4, "c2", OK_EDIT),
        _use(5, "Edit", {"file_path": "/p/a.py", "old_string": "missing", "new_string": "# new"}, "c3"),
        _res(6, "c3", OK_EDIT),
        _use(7, "Edit", {"file_path": "/p/a.py", "old_string": "x = 2", "new_string": "x = 3"}, "c4"),
        _res(8, "c4", OK_EDIT),
    ]
    st = ReconStats()
    vs = reconstruct_session(rows, st)
    # the mismatch proves the state had drifted at some point after the Write, so the
    # replayed edit at turn 3 is retracted and the edit after the mismatch is not trusted
    assert [v.content_flag for v in vs] == ["write"]
    assert st.retracted == {"replay_mismatch": 1}
    assert st.unreliable == {"replay_mismatch": 1, "edit_without_known_full_file": 1}


def test_ambiguous_old_string_is_a_mismatch():
    assert apply_edits("a\na\n", [E("a", "b")]) is None
    assert apply_edits("a\na\n", [E("a", "b", True)]) == "b\nb\n"


def test_disk_modified_note_drops_version():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1\n"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
        _use(3, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 2"}, "c2"),
        _res(4, "c2", OK_EDIT + " (note: the file had been modified on disk since you last read it)"),
    ]
    st = ReconStats()
    assert [v.content_flag for v in reconstruct_session(rows, st)] == ["write"]
    assert st.unreliable == {"disk_modified_note": 1}


def test_partial_read_mismatch_discards_state():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1\ny = 2\n"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
        _use(3, "Read", {"file_path": "/p/a.py", "offset": 2, "limit": 1}, "r1"),
        _res(4, "r1", "     2→y = 99\n", "Read"),  # changed by something else
        _use(5, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 5"}, "c2"),
        _res(6, "c2", OK_EDIT),
    ]
    st = ReconStats()
    assert len(reconstruct_session(rows, st)) == 1
    assert st.state_resets == {"partial_read_differs": 1}


def test_overwrite_of_unseen_file_is_not_attributable():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "# old comment\n"}, "c1"),
        _res(2, "c1", "The file /p/a.py has been overwritten successfully.", "Write"),
    ]
    vs = reconstruct_session(rows, ReconStats())
    assert vs[0].skip_reason == "overwrite_of_unseen_file"


def test_reconstruct_skips_failed_and_uses_read():
    read_out = "     1→x = 1\n     2→y = 2\n"
    rows = [
        _use(1, "Read", {"file_path": "/p/a.py"}, "r1"),
        _res(2, "r1", read_out, "Read"),
        _use(3, "Edit", {"file_path": "/p/a.py", "old_string": "y = 2", "new_string": "y = 3"}, "c1"),
        _res(4, "c1", "<tool_use_error>String to replace not found in file.</tool_use_error>"),
        _use(5, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 5"}, "c2"),
        _res(6, "c2", OK_EDIT),
    ]
    st = ReconStats()
    vs = reconstruct_session(rows, st)
    assert st.failed == 1
    assert len(vs) == 1 and vs[0].content_flag == "from_read"
    assert vs[0].content == "x = 5\ny = 2\n"


def test_reused_tool_call_ids_pair_with_nearest_result():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "a\n"}, "functions.Write:0"),
        _res(2, "functions.Write:0", "File created successfully at: /p/a.py", "Write"),
        _use(3, "Write", {"file_path": "/p/a.py", "content": "b\n"}, "functions.Write:0"),
        _res(4, "functions.Write:0", "The user doesn't want to proceed with this tool use.", "Write"),
    ]
    vs = reconstruct_session(rows, ReconStats())
    assert [v.content for v in vs] == ["a\n"]


def test_is_success_patterns():
    assert is_success("The file /x.py has been updated successfully.")
    assert is_success("The file /x.py has been overwritten successfully.")
    assert is_success("File created successfully at: /x.py")
    assert not is_success("<tool_use_error>File has not been read yet.</tool_use_error>")
    assert not is_success("The user doesn't want to proceed with this tool use.")
    assert not is_success("PreToolUse:Edit hook error: [python3 x]")


def test_parse_read_result_formats_and_partial():
    assert parse_read_result('{"file_path": "a.py"}', "1\tx\n2\ty\n") == "x\ny\n"
    reminder = "     1→x\n\n<system-reminder>\nWhenever you read a file...\n</system-reminder>\n"
    assert parse_read_result('{"file_path": "a.py"}', reminder) == "x\n"
    assert parse_read_result('{"file_path": "a.py", "limit": 10}', "1\tx\n") is None
    assert parse_read_result('{"file_path": "a.py", "offset": 5}', "5\tx\n") is None
    assert parse_read_result('{"file_path": "a.py"}', "File content (30000 tokens) exceeds maximum") is None


# ---------------------------------------------------------------- grouping

SRC = '''\
"""Module doc."""
# First line of a block
# second line of the block
x = 1  # inline note

# separate block after a blank line
def f():
    """Do the thing.

    More detail here.
    """
    # Lone block
    return 1
'''


def test_block_vs_inline_grouping():
    comments, fb = extract_comments(SRC)
    assert not fb
    by_kind = [(c.kind, c.line_start, c.line_end) for c in comments]
    assert ("block", 2, 3) in by_kind
    assert ("inline", 4, 4) in by_kind
    assert ("block", 6, 6) in by_kind
    assert ("block", 12, 12) in by_kind
    blk = next(c for c in comments if c.line_start == 2)
    assert blk.text == "First line of a block\nsecond line of the block"
    docs = [c for c in comments if c.kind == "docstring"]
    assert [(d.line_start, d.line_end) for d in docs] == [(1, 1), (8, 10)]
    assert docs[1].text == "Do the thing.\n\nMore detail here."


def test_hash_inside_string_is_not_comment():
    comments, _ = extract_comments('s = "a # b"\n')
    assert comments == []


def test_linescan_fallback_on_broken_source():
    comments, fb = extract_comments("def f(:\n    # still found\n    x = 'a # no'  # yes\n")
    assert fb
    assert [(c.kind, c.text) for c in comments] == [("block", "still found"), ("inline", "yes")]


# ---------------------------------------------------------------- diffing

def test_added_comment_diffing_multiset():
    before = "# keep\nx = 1\n# dup\n"
    after = "# keep\nx = 1  # new inline\n# dup\ny = 2\n# dup\n"
    cb, _ = extract_comments(before)
    ca, _ = extract_comments(after)
    added, unsure = new_comments(ca, cb, before, after)
    assert unsure == 0
    assert sorted(c.text for c in added) == ["dup", "new inline"]
    assert next(c for c in added if c.text == "dup").line_start == 5  # the changed occurrence


def test_no_prior_state_all_new():
    ca, _ = extract_comments("# a\nx = 1  # b\n")
    assert len(new_comments(ca, None)[0]) == 2


# ---------------------------------------------------------------- filters

def test_commented_out_code_filter():
    assert is_code_line("x = compute(a, b)")
    assert is_code_line("return foo.bar()")
    assert is_code_line("import os")
    assert is_code_line("if x > 0:")
    assert not is_code_line("TODO")
    assert not is_code_line("TODO: fix this later")
    assert not is_code_line("Note: this is slow")
    assert not is_code_line("Compute the value (cached).")
    assert not is_code_line("See foo.bar() for details")


def _one(src):
    return extract_comments(src)[0][0]


def test_drop_reasons():
    assert drop_reason(_one("#!/usr/bin/env python\n")) == "shebang_encoding"
    assert drop_reason(_one("# -*- coding: utf-8 -*-\n")) == "shebang_encoding"
    assert drop_reason(_one("x = 1  # noqa: E501\n")) == "directive"
    assert drop_reason(_one("x = 1  # type: ignore[attr-defined]\n")) == "directive"
    assert drop_reason(_one("# pylint: disable=foo\n")) == "directive"
    assert drop_reason(_one("# fmt: off\n")) == "directive"
    assert drop_reason(_one("# ----------------\n")) == "separator"
    assert drop_reason(_one("# Copyright 2024 Acme Inc.\n# Licensed under the MIT License.\n")) == "license"
    assert drop_reason(_one("# x = foo()\n# y = bar(x)\n")) == "commented_out_code"
    assert drop_reason(_one("#\n")) == "empty"
    assert drop_reason(_one("# Retry because the API is flaky.\n")) is None


def test_docstring_skips_doctest_and_code_blocks():
    src = 'def f():\n    """Add one.\n\n    >>> f(1)\n    2\n\n    Example::\n\n        y = f(x)\n\n    Done.\n    """\n'
    doc = next(c for c in extract_comments(src)[0] if c.kind == "docstring")
    lines, removed = prose_lines(doc)
    text = " ".join(t for _, t in lines if t.strip())
    assert ">>>" not in text and "y = f(x)" not in text
    assert "Add one." in text and "Done." in text
    assert removed["doctest"] == 2 and removed["docstring_code_block"] == 1


# ---------------------------------------------------------------- sentences

def _split(t):
    return [t[a:b] for a, b in split_text(t)]


def test_sentence_split_paths_versions_dotted():
    t = ("Reads config from ~/.config/app/settings.toml first. Requires Python 3.6 or "
         "torch 2.x to run. Call os.path.join() here. See https://example.com/a.b?c=1 for more.")
    assert _split(t) == [
        "Reads config from ~/.config/app/settings.toml first.",
        "Requires Python 3.6 or torch 2.x to run.",
        "Call os.path.join() here.",
        "See https://example.com/a.b?c=1 for more.",
    ]


def test_sentence_split_backticks():
    assert _split("Use `foo.bar(). baz` here. Then stop.") == ["Use `foo.bar(). baz` here.", "Then stop."]


def test_split_comment_line_mapping_and_paragraphs():
    lines = [(10, "First sentence spans"), (11, "two lines. Second one."), (12, ""),
             (13, "Args:"), (14, "x: the input value."), (15, "y: another one.")]
    assert split_comment(lines) == [
        ("First sentence spans two lines.", 10, 11),
        ("Second one.", 11, 11),
        ("x: the input value.", 14, 14),
        ("y: another one.", 15, 15),
    ]


# ---------------------------------------------------------------- regressions from the 50-session audit

def test_directive_with_rationale_keeps_rationale():
    c = _one("x = f()  # noqa: BLE001 - diagnostics must never break the read\n")
    assert drop_reason(c) is None
    lines, _ = prose_lines(c)
    assert [t for _, t in lines] == ["diagnostics must never break the read"]
    assert drop_reason(_one("import x  # noqa: F401\n")) == "directive"


def test_prose_with_call_or_data_line():
    assert not is_code_line("Set in auto_context()")
    assert is_code_line('"ast_grep":      {"available": bool, "version": str|null},')


def test_shebang_line_removed_from_block():
    c = _one("#!/usr/bin/env python3\n# Seed the demo database.\n")
    assert drop_reason(c) is None
    assert [t for _, t in prose_lines(c)[0]] == ["Seed the demo database."]


def test_sentence_split_leading_dot_names():
    assert _split("Probe all .co files for kernels. Then call .strip() on it.") == [
        "Probe all .co files for kernels.", "Then call .strip() on it."]


def test_header_lines_are_not_sentences():
    lines = [(1, "Probe quantization paths."), (2, "Tests:"), (3, "1. torch path"), (4, "2. timing")]
    assert [s for s, _, _ in split_comment(lines)] == ["Probe quantization paths.", "1. torch path", "2. timing"]


def test_decorated_header_and_prose_with_list_are_not_code():
    assert not is_code_line("=== EVALUATION ===")
    assert not is_code_line('for filtering: ["deploy", "frontend", "api", etc.]')
    assert drop_reason(_one("# === EVALUATION ===\n")) is None


def test_double_backtick_spans_do_not_swallow_prose():
    t = ("Resolved against ``~/.cfg/repos.yml`` (ABA-232); runs ``tool`` from anywhere, not inside a repo. "
         "The ``grade`` subcommand reads logs.")
    assert len(_split(t)) == 2


def test_url_path_with_query_is_protected():
    assert _split("Calls GET /api/search?q= for lookups. Done.") == ["Calls GET /api/search?q= for lookups.", "Done."]


def test_unicode_separator_and_decorated_heading():
    assert drop_reason(_one("# ──────────────\n")) == "separator"
    c = _one("# --- Budget caps (USD/day) ---\n# Rex is billed by subscription.\n")
    assert [s for s, _, _ in split_comment(prose_lines(c)[0])] == ["Budget caps (USD/day)", "Rex is billed by subscription."]
    c = _one("# ── consensus → ruliad ─────────\n")
    assert [s for s, _, _ in split_comment(prose_lines(c)[0])] == ["consensus → ruliad"]


def test_docstring_usage_section_commands_removed():
    src = 'def f():\n    """Merge things.\n\n    CLI:\n\n      python3 merge.py \\\\\n          --root /abs/path\n\n    Done.\n    """\n'
    doc = next(c for c in extract_comments(src)[0] if c.kind == "docstring")
    sents = [s for s, _, _ in split_comment(prose_lines(doc)[0])]
    assert sents == ["Merge things.", "Done."]


def test_moved_duplicate_comment_is_unattributable():
    before = "# dup\nx = 1\n"
    after = "# dup\nx = 1\n\n# dup\ny = 2\n"
    ca, cb = extract_comments(after)[0], extract_comments(before)[0]
    added, unsure = new_comments(ca, cb, before, after)
    assert [c.line_start for c in added] == [4] and unsure == 0
    # same text appears again but the diff can't tell which copy is new
    before2 = "x = 1\n# dup\n"
    after2 = "# dup\nx = 1\n# dup\n"
    added, unsure = new_comments(extract_comments(after2)[0], extract_comments(before2)[0], before2, after2)
    assert len(added) + unsure == 1


def test_mixed_code_and_prose_block_dropped():
    assert drop_reason(_one("# Retry because the API is flaky.\n# Also see the docs.\n# x = retry(3)\n")) == "mixed_code_and_prose"


# ---------------------------------------------------------------- outside-modification guards

from reconstruct import bash_may_modify  # noqa: E402


def test_bash_may_modify():
    P = ["/r/pkg/foo.py", "/r/pkg/bar.py"]
    assert bash_may_modify("cat pkg/foo.py | head -20", P) == set()
    assert bash_may_modify("uv run pytest tests/test_foo.py -q 2>&1 | tail -5", P) == set()
    assert bash_may_modify("python pkg/foo.py --help", P) == set()
    assert bash_may_modify("git diff pkg/foo.py", P) == set()
    assert bash_may_modify("sed -i 's/a/b/' pkg/foo.py", P) == {"/r/pkg/foo.py"}
    assert bash_may_modify("echo x > pkg/bar.py", P) == {"/r/pkg/bar.py"}
    assert bash_may_modify("cp /tmp/x.py pkg/foo.py", P) == {"/r/pkg/foo.py"}
    assert bash_may_modify("git checkout -- .", P) is None
    assert bash_may_modify("ruff format .", P) is None
    assert bash_may_modify("ls -la", P) == set()


def _bash(turn, cmd):
    return {"session_id": "s", "turn_number": turn, "turn_type": "tool_use", "tool_name": "Bash",
            "tool_call_id": f"b{turn}", "file_path": None, "command": cmd,
            "tool_input_json": json.dumps({"command": cmd}), "content": None}


def test_bash_modification_discards_state():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1\n"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
        _bash(3, "black a.py"),
        _use(4, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 2"}, "c2"),
        _res(5, "c2", OK_EDIT),
    ]
    st = ReconStats()
    assert [v.content_flag for v in reconstruct_session(rows, st)] == ["write"]
    assert st.state_resets == {"bash_may_modify": 1}


def test_editing_subagent_discards_state_but_explore_does_not():
    def task(turn, sub):
        return _use(turn, "Task", {"subagent_type": sub, "prompt": "..."}, f"t{turn}")
    base = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1\n"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
    ]
    edit = [_use(4, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 2"}, "c2"),
            _res(5, "c2", OK_EDIT)]
    assert len(reconstruct_session(base + [task(3, "Explore")] + edit, ReconStats())) == 2
    assert len(reconstruct_session(base + [task(3, "general-purpose")] + edit, ReconStats())) == 1


def test_drift_detected_by_full_read_retracts_unconfirmed_edits():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1\n"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
        _use(3, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 2"}, "c2"),
        _res(4, "c2", OK_EDIT),
        _use(5, "Read", {"file_path": "/p/a.py"}, "r1"),
        _res(6, "r1", "     1→import os\n     2→x = 2\n", "Read"),  # someone added a line
        _use(7, "Edit", {"file_path": "/p/a.py", "old_string": "x = 2", "new_string": "x = 3"}, "c3"),
        _res(8, "c3", OK_EDIT),
    ]
    st = ReconStats()
    vs = reconstruct_session(rows, st)
    assert [(v.turn_number, v.content_flag) for v in vs] == [(1, "write"), (7, "from_read")]
    assert vs[1].content == "import os\nx = 3\n"
    assert st.retracted == {"full_read_differs": 1}


def test_trailing_newline_difference_is_not_drift():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
        _use(3, "Read", {"file_path": "/p/a.py"}, "r1"),
        _res(4, "r1", "     1→x = 1\n", "Read"),
    ]
    st = ReconStats()
    reconstruct_session(rows, st)
    assert st.retracted == {} and st.state_resets == {}


def test_bash_may_modify_shell_parsing():
    P = ["/r/pkg/foo.py", "/r/tests/test_x.py"]
    assert bash_may_modify('grep -n "a\\|b\\|def" pkg/foo.py | head -40', P) == set()
    assert bash_may_modify("git -C /r add pkg/foo.py && git commit -m \"$(cat <<'EOF'\nfix foo.py\nEOF\n)\"", P) == set()
    assert bash_may_modify("uv run --with pytest pytest tests/test_x.py -v 2>&1 | tail -30", P) == set()
    assert bash_may_modify('echo "black is a color" && cat pkg/foo.py', P) == set()
    assert bash_may_modify("cat <<EOF > pkg/foo.py\nx = 1\nEOF", P) == {"/r/pkg/foo.py"}
    assert bash_may_modify("ruff format pkg/foo.py", P) == {"/r/pkg/foo.py"}
    assert bash_may_modify("git rm -r --cached build/", P) == set()
    assert bash_may_modify("some-tool submit pkg/foo.py", P) == {"/r/pkg/foo.py"}  # unknown tool: assume it may write
    assert bash_may_modify("python -c \"open('pkg/foo.py').read()\"", P) == {"/r/pkg/foo.py"}
    assert bash_may_modify("git -C /r checkout main", P) is None
    assert bash_may_modify("curl -o pkg/foo.py https://x", P) == {"/r/pkg/foo.py"}


def test_result_recorded_before_its_call_is_paired():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1\ny = 2\n"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
        _res(3, "c2", OK_EDIT),  # dataset ordering quirk: result before its tool_use
        _use(4, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 5"}, "c2"),
        _use(5, "Edit", {"file_path": "/p/a.py", "old_string": "y = 2", "new_string": "y = 6"}, "c3"),
        _res(6, "c3", OK_EDIT),
    ]
    vs = reconstruct_session(rows, ReconStats())
    assert [v.content for v in vs][-1] == "x = 5\ny = 6\n"


def test_unknown_write_outcome_discards_state():
    rows = [
        _use(1, "Write", {"file_path": "/p/a.py", "content": "x = 1\ny = 2\n"}, "c1"),
        _res(2, "c1", "File created successfully at: /p/a.py", "Write"),
        _use(3, "Edit", {"file_path": "/p/a.py", "old_string": "x = 1", "new_string": "x = 5"}, "c2"),  # no result
        _use(5, "Edit", {"file_path": "/p/a.py", "old_string": "y = 2", "new_string": "y = 6"}, "c3"),
        _res(6, "c3", OK_EDIT),
    ]
    st = ReconStats()
    assert [v.turn_number for v in reconstruct_session(rows, st)] == [1]
    assert st.state_resets == {"write_outcome_unknown": 1}
    # a known failure (tool error) means nothing was written: tracking continues
    rows.insert(3, _res(4, "c2", "<tool_use_error>String to replace not found in file.</tool_use_error>"))
    vs = reconstruct_session(rows, ReconStats())
    assert [v.content for v in vs][-1] == "x = 1\ny = 6\n"
