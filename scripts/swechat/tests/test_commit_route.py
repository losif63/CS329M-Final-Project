import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commit_route import CommitStats, certify, file_patches, reverse_apply  # noqa: E402
from reconstruct import ReconStats, reconstruct_session  # noqa: E402

PATCH = """commit abc
Author: x

    msg

diff --git a/src/a.go b/src/a.go
index 1..2 100644
--- a/src/a.go
+++ b/src/a.go
@@ -1,3 +1,4 @@
 package a
-func F() int { return 1 }
+// F returns two now.
+func F() int { return 2 }

"""
PARENT = "package a\nfunc F() int { return 1 }\n\nvar x = 1\n"
COMMITTED = "package a\n// F returns two now.\nfunc F() int { return 2 }\n\nvar x = 1\n"
ABS = "/home/u/proj/src/a.go"


def test_reverse_apply_uses_hunk_counts():
    fps = file_patches(PATCH)
    assert list(fps) == ["src/a.go"]
    # the trailing empty line after the hunk must not be swallowed into it
    assert reverse_apply(COMMITTED, fps["src/a.go"]) == PARENT


def test_reverse_apply_new_file_and_mismatch():
    fps = file_patches("diff --git a/n.go b/n.go\nnew file mode 100644\n--- /dev/null\n+++ b/n.go\n@@ -0,0 +1 @@\n+x\n")
    assert reverse_apply("x\n", fps["n.go"]) == ""
    assert reverse_apply("totally different\n", file_patches(PATCH)["src/a.go"]) is None


def _commit(changes, committed=COMMITTED, patch=PATCH):
    attrs = {"src/a.go": {"attribution": "agent_only", "agent_version": committed, "committed_version": committed}}
    return ("2026-01-01 12:00:00+00", "2026-01-01 12:00:00+00", patch, json.dumps(changes), json.dumps(attrs))


def _change(old, new, pos, ts="2026-01-01T11:00:00Z", sid="s"):
    return {"tool_name": "Edit", "file_path": ABS, "old_string": old, "new_string": new,
            "session_id": sid, "change_position": str(pos), "timestamp": ts}


def _use(turn, old, new):
    return {"turn": turn, "tool": "Edit", "ts": None,
            "inp": {"file_path": ABS, "edits": [{"old_string": old, "new_string": new, "replace_all": False}]}}


def test_certify_two_edits_explained_by_commit():
    e1 = ("func F() int { return 1 }", "func F() int { return 2 }")
    e2 = ("func F() int { return 2 }", "// F returns two now.\nfunc F() int { return 2 }")
    changes = [_change(*e1, 0, "2026-01-01T11:00:00Z"), _change(*e2, 1, "2026-01-01T11:01:00Z")]
    st = CommitStats()
    got = certify("s", [_use(10, *e1), _use(12, *e2)], [_commit(changes)], st)
    assert sorted(got) == [10, 12]
    assert got[10].before == PARENT and got[10].content == "package a\nfunc F() int { return 2 }\n\nvar x = 1\n"
    assert got[12].content == COMMITTED
    assert st.reasons["explained_files"] == 1


def test_certify_rejects_when_commit_has_other_changes():
    # the human also changed `var x` before committing: agent changes don't explain the diff
    committed = COMMITTED.replace("var x = 1", "var x = 9")
    patch = PATCH + "@@ -4,1 +5,1 @@\n-var x = 1\n+var x = 9\n"
    assert reverse_apply(committed, file_patches(patch)["src/a.go"]) == PARENT  # the patch itself is valid
    e1 = ("func F() int { return 1 }", "// F returns two now.\nfunc F() int { return 2 }")
    st = CommitStats()
    assert certify("s", [_use(10, *e1)], [_commit([_change(*e1, 0)], committed, patch)], st) == {}
    assert st.reasons.get("not_explained_by_agent_changes") == 1


def test_certify_ignores_changes_after_commit_time():
    e1 = ("func F() int { return 1 }", "// F returns two now.\nfunc F() int { return 2 }")
    late = _change("var x = 1", "var x = 2", 1, "2026-01-01T13:00:00Z")  # after the commit
    got = certify("s", [_use(10, *e1)], [_commit([_change(*e1, 0), late])], CommitStats())
    assert list(got) == [10]


def _rows_for_session():
    def use(turn, inp, cid, tool="Edit"):
        return {"session_id": "s", "turn_number": turn, "turn_type": "tool_use", "tool_name": tool, "tool_call_id": cid,
                "file_path": ABS, "command": None, "tool_input_json": json.dumps(inp), "content": None, "timestamp": None}

    def res(turn, cid):
        return {"session_id": "s", "turn_number": turn, "turn_type": "tool_result", "tool_name": "Edit", "tool_call_id": cid,
                "file_path": None, "command": None, "tool_input_json": None,
                "content": f"The file {ABS} has been updated successfully.", "timestamp": None}
    e = {"file_path": ABS, "old_string": "func F() int { return 1 }",
         "new_string": "// F returns two now.\nfunc F() int { return 2 }", "replace_all": False}
    return [use(10, e, "c1"), res(11, "c1")], e


def test_reconstruct_adds_commit_only_version():
    rows, e = _rows_for_session()  # an Edit with no Read: the session route can't certify it
    st, cs = ReconStats(), CommitStats()
    vs = reconstruct_session(rows, st, [_commit([_change(e["old_string"], e["new_string"], 0)])], cs)
    assert [(v.turn_number, v.content_flag, v.certified_by) for v in vs] == [(10, "commit", "commit")]
    assert vs[0].before == PARENT and vs[0].content == COMMITTED
    assert st.unreliable == {"edit_without_known_full_file": 1}


E1 = ("func F() int { return 1 }", "// F returns two now.\nfunc F() int { return 2 }")


def test_rebased_commit_refused():
    c = list(_commit([_change(*E1, 0)]))
    c[0] = "2026-01-02 09:00:00+00"  # committed a day after authoring: rebased/amended
    st = CommitStats()
    assert certify("s", [_use(10, *E1)], [tuple(c)], st) == {}
    assert st.reasons["commit_rebased_or_amended"] == 1


def test_commit_after_session_refused():
    st = CommitStats()
    assert certify("s", [_use(10, *E1)], [_commit([_change(*E1, 0)])], st, session_end="2026-01-01T11:30:00Z") == {}
    assert st.reasons["committed_after_session"] == 1


def test_base_move_between_edit_and_commit_refused():
    st = CommitStats()
    got = certify("s", [_use(10, *E1)], [_commit([_change(*E1, 0)])], st,
                  session_end="2026-01-01T12:01:00Z", base_moves=["2026-01-01T11:30:00Z"])
    assert got == {} and st.reasons["base_may_have_moved"] == 1
    # a base move before the first explained change is harmless
    got = certify("s", [_use(10, *E1)], [_commit([_change(*E1, 0)])], CommitStats(),
                  session_end="2026-01-01T12:01:00Z", base_moves=["2026-01-01T10:00:00Z"])
    assert list(got) == [10]


def test_base_move_regex_distinguishes_harmless_git_commands():
    from commit_route import BASE_MOVE_RE
    for cmd in ["git pull --ff-only", "git stash pop", "git checkout main", "git reset --hard HEAD~1", "git rebase main"]:
        assert BASE_MOVE_RE.search(cmd), cmd
    for cmd in ["git checkout -b feat/x", "git stash -q -u", "git checkout -- a.go", "git reset -q", "git switch -c new"]:
        assert not BASE_MOVE_RE.search(cmd), cmd


def test_same_modulo_redaction():
    from commit_route import same_modulo_redaction
    real = "x = call(\n    worktree_path=_UNRESOLVED,\n)\nurl = 'https://farcaster.xyz/a/b'\n"
    assert same_modulo_redaction("x = call(\n    REDACTED,\n)\nurl = 'https://farcaster.REDACTED'\n", real)
    assert not same_modulo_redaction("x = call(\n    REDACTED,\n)\nurl = 'https://other.REDACTED'\n", real)
    assert not same_modulo_redaction("x = call(y)\n", real)
