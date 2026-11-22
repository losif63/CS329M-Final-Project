import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation import build_items, items_before, render_text, tool_target  # noqa: E402

TURNS = [
    (1, "user_prompt", None, "When launching gemini, the banner is printed twice.", None, None),
    (2, "assistant_thinking", None, "", None, None),
    (3, "tool_use", "Read", json.dumps({"file_path": "/home/u/proj/cmd/cli/lifecycle.go"}), "/home/u/proj/cmd/cli/lifecycle.go", None),
    (4, "tool_result", "Read", "1\tpackage cli\n…huge file…", None, None),
    (5, "tool_use", "Grep", json.dumps({"pattern": "SessionStart", "path": "cmd"}), None, None),
    (6, "tool_result", "Grep", "many matches", None, None),
    (7, "tool_use", "Bash", json.dumps({"command": "go test ./cmd/...\necho done"}), None, "go test ./cmd/...\necho done"),
    (8, "tool_result", "Bash", "ok", None, None),
    (9, "assistant_response", None, "Found it: Gemini fires SessionStart for startup and resume.", None, None),
    (10, "attachment", None, "{...}", None, None),
    (11, "user_prompt", None, "Great, add a test.", None, None),
    (12, "tool_use", "Edit", json.dumps({"file_path": "/home/u/proj/cmd/cli/lifecycle_test.go"}), "/home/u/proj/cmd/cli/lifecycle_test.go", None),
    (13, "tool_result", "Edit", "The file ... has been updated successfully.", None, None),
    (14, "tool_use", "Write", json.dumps({"file_path": "/home/u/proj/cmd/cli/x.go"}), "/home/u/proj/cmd/cli/x.go", None),
]


def test_build_items_collapses_agent_work_and_drops_results():
    items = build_items(TURNS)
    assert [it["k"] for it in items] == ["user", "work", "assistant", "user", "work"]
    work = items[1]
    assert work["h"] == [2]
    assert [c[:2] for c in work["c"]] == [["Read", "…/cmd/cli/lifecycle.go"],
                                          ["Grep", '"SessionStart" in cmd'], ["Bash", "go test ./cmd/... …"]]
    assert "huge file" not in json.dumps(items)  # tool results never appear


def test_items_before_clips_work_spanning_the_write():
    items = items_before(build_items(TURNS), 14)  # the Write at turn 14 produced the version
    assert [it["k"] for it in items] == ["user", "work", "assistant", "user", "work"]
    assert [c[0] for c in items[-1]["c"]] == ["Edit"]  # the Write itself is not part of its own context
    assert [it["k"] for it in items_before(build_items(TURNS), 9)] == ["user", "work"]


def test_render_text_and_cap():
    text = render_text(items_before(build_items(TURNS), 14))
    assert text.splitlines()[0] == "[user] When launching gemini, the banner is printed twice."
    assert text.splitlines()[1] == ('[agent: thinking · 3 tool calls: Read …/cmd/cli/lifecycle.go · '
                                    'Grep "SessionStart" in cmd · Bash go test ./cmd/... …]')
    capped = render_text(items_before(build_items(TURNS), 14), max_total_chars=80)
    assert capped.startswith("[… ") and capped.endswith("lifecycle_test.go]")


def test_tool_target_variants():
    assert tool_target("WebFetch", json.dumps({"url": "https://example.com/a"})) == "https://example.com/a"
    assert tool_target("Agent", json.dumps({"description": "Explore auth flow", "subagent_type": "Explore"})) == "Explore auth flow"
    assert tool_target("TodoWrite", json.dumps({"todos": [1, 2, 3]})) == "3 todos"
    assert tool_target("Bash", None, command="x" * 200).endswith("…")
