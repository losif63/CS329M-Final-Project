import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))

import pilot  # noqa: E402


def test_wilson_interval():
    p, lo, hi = pilot.wilson(10, 100)
    assert p == 0.1 and 0.05 < lo < 0.1 < hi < 0.18
    assert pilot.wilson(0, 0) == (0.0, 0.0, 0.0)
    assert pilot.wilson(0, 50)[1] == 0.0


def test_kappa():
    assert pilot.kappa(["slop", "not_slop"] * 5, ["slop", "not_slop"] * 5) == 1.0
    assert abs(pilot.kappa(["slop", "slop", "not_slop", "not_slop"], ["slop", "not_slop", "slop", "not_slop"])) < 1e-9  # chance level
    assert pilot.kappa(["slop"] * 3, ["slop"] * 3) is None  # no variation: undefined


def test_source_excerpt_marks_target_lines():
    src = "\n".join(f"line {i}" for i in range(1, 201))
    ex = pilot.source_excerpt(src, 100, 101, window=2).splitlines()
    assert ex[0] == "(lines 98–103 of 200)"
    assert [ln[:2] for ln in ex[1:]] == ["  ", "  ", ">>", ">>", "  ", "  "]
    assert ex[3].endswith("| line 100")


def test_valid_output_and_schema():
    ok = {"label": "slop", "category": "conversation", "evidence_turns": [3], "rationale": "x"}
    assert pilot.valid(ok)
    assert not pilot.valid({**ok, "label": "maybe"})
    assert not pilot.valid({**ok, "category": "vibes"})
    assert set(pilot.SCHEMA["required"]) == set(pilot.SCHEMA["properties"])  # strict-schema backends need this


def test_load_hand_single_labeler_category_optional(tmp_path):
    (tmp_path / "handsheet_key.json").write_text(json.dumps({"h001": "m0007", "h002": "r0003", "h003": "r0009"}))
    with (tmp_path / "handsheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hand_id", "label", "category"])
        w.writerow(["h001", "Slop", "session history"])
        w.writerow(["h002", "not slop", ""])
        w.writerow(["h003", "", ""])  # not labeled yet
    assert pilot.load_hand(tmp_path) == {"you": {
        "m0007": {"label": "slop", "category": "session_history"},
        "r0003": {"label": "not_slop", "category": ""}}}


def test_limit_detection():
    weekly = ('claude exited 1: ..."api_error_status":429,"result":"You\'ve hit your weekly limit · resets Nov 21 at 7pm"')
    assert pilot.LIMIT_RE.search(weekly)
    assert pilot.LIMIT_RE.search("codex exited 1: stream error: 429 Too Many Requests")
    assert not pilot.LIMIT_RE.search("claude returned no structured output: invalid JSON")
