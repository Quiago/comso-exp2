"""
tests/test_loader.py
Tests for pir.loader — annotation parsing and target selection.
Run: pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import tempfile
import os

from pir.loader import load_annotations, select_target, annotation_summary
from pir.types import AssemblyAction


# ── Fixtures ──────────────────────────────────────────────

SAMPLE_CSV = """\
7267,7497,attach,interior,chassis,correct,
8419,8736,detach,interior,chassis,mistake,shouldn't have happened
10039,10355,attach,body,chassis,mistake,wrong order
14658,15729,attach,arm connector,chassis,mistake,wrong order
16848,18532,attach,arm connector,boom,mistake,wrong position
18532,19342,attach,arm connector,chassis,correct,
"""

@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "test_sequence.csv"
    p.write_text(SAMPLE_CSV)
    return str(p)


# ── Tests ─────────────────────────────────────────────────

def test_load_annotations_count(csv_path):
    actions = load_annotations(csv_path)
    assert len(actions) == 6

def test_load_annotations_fields(csv_path):
    actions = load_annotations(csv_path)
    first = actions[0]
    assert first.start == 7267
    assert first.end == 7497
    assert first.verb == "attach"
    assert first.obj1 == "interior"
    assert first.obj2 == "chassis"
    assert first.status == "correct"

def test_load_annotations_mistake_flag(csv_path):
    actions = load_annotations(csv_path)
    mistakes = [a for a in actions if a.is_mistake]
    assert len(mistakes) == 4

def test_load_annotations_remark(csv_path):
    actions = load_annotations(csv_path)
    wrong_pos = [a for a in actions if a.remark == "wrong position"]
    assert len(wrong_pos) == 1
    assert wrong_pos[0].obj1 == "arm connector"
    assert wrong_pos[0].obj2 == "boom"

def test_select_target_auto_prefers_wrong_position(csv_path):
    actions = load_annotations(csv_path)
    target = select_target(actions, "auto")
    assert target.remark == "wrong position"

def test_select_target_explicit_type(csv_path):
    actions = load_annotations(csv_path)
    target = select_target(actions, "wrong_order")
    assert "wrong order" in target.remark

def test_select_target_fallback_to_longest(csv_path):
    actions = load_annotations(csv_path)
    # Request type that doesn't exist → should use longest mistake
    target = select_target(actions, "nonexistent_type")
    # wrong_position mistake is 16848-18532 = 1684 frames — longest
    assert target.remark == "wrong position"

def test_annotation_summary(csv_path):
    actions = load_annotations(csv_path)
    s = annotation_summary(actions)
    assert s["total_actions"] == 6
    assert s["total_mistakes"] == 4
    assert "wrong position" in s["mistake_types"]
    assert "arm connector" in s["components"]

def test_describe_format(csv_path):
    actions = load_annotations(csv_path)
    target = select_target(actions, "auto")
    desc = target.describe(fps=60.0)
    assert "MISTAKE" in desc
    assert "wrong position" in desc
    assert "attach" in desc

def test_load_skips_malformed_rows(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("not,enough,cols\n7267,7497,attach,interior,chassis,correct,\n")
    actions = load_annotations(str(p))
    assert len(actions) == 1
