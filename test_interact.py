"""
tests/test_interact.py
Tests for pir.interact — physical temporal graph building.
Run: pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pir.types import AssemblyAction
from pir.interact import (
    reconstruct_state_at,
    build_graph_context,
)


def _make_actions():
    return [
        AssemblyAction(100, 200, "attach", "interior", "chassis", "correct"),
        AssemblyAction(300, 400, "attach", "cabin", "interior", "correct"),
        AssemblyAction(500, 600, "attach", "arm connector", "chassis", "mistake", "wrong order"),
        AssemblyAction(700, 800, "detach", "arm connector", "chassis", "correction"),
        AssemblyAction(900, 1000, "attach", "arm connector", "boom", "mistake", "wrong position"),
        AssemblyAction(1100, 1200, "attach", "arm connector", "chassis", "correct"),
    ]


def test_reconstruct_state_at_empty():
    actions = _make_actions()
    state = reconstruct_state_at(actions, before_frame=50)
    assert state.attached == {}


def test_reconstruct_state_at_mid():
    actions = _make_actions()
    # After attach(interior→chassis) and attach(cabin→interior), before the mistake
    state = reconstruct_state_at(actions, before_frame=450)
    assert state.attached["interior"] == "chassis"
    assert state.attached["cabin"] == "interior"
    assert "arm connector" not in state.attached


def test_reconstruct_state_after_detach():
    actions = _make_actions()
    # After detach(arm connector→chassis)
    state = reconstruct_state_at(actions, before_frame=850)
    assert "arm connector" not in state.attached


def test_build_graph_context_contains_anomaly():
    actions = _make_actions()
    target = actions[4]  # wrong position
    ctx = build_graph_context(actions, target, fps=60.0)
    assert "ANOMALY DETECTED" in ctx.text
    assert "wrong position" in ctx.text


def test_build_graph_context_contains_sequence():
    actions = _make_actions()
    target = actions[4]
    ctx = build_graph_context(actions, target, fps=60.0)
    assert "COMPLETE ASSEMBLY SEQUENCE" in ctx.text
    assert "attach(interior → chassis)" in ctx.text


def test_build_graph_context_cascade():
    actions = _make_actions()
    target = actions[2]  # wrong order — has things after it
    ctx = build_graph_context(actions, target, fps=60.0)
    assert "SUBSEQUENT ACTIONS" in ctx.text


def test_build_graph_context_physical_state():
    actions = _make_actions()
    target = actions[4]  # wrong position at frame 900
    ctx = build_graph_context(actions, target, fps=60.0)
    # At frame 900: interior→chassis, cabin→interior attached; arm connector detached
    assert "interior" in ctx.text
    assert "chassis" in ctx.text


def test_graph_context_fps():
    actions = _make_actions()
    target = actions[4]
    ctx = build_graph_context(actions, target, fps=60.0)
    assert ctx.fps == 60.0
    assert ctx.target == target
