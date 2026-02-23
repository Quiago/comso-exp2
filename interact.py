"""
pir/interact.py
===============
INTERACT layer — Physical Temporal Graph builder.

Takes the annotation sequence and builds a structured representation
of the physical assembly state at the time of the detected mistake.
This is the "subtitles" that Cosmos needs to reason about physical causality.

No hardcoded domain knowledge. Everything is derived from the CSV.
"""

from __future__ import annotations
from typing import List, Dict, Optional

from pir.types import AssemblyAction, PhysicalState, GraphContext


# ══════════════════════════════════════════════════════════
# STATE RECONSTRUCTION
# ══════════════════════════════════════════════════════════

def reconstruct_state_at(
    actions: List[AssemblyAction],
    before_frame: int,
) -> PhysicalState:
    """
    Replay the action sequence up to before_frame and return
    the resulting physical state (what is attached to what).
    """
    attached: Dict[str, str] = {}
    for a in actions:
        if a.end > before_frame:
            break
        if a.verb == "attach":
            attached[a.obj1] = a.obj2
        elif a.verb == "detach":
            attached.pop(a.obj1, None)

    return PhysicalState(
        frame=before_frame,
        attached=dict(attached),
    )


# ══════════════════════════════════════════════════════════
# GRAPH SERIALIZATION
# ══════════════════════════════════════════════════════════

def build_graph_context(
    actions: List[AssemblyAction],
    target: AssemblyAction,
    fps: float,
    window_before: int = 5,
    window_after: int = 6,
) -> GraphContext:
    """
    Build the full physical temporal graph context for Cosmos.
    Returns a GraphContext with the serialized text and structured data.
    """
    state = reconstruct_state_at(actions, target.start)
    components = sorted(
        set(a.obj1 for a in actions) | set(a.obj2 for a in actions)
    )

    lines = _header(components)
    lines += _full_sequence(actions, fps)
    lines += _anomaly_block(target, fps)
    lines += _preceding_context(actions, target, fps, window_before)
    lines += _cascade_context(actions, target, fps, window_after)
    lines += _physical_state_block(state)

    return GraphContext(
        text="\n".join(lines),
        target=target,
        all_actions=actions,
        physical_state=state,
        fps=fps,
    )


def _header(components: List[str]) -> List[str]:
    return [
        "PHYSICAL TEMPORAL GRAPH",
        "=" * 54,
        f"Assembly components: {', '.join(components)}",
        "",
    ]


def _full_sequence(actions: List[AssemblyAction], fps: float) -> List[str]:
    lines = ["COMPLETE ASSEMBLY SEQUENCE:"]
    for a in actions:
        lines.append(f"  {a.describe(fps)}")
    lines.append("")
    return lines


def _anomaly_block(target: AssemblyAction, fps: float) -> List[str]:
    return [
        "ANOMALY DETECTED:",
        f"  Action:  {target.verb}({target.obj1} → {target.obj2})",
        f"  Type:    {target.remark}",
        f"  Window:  t={target.start/fps:.1f}s → t={target.end/fps:.1f}s",
        "",
    ]


def _preceding_context(
    actions: List[AssemblyAction],
    target: AssemblyAction,
    fps: float,
    n: int,
) -> List[str]:
    before = [a for a in actions if a.end <= target.start][-n:]
    lines = [f"PRECEDING CONTEXT (last {n} actions):"]
    for a in before:
        lines.append(f"  {a.describe(fps)}")
    lines.append("")
    return lines


def _cascade_context(
    actions: List[AssemblyAction],
    target: AssemblyAction,
    fps: float,
    n: int,
) -> List[str]:
    after = [a for a in actions if a.start >= target.end][:n]
    cascade = [a for a in after if a.is_mistake]
    if not after:
        return []
    lines = ["SUBSEQUENT ACTIONS:"]
    for a in after:
        lines.append(f"  {a.describe(fps)}")
    if cascade:
        lines.append(
            f"  ⚠  CASCADE: {len(cascade)} additional mistakes follow this error"
        )
    lines.append("")
    return lines


def _physical_state_block(state: PhysicalState) -> List[str]:
    lines = ["PHYSICAL STATE AT TIME OF ANOMALY:"]
    if state.attached:
        for obj, parent in state.attached.items():
            lines.append(f"  {obj}  →attached_to→  {parent}")
    else:
        lines.append("  (no components attached at this point)")
    return lines
