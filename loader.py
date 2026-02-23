"""
pir/loader.py
=============
LOAD layer — annotation parsing and video probing.
Pure functions, no side effects, no model loading.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Optional

import cv2

from pir.types import AssemblyAction, VideoMeta


# ══════════════════════════════════════════════════════════
# VIDEO
# ══════════════════════════════════════════════════════════

def probe_video(path: str, fps_override: Optional[float] = None) -> VideoMeta:
    """Read video metadata without loading frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    meta = VideoMeta(
        path=path,
        fps=fps_override or cap.get(cv2.CAP_PROP_FPS) or 60.0,
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return meta


# ══════════════════════════════════════════════════════════
# ANNOTATIONS
# ══════════════════════════════════════════════════════════

def load_annotations(csv_path: str) -> List[AssemblyAction]:
    """
    Parse Assembly101 mistake detection CSV.
    Format (no header): start, end, verb, obj1, obj2, status, [remark]
    """
    actions: List[AssemblyAction] = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            try:
                actions.append(AssemblyAction(
                    start=int(row[0].strip()),
                    end=int(row[1].strip()),
                    verb=row[2].strip(),
                    obj1=row[3].strip(),
                    obj2=row[4].strip(),
                    status=row[5].strip(),
                    remark=row[6].strip() if len(row) > 6 else "",
                ))
            except (ValueError, IndexError):
                continue
    return actions


def select_target(
    actions: List[AssemblyAction],
    mistake_type: str = "auto",
) -> AssemblyAction:
    """
    Select the most representative mistake to analyze.
    Priority for 'auto': wrong position > wrong order > longest mistake.
    """
    mistakes = [a for a in actions if a.is_mistake]
    if not mistakes:
        raise ValueError("No mistakes found in annotation CSV")

    if mistake_type == "auto":
        for priority in ["wrong position", "wrong order"]:
            for m in mistakes:
                if priority in m.remark.lower():
                    return m
        return max(mistakes, key=lambda m: m.duration_frames)

    for m in mistakes:
        if mistake_type.replace("_", " ") in m.remark.lower():
            return m

    print(f"  WARNING: mistake_type='{mistake_type}' not found — using longest mistake")
    return max(mistakes, key=lambda m: m.duration_frames)


def annotation_summary(actions: List[AssemblyAction]) -> dict:
    mistakes = [a for a in actions if a.is_mistake]
    return {
        "total_actions": len(actions),
        "total_mistakes": len(mistakes),
        "mistake_types": sorted(set(m.remark for m in mistakes if m.remark)),
        "components": sorted(set(a.obj1 for a in actions) | set(a.obj2 for a in actions)),
    }
