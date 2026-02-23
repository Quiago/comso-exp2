"""
pir/act.py
==========
ACT layer — compile annotated video and save inspection report.

Responsibilities:
  - Render each frame with SAM2 masks, DINO bboxes, relation lines, status bar
  - Attach Cosmos report panel below the video
  - Save JSON report
  - Write final MP4
"""

from __future__ import annotations
import json
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import List

import cv2
import numpy as np

from pir.types import (
    Detection, AssemblyAction, CosmosResponse,
    PerceptionResult, GraphContext, PipelineResult,
)
from pir.perceive import draw_detections, _color


# ══════════════════════════════════════════════════════════
# COSMOS PANEL
# ══════════════════════════════════════════════════════════

_SECTION_COLORS = {
    "ANOMALY":    (80, 80, 255),
    "PHYSICAL":   (130, 180, 255),
    "ROOT":       (255, 180, 50),
    "SEVERITY":   (60, 200, 255),
    "CORRECTIVE": (80, 255, 130),
    "CONFIDENCE": (200, 200, 200),
}

def _section_color(line: str) -> tuple:
    upper = line.strip().upper()
    for key, color in _SECTION_COLORS.items():
        if upper.startswith(key):
            return color
    return (210, 210, 210)


def render_cosmos_panel(
    width: int,
    cosmos: CosmosResponse,
    panel_h: int = 320,
) -> np.ndarray:
    """Render Cosmos report as a dark panel to stack below the video."""
    panel = np.full((panel_h, width, 3), (18, 18, 28), dtype=np.uint8)

    # Title bar
    cv2.rectangle(panel, (0, 0), (width, 40), (30, 30, 50), -1)
    cv2.putText(
        panel,
        f"COSMOS REASON2 — Physical Inspection Report  "
        f"[{cosmos.model.split('/')[-1]}  {cosmos.latency_s:.1f}s  {cosmos.confidence}% confidence]",
        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA,
    )
    cv2.line(panel, (12, 42), (width - 12, 42), (50, 50, 70), 1)

    # Word-wrap answer text
    lines: List[str] = []
    for raw_line in cosmos.answer.split("\n"):
        if len(raw_line) > 95:
            lines.extend(textwrap.wrap(raw_line, width=95))
        else:
            lines.append(raw_line)

    y = 60
    line_h = 18
    max_lines = (panel_h - 70) // line_h

    for line in lines[:max_lines]:
        color = _section_color(line)
        cv2.putText(panel, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        y += line_h

    return panel


# ══════════════════════════════════════════════════════════
# VIDEO COMPILATION
# ══════════════════════════════════════════════════════════

def compile_video(
    perception: PerceptionResult,
    cosmos: CosmosResponse,
    out_path: Path,
    output_fps: float = 15.0,
    panel_h: int = 320,
) -> Path:
    """
    Compile all annotated frames into a single MP4.
    Each frame = annotated video frame + Cosmos report panel below.
    """
    frames   = perception.frames
    indices  = perception.frame_indices
    tracked  = perception.tracked_per_frame
    target   = perception.target
    fps      = perception.video_meta.fps

    if not frames:
        raise ValueError("No frames to compile")

    h, w = frames[0].shape[:2]
    total_h = h + panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (w, total_h))

    cosmos_panel = render_cosmos_panel(w, cosmos, panel_h)

    print(f"  Rendering {len(frames)} frames → {out_path.name}...")
    for i, (frame, fidx) in enumerate(zip(frames, indices)):
        dets = tracked[i] if i < len(tracked) else []
        annotated = draw_detections(frame, dets, fidx, fps, target)
        combined = np.vstack([annotated, cosmos_panel])
        writer.write(combined)
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(frames)}")

    writer.release()
    print(f"  ✓ Video: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════

def save_report(
    out_dir: Path,
    perception: PerceptionResult,
    graph: GraphContext,
    cosmos: CosmosResponse,
    video_path: Path,
) -> Path:
    """Save JSON inspection report."""
    report = {
        "video": str(perception.video_meta),
        "target_mistake": asdict(perception.target),
        "physical_context": graph.text,
        "cosmos": {
            "model": cosmos.model,
            "latency_s": cosmos.latency_s,
            "confidence": cosmos.confidence,
            "severity": cosmos.severity,
            "think_length_chars": len(cosmos.think),
            "anomaly": cosmos.anomaly,
            "physical_analysis": cosmos.physical_analysis,
            "root_cause": cosmos.root_cause,
            "corrective_action": cosmos.corrective_action,
            "full_response": cosmos.raw,
        },
        "keyframes": perception.keyframe_paths,
        "output_video": str(video_path),
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  ✓ Report: {path}")
    return path


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════

def act(
    perception: PerceptionResult,
    graph: GraphContext,
    cosmos: CosmosResponse,
    out_dir: Path,
    output_fps: float = 15.0,
) -> tuple[Path, Path]:
    """
    Run the ACT layer.
    Returns (video_path, report_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "inspection_report.mp4"

    compile_video(perception, cosmos, video_path, output_fps)
    report_path = save_report(out_dir, perception, graph, cosmos, video_path)

    return video_path, report_path
