"""
pir/types.py
============
Shared data structures for the PIR pipeline.
All other modules import from here — never cross-import between layers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# ══════════════════════════════════════════════════════════
# VIDEO
# ══════════════════════════════════════════════════════════

@dataclass
class VideoMeta:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps

    def frame_to_time(self, frame: int) -> float:
        return frame / self.fps

    def time_to_frame(self, t: float) -> int:
        return int(t * self.fps)

    def __str__(self) -> str:
        return (
            f"{self.path.split('/')[-1]}  "
            f"{self.width}x{self.height} @ {self.fps:.0f}fps  "
            f"{self.duration_s:.1f}s ({self.total_frames} frames)"
        )


# ══════════════════════════════════════════════════════════
# ANNOTATIONS
# ══════════════════════════════════════════════════════════

@dataclass
class AssemblyAction:
    start: int        # frame
    end: int          # frame
    verb: str         # attach | detach
    obj1: str         # part being acted upon
    obj2: str         # target part
    status: str       # correct | mistake | correction
    remark: str = ""  # wrong position | wrong order | shouldn't have happened

    @property
    def is_mistake(self) -> bool:
        return self.status == "mistake"

    @property
    def duration_frames(self) -> int:
        return self.end - self.start

    def describe(self, fps: float = 60.0) -> str:
        s = f"t={self.start/fps:.1f}s: {self.verb}({self.obj1} → {self.obj2})"
        if self.is_mistake:
            s += f"  *** MISTAKE [{self.remark}]"
        elif self.status == "correction":
            s += "  [correction]"
        return s


# ══════════════════════════════════════════════════════════
# PERCEPTION
# ══════════════════════════════════════════════════════════

@dataclass
class Detection:
    label: str
    score: float
    bbox: List[float]                        # [x1, y1, x2, y2] pixels
    mask: Optional[np.ndarray] = None        # H×W binary uint8
    centroid: Optional[Tuple[int, int]] = None
    obj_id: int = 0

    @property
    def cx(self) -> int:
        return int((self.bbox[0] + self.bbox[2]) / 2)

    @property
    def cy(self) -> int:
        return int((self.bbox[1] + self.bbox[3]) / 2)

    @property
    def area_px(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class PerceptionResult:
    """Output of the PERCEIVE layer for one analysis window."""
    frames: List[np.ndarray]
    frame_indices: List[int]
    tracked_per_frame: List[List[Detection]]  # one list of detections per frame
    keyframe_paths: List[str]
    video_meta: VideoMeta
    target: AssemblyAction


# ══════════════════════════════════════════════════════════
# PHYSICAL GRAPH
# ══════════════════════════════════════════════════════════

@dataclass
class PhysicalState:
    """Snapshot of the assembly at one point in time."""
    frame: int
    attached: Dict[str, str]      # obj → parent
    action: Optional[str] = None
    anomalies: List[str] = field(default_factory=list)


@dataclass
class GraphContext:
    """Output of the INTERACT layer — input to Cosmos."""
    text: str                         # serialized temporal graph for the LLM
    target: AssemblyAction
    all_actions: List[AssemblyAction]
    physical_state: PhysicalState
    fps: float


# ══════════════════════════════════════════════════════════
# REASONING
# ══════════════════════════════════════════════════════════

@dataclass
class CosmosResponse:
    """Parsed output from Cosmos Reason2."""
    raw: str                   # full response including <think> block
    think: str                 # chain-of-thought reasoning
    answer: str                # final structured answer
    anomaly: str = ""
    physical_analysis: str = ""
    root_cause: str = ""
    severity: str = "UNKNOWN"
    corrective_action: str = ""
    confidence: int = 0
    latency_s: float = 0.0
    model: str = ""


# ══════════════════════════════════════════════════════════
# PIPELINE OUTPUT
# ══════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Final output of the full PIR pipeline."""
    perception: PerceptionResult
    graph: GraphContext
    reasoning: CosmosResponse
    output_video_path: str
    report_path: str
