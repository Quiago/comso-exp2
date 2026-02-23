"""
pir/perceive.py
===============
PERCEIVE layer — Grounding DINO + SAM2 video tracking.

Responsibilities:
  1. Extract frames from the analysis window
  2. Run Grounding DINO on the init frame to detect objects
  3. Run SAM2 to propagate tracks across all frames
  4. Save annotated keyframes for Cosmos input
  5. Return PerceptionResult

Design:
  - DINO prompts are derived from CSV object names — never hardcoded
  - SAM2 is initialized from DINO boxes — no manual prompts needed
  - Falls back to DINO-only if SAM2 is unavailable
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pir.types import (
    AssemblyAction, VideoMeta, Detection, PerceptionResult
)


# ══════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ══════════════════════════════════════════════════════════

def extract_frames(
    video: VideoMeta,
    start_frame: int,
    end_frame: int,
    max_frames: int = 600,
) -> tuple[list, list]:
    """Extract frames from a video window. Returns (frames, frame_indices)."""
    cap = cv2.VideoCapture(video.path)
    end_frame = min(end_frame, video.total_frames)
    step = max(1, (end_frame - start_frame) // max_frames)

    frames, indices = [], []
    for i in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        indices.append(i)
    cap.release()

    print(f"  Extracted {len(frames)} frames  "
          f"({start_frame/video.fps:.1f}s → {end_frame/video.fps:.1f}s,  step={step})")
    return frames, indices


# ══════════════════════════════════════════════════════════
# GROUNDING DINO
# ══════════════════════════════════════════════════════════

def build_dino_prompt(obj1: str, obj2: str) -> str:
    """
    Build Grounding DINO prompt from CSV object names.
    Uses generic descriptors — no domain-specific hardcoding.
    """
    objects = {obj1, obj2}
    parts = ["human hand", "left hand", "right hand"]
    for obj in objects:
        parts.append(obj)
        parts.append(f"toy {obj}")
    parts.append("assembly component")
    return " . ".join(parts) + " ."


def run_dino(
    frame: np.ndarray,
    obj1: str,
    obj2: str,
    device: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> List[Detection]:
    """Run Grounding DINO on a single frame. Returns list of detections."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image
    import torch

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    prompt = build_dino_prompt(obj1, obj2)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    detections: List[Detection] = []
    seen: dict[str, float] = {}
    obj_counter = 1

    for label, score, box in zip(
        results["labels"], results["scores"], results["boxes"].cpu()
    ):
        label, score = str(label), float(score)
        if label not in seen or score > seen[label]:
            seen[label] = score
            detections.append(Detection(
                label=label,
                score=score,
                bbox=box.tolist(),
                obj_id=obj_counter,
            ))
            obj_counter += 1

    print(f"  DINO [{len(detections)} objects]: "
          + ", ".join(f"{d.label}({d.score:.2f})" for d in detections))
    return detections


# ══════════════════════════════════════════════════════════
# SAM2 TRACKING
# ══════════════════════════════════════════════════════════

def run_sam2(
    frames: list,
    indices: list,
    init_detections: List[Detection],
    device: str,
) -> List[List[Detection]]:
    """
    SAM2 video tracking initialized from DINO boxes.
    Returns one list of detections per frame.
    """
    from sam2.build_sam import build_sam2_video_predictor_hf

    print(f"  Loading SAM2 (hiera-small) on {device}...")
    predictor = build_sam2_video_predictor_hf(
        "facebook/sam2-hiera-small", device=device
    )

    id_to_label = {d.obj_id: d.label for d in init_detections}

    with tempfile.TemporaryDirectory() as tmp:
        # Write frames as JPEG files (SAM2 video predictor reads from disk)
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), frame)

        state = predictor.init_state(video_path=tmp)
        predictor.reset_state(state)

        for det in init_detections:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=det.obj_id,
                box=np.array(det.bbox, dtype=np.float32),
            )

        tracked: List[List[Detection]] = [[] for _ in frames]

        for out_fidx, out_obj_ids, out_logits in predictor.propagate_in_video(state):
            for oid, logit in zip(out_obj_ids, out_logits):
                mask = (logit[0].cpu().numpy() > 0).astype(np.uint8)
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                tracked[out_fidx].append(Detection(
                    label=id_to_label.get(int(oid), f"obj_{oid}"),
                    score=1.0,
                    bbox=[x1, y1, x2, y2],
                    mask=mask,
                    centroid=((x1 + x2) // 2, (y1 + y2) // 2),
                    obj_id=int(oid),
                ))

    found = sum(1 for f in tracked if f)
    print(f"  SAM2: tracked in {found}/{len(frames)} frames")
    return tracked


def dino_only_tracking(
    frames: list,
    init_detections: List[Detection],
) -> List[List[Detection]]:
    """
    Fallback when SAM2 is unavailable.
    Replicates init detections across all frames (no motion tracking).
    """
    print("  SAM2 unavailable — using static DINO detections across all frames")
    return [list(init_detections) for _ in frames]


# ══════════════════════════════════════════════════════════
# KEYFRAME SAVING
# ══════════════════════════════════════════════════════════

# Object color palette — deterministic by obj_id
_COLORS = [
    (255, 80,  80),   # red
    (80, 200, 255),   # cyan
    (80, 255, 130),   # green
    (255, 180,  50),  # orange
    (200,  80, 255),  # purple
    (255, 255,  80),  # yellow
    (80, 130, 255),   # blue
    (255, 130, 200),  # pink
]

def _color(obj_id: int) -> tuple:
    return _COLORS[(obj_id - 1) % len(_COLORS)]


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    frame_idx: int,
    fps: float,
    target: AssemblyAction,
) -> np.ndarray:
    """Draw masks, bboxes, labels, and status bar on a frame."""
    out = frame.copy()
    in_mistake = target.start <= frame_idx <= target.end

    # Semi-transparent masks
    overlay = out.copy()
    for det in detections:
        if det.mask is not None:
            color_layer = np.zeros_like(out)
            color_layer[det.mask > 0] = _color(det.obj_id)
            cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0, overlay)
    out = overlay

    # Hand → component relation lines
    hands = [d for d in detections if "hand" in d.label.lower() and d.centroid]
    comps = [d for d in detections if "hand" not in d.label.lower() and d.centroid]
    for h in hands:
        for c in comps:
            cv2.line(out, h.centroid, c.centroid, (255, 255, 255), 2, cv2.LINE_AA)

    # Bounding boxes + labels
    for det in detections:
        color = _color(det.obj_id)
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        thickness = 3 if in_mistake else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.label} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Status bar
    bar_color = (30, 0, 0) if in_mistake else (0, 20, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1], 50), bar_color, -1)
    status = f"*** MISTAKE: {target.remark} ***" if in_mistake else "monitoring"
    status_color = (60, 80, 255) if in_mistake else (0, 200, 80)
    cv2.putText(out, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"t={frame_idx/fps:.1f}s  f={frame_idx}",
                (out.shape[1] - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return out


def save_keyframes(
    frames: list,
    indices: list,
    tracked_per_frame: List[List[Detection]],
    target: AssemblyAction,
    fps: float,
    out_dir: Path,
    n: int = 8,
) -> List[str]:
    """Save N annotated keyframes. Returns list of saved paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, len(frames))
    positions = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]

    saved = []
    for pos in positions:
        dets = tracked_per_frame[pos] if pos < len(tracked_per_frame) else []
        annotated = draw_detections(frames[pos], dets, indices[pos], fps, target)
        path = out_dir / f"kf_{pos:03d}_f{indices[pos]:06d}.jpg"
        cv2.imwrite(str(path), annotated)
        saved.append(str(path))

    print(f"  {len(saved)} keyframes → {out_dir}/")
    return saved


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════

def perceive(
    video: VideoMeta,
    target: AssemblyAction,
    out_dir: Path,
    device: str,
    padding_s: float = 5.0,
    max_frames: int = 600,
    use_sam2: bool = True,
    n_keyframes: int = 8,
) -> PerceptionResult:
    """
    Run the full PERCEIVE layer.
    Returns PerceptionResult with frames, tracks, and keyframe paths.
    """
    padding = video.time_to_frame(padding_s)
    start = max(0, target.start - padding)
    end = min(video.total_frames, target.end + padding)

    print(f"  Window: {start/video.fps:.1f}s → {end/video.fps:.1f}s")

    frames, indices = extract_frames(video, start, end, max_frames)

    print("  Running Grounding DINO on init frame...")
    init_dets = run_dino(frames[0], target.obj1, target.obj2, device)

    if use_sam2 and init_dets:
        try:
            print("  Running SAM2 video tracking...")
            tracked = run_sam2(frames, indices, init_dets, device)
        except ImportError:
            tracked = dino_only_tracking(frames, init_dets)
    else:
        tracked = dino_only_tracking(frames, init_dets)

    kf_paths = save_keyframes(
        frames, indices, tracked, target, video.fps,
        out_dir / "keyframes", n_keyframes
    )

    return PerceptionResult(
        frames=frames,
        frame_indices=indices,
        tracked_per_frame=tracked,
        keyframe_paths=kf_paths,
        video_meta=video,
        target=target,
    )
