"""
pir/reason.py
=============
REASON layer — Cosmos Reason2 physical reasoning via vLLM.

Calls the vLLM server (running Cosmos-Reason2-8B) via OpenAI-compatible
HTTP endpoint. Parses the <think> / <answer> structure.

Why vLLM + local server vs direct transformers:
  - Model loads once, inference is stateless HTTP — no GPU re-init per run
  - Same interface works for local (Lightning) and remote (NIM) deployments
  - Easy to swap model without changing pipeline code (just change --model flag)
"""

from __future__ import annotations
import base64
import re
import time
from pathlib import Path
from typing import List, Optional
import urllib.request

from pir.types import GraphContext, CosmosResponse


# ══════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════
# Instructs Cosmos to produce structured physical reasoning.
# Uses <think> / <answer> format per NVIDIA's Cosmos training protocol.

SYSTEM_PROMPT = """\
You are an autonomous Industrial Assembly Inspector Agent with deep physical common sense.

You receive:
1. A Physical Temporal Graph describing the assembly sequence and object states
2. Video keyframes from the mistake window (sampled at 4fps)
3. A detected anomaly signal

Your task: analyze the physical mistake autonomously — do not wait to be asked.

Answer in this exact format:

<think>
[Step-by-step physical reasoning:
- What is the correct assembly hierarchy for these components?
- What physical constraint does this action violate?
- What is the immediate mechanical consequence?
- Why does this cause cascade failures downstream?
- What is the root cause of the operator error?]
</think>

<answer>
ANOMALY CONFIRMED: [one-line physical description]

PHYSICAL ANALYSIS:
[2-3 sentences: constraint violated, mechanical consequence, cascade effect]

ROOT CAUSE:
[Why the operator made this mistake]

SEVERITY: CRITICAL / HIGH / MEDIUM / LOW
[One sentence justification]

CORRECTIVE ACTION:
1. [step]
2. [step]
3. [step]

CONFIDENCE: [0-100%]
</answer>\
"""

USER_PROMPT_TEMPLATE = """\
{graph_context}

The keyframes above are sampled at 4fps from the mistake window (t={t_start:.1f}s → t={t_end:.1f}s).

Analyze this assembly mistake autonomously:
1. What physical constraint is violated by {verb}({obj1} → {obj2})?
2. Why does this cause cascade failures in subsequent assembly steps?
3. What is the corrective sequence to restore the correct physical state?
"""


# ══════════════════════════════════════════════════════════
# SERVER CHECK
# ══════════════════════════════════════════════════════════

def check_server(base_url: str, timeout: int = 5) -> bool:
    try:
        urllib.request.urlopen(f"{base_url}/health", timeout=timeout)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════
# RESPONSE PARSING
# ══════════════════════════════════════════════════════════

def _extract_block(text: str, tag: str) -> str:
    """Extract content between <tag> and </tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_answer(answer: str) -> dict:
    """Extract structured fields from the <answer> block."""
    sections = {
        "ANOMALY CONFIRMED":  "anomaly",
        "PHYSICAL ANALYSIS":  "physical_analysis",
        "ROOT CAUSE":         "root_cause",
        "SEVERITY":           "severity",
        "CORRECTIVE ACTION":  "corrective_action",
        "CONFIDENCE":         "confidence",
    }
    parsed = {v: "" for v in sections.values()}
    parsed["confidence"] = 0

    current_key: Optional[str] = None
    current_lines: List[str] = []

    for line in answer.split("\n"):
        matched = False
        for header, key in sections.items():
            if line.strip().upper().startswith(header):
                if current_key:
                    parsed[sections[current_key]] = "\n".join(current_lines).strip()
                current_key = header
                current_lines = [line.split(":", 1)[-1].strip()] if ":" in line else []
                matched = True
                break
        if not matched and current_key:
            current_lines.append(line)

    if current_key:
        parsed[sections[current_key]] = "\n".join(current_lines).strip()

    # Parse confidence as int
    conf_text = parsed.get("confidence", "0")
    nums = re.findall(r"\d+", str(conf_text))
    parsed["confidence"] = int(nums[0]) if nums else 0

    return parsed


def parse_response(raw: str, latency: float, model: str) -> CosmosResponse:
    think = _extract_block(raw, "think")
    answer = _extract_block(raw, "answer") or raw  # fallback if tags absent
    fields = _parse_answer(answer)

    return CosmosResponse(
        raw=raw,
        think=think,
        answer=answer,
        latency_s=latency,
        model=model,
        **fields,
    )


# ══════════════════════════════════════════════════════════
# COSMOS CLIENT
# ══════════════════════════════════════════════════════════

def _build_user_text(graph: GraphContext) -> str:
    return (
        f"{graph.text}\n\n"
        f"The keyframes above show the assembly at 4fps around the mistake window "
        f"(t={graph.target.start / graph.fps:.1f}s → t={graph.target.end / graph.fps:.1f}s).\n\n"
        f"Analyze this mistake autonomously:\n"
        f"1. What physical constraint is violated by "
        f"{graph.target.verb}({graph.target.obj1} → {graph.target.obj2})?\n"
        f"2. Why does this cause cascade failures in subsequent assembly steps?\n"
        f"3. What is the corrective sequence to restore the correct physical state?"
    )

def call_cosmos(
    graph: GraphContext,
    keyframe_paths: List[str],
    base_url: str,
    model: str = "nvidia/Cosmos-Reason2-8B",
    max_tokens: int = 4096,
    temperature: float = 0.6,
    max_keyframes: int = 6,
) -> CosmosResponse:
    """
    Call Cosmos Reason2 via vLLM OpenAI-compatible endpoint.
    FPS=4 for keyframe sampling matches Cosmos training protocol.
    """
    import tempfile
    import cv2
    from openai import OpenAI
    import os

    client = OpenAI(base_url=f"{base_url}/v1", api_key="local")

    # Build a temp MP4 from keyframes so vLLM can read it as video
    # vLLM needs a file path, not base64 — this is what --allowed-local-media-path enables
    video_path = None
    if keyframe_paths:
        valid_paths = [p for p in keyframe_paths if Path(p).exists()]
        if valid_paths:
            sample = cv2.imread(valid_paths[0])
            if sample is not None:
                h, w = sample.shape[:2]
                fd, video_path = tempfile.mkstemp(suffix=".mp4", dir=os.getcwd())
                os.close(fd)
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    4.0,           # 4 FPS — matches Cosmos training protocol
                    (w, h),
                )
                for p in valid_paths[:32]:
                    frame = cv2.imread(p)
                    if frame is not None:
                        writer.write(frame)
                writer.release()

    # Content: video file path FIRST, then text (NVIDIA media-first convention)
    user_content = []
    if video_path and Path(video_path).exists():
        user_content.append({
            "type": "video_url",
            "video_url": {
                "url": f"file://{Path(video_path).absolute()}",
                # FPS can also be overridden here per-request
                "fps": 4,
            },
        })
        print(f"  Running Cosmos via vLLM (video: {Path(video_path).name})...")
    else:
        print("  WARNING: no valid keyframes for vLLM video input")

    user_content.append({"type": "text", "text": _build_user_text(graph)})

    t0 = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"skip_special_tokens": False},  # preserve <think> tokens
    )

    latency = time.time() - t0

    # Cleanup temp video
    if video_path:
        try:
            Path(video_path).unlink()
        except OSError:
            pass

    print(response)
    raw = response.choices[0].message.content
    print(f"  ✓ Cosmos inference: {latency:.1f}s")

    result = parse_response(raw, latency, model)

    # Print CoT summary (not the full text — can be very long)
    if result.think:
        preview = result.think[:400].replace("\n", " ")
        print(f"\n  [Cosmos CoT — {len(result.think)} chars]: {preview}...")

    return result

