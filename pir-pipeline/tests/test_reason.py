"""
tests/test_reason.py
Tests for pir.reason — response parsing (no API call needed).
Run: pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pir.reason import _extract_block, _parse_answer, parse_response


SAMPLE_RAW = """\
<think>
The arm connector must attach to the chassis before the boom can be connected.
By attaching it to the boom first, the assembly hierarchy is inverted.
This prevents the chassis connection from being made correctly.
</think>

<answer>
ANOMALY CONFIRMED: arm connector attached to boom instead of chassis

PHYSICAL ANALYSIS:
The arm connector is a structural mount that must anchor to the chassis before the boom extension can be attached. Attaching it directly to the boom inverts the load path — the boom becomes the base instead of the extension. This makes it geometrically impossible to later connect the arm connector to the chassis without full disassembly.

ROOT CAUSE:
The boom and chassis mounting points are visually similar when viewed from above. After a previous wrong_order correction, the operator re-attempted attachment but confused the two anchor points.

SEVERITY: HIGH
This error creates 4 cascade mistakes as the operator repeatedly attempts and fails to correct the assembly.

CORRECTIVE ACTION:
1. Detach arm connector from boom
2. Identify chassis mounting slot (flat rectangular socket)
3. Insert arm connector into chassis slot until click confirms engagement
4. Verify arm connector is flush with chassis surface
5. Only then attach boom to arm connector from above

CONFIDENCE: 92%
</answer>
"""


def test_extract_think_block():
    think = _extract_block(SAMPLE_RAW, "think")
    assert "arm connector must attach" in think
    assert len(think) > 50


def test_extract_answer_block():
    answer = _extract_block(SAMPLE_RAW, "answer")
    assert "ANOMALY CONFIRMED" in answer
    assert "CORRECTIVE ACTION" in answer


def test_parse_answer_anomaly():
    answer = _extract_block(SAMPLE_RAW, "answer")
    parsed = _parse_answer(answer)
    assert "arm connector" in parsed["anomaly"].lower()


def test_parse_answer_severity():
    answer = _extract_block(SAMPLE_RAW, "answer")
    parsed = _parse_answer(answer)
    assert parsed["severity"].strip().startswith("HIGH")


def test_parse_answer_confidence():
    answer = _extract_block(SAMPLE_RAW, "answer")
    parsed = _parse_answer(answer)
    assert parsed["confidence"] == 92


def test_parse_answer_corrective_action():
    answer = _extract_block(SAMPLE_RAW, "answer")
    parsed = _parse_answer(answer)
    assert "chassis" in parsed["corrective_action"].lower()


def test_parse_response_full():
    result = parse_response(SAMPLE_RAW, latency=1.23, model="nvidia/Cosmos-Reason2-2B")
    assert result.think != ""
    assert result.answer != ""
    assert result.confidence == 92
    assert result.severity.startswith("HIGH")
    assert result.latency_s == 1.23
    assert result.model == "nvidia/Cosmos-Reason2-2B"


def test_parse_response_missing_tags():
    # If model returns answer without tags, should degrade gracefully
    raw_no_tags = "ANOMALY CONFIRMED: test\nSEVERITY: LOW\nCONFIDENCE: 50%"
    result = parse_response(raw_no_tags, latency=0.5, model="test")
    assert result.raw == raw_no_tags
    # answer falls back to raw when no <answer> tag
    assert result.confidence == 50
