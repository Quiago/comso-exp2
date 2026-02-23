# PIR — Physical Inspection & Reasoning Pipeline

**PERCEIVE → INTERACT → REASON → ACT**

Autonomous assembly inspection using Grounding DINO + SAM2 + Cosmos Reason2.

Detects assembly mistakes from video, reasons about physical consequences using Cosmos chain-of-thought, and produces an annotated inspection video.

---

## Architecture

```
video + CSV
    │
    ▼
[LOAD]       pir/loader.py      — parse annotations, probe video
    │
    ▼
[PERCEIVE]   pir/perceive.py    — Grounding DINO + SAM2 tracking
    │                              keyframes extracted + annotated
    ▼
[INTERACT]   pir/interact.py    — Physical Temporal Graph builder
    │                              serialized context for Cosmos
    ▼
[REASON]     pir/reason.py      — Cosmos Reason2 via vLLM
    │                              <think> CoT + structured answer
    ▼
[ACT]        pir/act.py         — compile annotated MP4 + JSON report
```

---

## Setup (Lightning.ai, A100)

```bash
# Clone repo
git clone https://github.com/your-org/pir-pipeline.git
cd pir-pipeline

# One-time setup (installs deps, downloads Cosmos 16GB + SAM2)
bash scripts/setup.sh

# Start Cosmos vLLM server (run once per session)
bash scripts/start_server.sh

# Verify server is ready
curl http://localhost:8000/health
```

---

## Run

```bash
# Full pipeline (DINO + SAM2 + Cosmos)
python scripts/run_pipeline.py \
    --video /path/to/C10095_rgb.mp4 \
    --annot /path/to/sequence.csv

# Specific mistake type
python scripts/run_pipeline.py \
    --video ... --annot ... --mistake-type wrong_position

# DINO-only, no SAM2 (faster, less visual)
python scripts/run_pipeline.py \
    --video ... --annot ... --no-sam2

# Custom server (e.g. remote GPU)
python scripts/run_pipeline.py \
    --video ... --annot ... --cosmos-url http://my-server:8000

# Smaller model (faster, less VRAM)
bash scripts/start_server.sh --model nvidia/Cosmos-Reason2-2B
python scripts/run_pipeline.py --video ... --annot ... \
    --cosmos-model nvidia/Cosmos-Reason2-2B
```

---

## Output

```
output/<sequence_name>/
├── inspection_report.mp4    # annotated video with Cosmos panel
├── report.json              # full structured report
└── keyframes/               # annotated keyframes (DINO + SAM2)
    ├── kf_000_f016548.jpg
    ├── kf_001_f016698.jpg
    └── ...
```

The video includes:
- SAM2 segmentation masks (color-coded per object)
- DINO bounding boxes + labels
- Hand → component relation lines
- Status bar (red = inside mistake window)
- Cosmos report panel: ANOMALY / PHYSICAL ANALYSIS / ROOT CAUSE / SEVERITY / CORRECTIVE ACTION

---

## Tests

```bash
# Run all tests (no GPU, no API needed)
pytest tests/ -v

# Individual modules
pytest tests/test_loader.py -v
pytest tests/test_interact.py -v
pytest tests/test_reason.py -v    # tests parsing only, no API call
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to video (.mp4) |
| `--annot` | required | Path to annotation CSV |
| `--output` | `./output` | Output directory |
| `--mistake-type` | `auto` | `auto` / `wrong_position` / `wrong_order` / `shouldnt_have_happened` |
| `--padding` | `5.0` | Seconds of context around mistake |
| `--no-sam2` | false | Skip SAM2, use DINO-only |
| `--cosmos-url` | `http://localhost:8000` | vLLM server URL |
| `--cosmos-model` | `nvidia/Cosmos-Reason2-2B` | Model name |
| `--output-fps` | `15.0` | Output video FPS |
| `--n-keyframes` | `8` | Keyframes sent to Cosmos |

---

## Why Cosmos over Claude/GPT-4?

Cosmos Reason2 is trained on physical AI datasets with explicit chain-of-thought reasoning about the physical world. The `<think>` block shows step-by-step mechanical reasoning:

> *"The arm connector must anchor to the chassis before the boom can be attached. Attaching it to the boom first inverts the load path — the boom becomes the structural base instead of the extension, making it geometrically impossible to later connect to the chassis without full disassembly..."*

Claude and GPT-4 produce direct answers. Cosmos produces **auditable physical reasoning** — which is what industrial inspection systems require.

uv run python scripts/run_pipeline.py --video /teamspace/studios/this_studio/pir-pipeline/videos/C10095_rgb.mp4 --annot /teamspace/studios/this_studio/pir-pipeline/annot/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532.csv