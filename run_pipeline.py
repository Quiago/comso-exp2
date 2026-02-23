"""
scripts/run_pipeline.py
========================
CLI entrypoint for the PIR pipeline.
This script is a thin orchestrator — all logic lives in pir/*.

Usage:
  python scripts/run_pipeline.py \
      --video /path/C10095_rgb.mp4 \
      --annot /path/sequence.csv

  # Analyze a specific mistake type
  python scripts/run_pipeline.py \
      --video ... --annot ... --mistake-type wrong_position

  # Skip SAM2 (DINO-only, faster)
  python scripts/run_pipeline.py \
      --video ... --annot ... --no-sam2

  # Custom vLLM endpoint (e.g. remote Lightning server)
  python scripts/run_pipeline.py \
      --video ... --annot ... --cosmos-url http://my-server:8000
"""

import argparse
import sys
from pathlib import Path

# Make pir package importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pir.loader  import probe_video, load_annotations, select_target, annotation_summary
from pir.perceive import perceive
from pir.interact import build_graph_context
from pir.reason  import call_cosmos, check_server
from pir.act     import act


def parse_args():
    p = argparse.ArgumentParser(
        description="PIR — Physical Inspection & Reasoning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video",   required=True,  help="Path to video (.mp4)")
    p.add_argument("--annot",   required=True,  help="Path to annotation CSV")
    p.add_argument("--output",  default="./output", help="Output directory")

    p.add_argument(
        "--mistake-type", default="auto",
        choices=["auto", "wrong_position", "wrong_order", "shouldnt_have_happened"],
        help="Which mistake type to analyze (default: auto — prefers wrong_position)",
    )
    p.add_argument(
        "--padding", type=float, default=5.0,
        help="Seconds of context around mistake window (default: 5.0)",
    )
    p.add_argument(
        "--no-sam2", action="store_true",
        help="Skip SAM2 tracking, use DINO detections only (faster)",
    )
    p.add_argument(
        "--cosmos-url", default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)",
    )
    p.add_argument(
        "--cosmos-model", default="nvidia/Cosmos-Reason2-8B",
        help="Model name as registered in vLLM (default: nvidia/Cosmos-Reason2-8B)",
    )
    p.add_argument(
        "--output-fps", type=float, default=15.0,
        help="Output video FPS (default: 15.0)",
    )
    p.add_argument(
        "--n-keyframes", type=int, default=8,
        help="Number of keyframes to extract for Cosmos (default: 8)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────
    for path, label in [(args.video, "Video"), (args.annot, "Annotation CSV")]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    if not check_server(args.cosmos_url):
        print(f"\nERROR: Cosmos vLLM server not reachable at {args.cosmos_url}")
        print("  Start the server first:")
        print("    bash scripts/start_server.sh")
        print("  Or check logs: tail -f /tmp/vllm_cosmos.log")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_name = Path(args.annot).stem
    out_dir = Path(args.output) / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"PIR PIPELINE — Physical Inspection & Reasoning")
    print(f"  Video:    {Path(args.video).name}")
    print(f"  Sequence: {seq_name}")
    print(f"  Device:   {device.upper()}")
    print(f"  Perceive: {'DINO only' if args.no_sam2 else 'DINO + SAM2'}")
    print(f"  Cosmos:   {args.cosmos_model}  @ {args.cosmos_url}")
    print(f"{'='*62}")

    # ── LOAD ──────────────────────────────────────────────
    print("\n[0/4] LOAD")
    video = probe_video(args.video)
    actions = load_annotations(args.annot)
    summary = annotation_summary(actions)
    print(f"  {video}")
    print(f"  {summary['total_actions']} actions | {summary['total_mistakes']} mistakes")
    print(f"  Types: {summary['mistake_types']}")
    print(f"  Components: {summary['components']}")

    target = select_target(actions, args.mistake_type)
    print(f"\n  TARGET → {target.describe(video.fps)}")

    # ── PERCEIVE ──────────────────────────────────────────
    print("\n[1/4] PERCEIVE")
    perception = perceive(
        video=video,
        target=target,
        out_dir=out_dir,
        device=device,
        padding_s=args.padding,
        use_sam2=not args.no_sam2,
        n_keyframes=args.n_keyframes,
    )

    # ── INTERACT ──────────────────────────────────────────
    print("\n[2/4] INTERACT")
    graph = build_graph_context(actions, target, video.fps)
    print(graph.text)

    # ── REASON ────────────────────────────────────────────
    print("\n[3/4] REASON")
    cosmos = call_cosmos(
        graph=graph,
        keyframe_paths=perception.keyframe_paths,
        base_url=args.cosmos_url,
        model=args.cosmos_model,
    )
    print("\n" + "─" * 54)
    print(cosmos.answer)
    print("─" * 54)

    # ── ACT ───────────────────────────────────────────────
    print("\n[4/4] ACT")
    video_path, report_path = act(
        perception=perception,
        graph=graph,
        cosmos=cosmos,
        out_dir=out_dir,
        output_fps=args.output_fps,
    )

    print(f"\n{'='*62}")
    print(f"✓  DONE")
    print(f"   Video:    {video_path}")
    print(f"   Report:   {report_path}")
    print(f"   Keyframes:{out_dir}/keyframes/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
