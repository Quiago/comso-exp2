"""
scripts/test_perceive.py
========================
Validación visual de DINO + SAM2 usando el código de pir/perceive.py.

Genera un video corto anotado para verificar que:
  1. DINO detecta los objetos en el frame correcto (mistake onset)
  2. SAM2 propaga las máscaras frame a frame (tracking real, no estático)

Uso:
  uv run python test/test_perceive.py \
      --video /path/C10095_rgb.mp4 \
      --annot  /path/sequence.csv \
      --output ./test_output

Indicadores de éxito en el video generado:
  ✅ DINO: bounding boxes en posiciones correctas en t=280.8s
  ✅ SAM2:  máscaras de color que SE MUEVEN con los objetos entre frames
  ❌ FALLO: bounding boxes congelados en la misma posición en todos los frames
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import torch

# Importa desde pir/ — reutiliza código existente sin modificarlo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pir.loader   import probe_video, load_annotations, select_target
from pir.perceive import (
    extract_frames,
    run_dino,
    run_sam2,
    dino_only_tracking,
    draw_detections,
)
from pir.types import VideoMeta, AssemblyAction


# ══════════════════════════════════════════════════════════
# REPORTE DE TEXTO
# ══════════════════════════════════════════════════════════

def print_tracking_report(tracked_per_frame: list, indices: list, fps: float) -> None:
    """
    Imprime estadísticas del tracking para validar que SAM2 funcionó.
    Si los bboxes son idénticos en todos los frames → SAM2 no funcionó (estático).
    Si los bboxes cambian → SAM2 está trackeando.
    """
    print("\n── Tracking Report ──────────────────────────────")

    frames_with_dets = [(i, dets) for i, dets in enumerate(tracked_per_frame) if dets]
    print(f"  Frames con detecciones: {len(frames_with_dets)}/{len(tracked_per_frame)}")

    if not frames_with_dets:
        print("  ❌ FALLO: ningún frame tiene detecciones")
        return

    # Verificar si los bboxes se mueven entre frames
    # Tomamos el primer objeto detectado y comparamos su bbox en N frames
    obj_bboxes: dict[str, list] = {}
    for frame_i, dets in frames_with_dets:
        for det in dets:
            key = det.label
            if key not in obj_bboxes:
                obj_bboxes[key] = []
            obj_bboxes[key].append((indices[frame_i] / fps, det.bbox))

    print(f"\n  Objetos trackeados: {list(obj_bboxes.keys())}")

    for label, entries in obj_bboxes.items():
        if len(entries) < 2:
            continue
        # Comparar primer y último bbox
        t0, bbox0 = entries[0]
        t1, bbox1 = entries[-1]
        # Desplazamiento del centroide
        cx0 = (bbox0[0] + bbox0[2]) / 2
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy0 = (bbox0[1] + bbox0[3]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        drift = ((cx1 - cx0)**2 + (cy1 - cy0)**2) ** 0.5

        status = "✅ MOVING" if drift > 5 else "❌ STATIC (possible SAM2 failure)"
        print(f"\n  [{label}]")
        print(f"    t={t0:.1f}s  bbox={[round(v) for v in bbox0]}")
        print(f"    t={t1:.1f}s  bbox={[round(v) for v in bbox1]}")
        print(f"    centroid drift: {drift:.1f}px  →  {status}")

    print("─────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════
# VIDEO DE VALIDACIÓN
# ══════════════════════════════════════════════════════════

def compile_test_video(
    frames: list,
    indices: list,
    tracked_per_frame: list,
    target: AssemblyAction,
    fps: float,
    out_path: Path,
    output_fps: float = 10.0,
) -> None:
    """
    Genera video anotado con bboxes, máscaras y timestamps.
    Añade indicador visual SAM2/DINO-only en cada frame.
    """
    if not frames:
        print("  ❌ Sin frames para compilar")
        return

    h, w = frames[0].shape[:2]
    # Panel inferior pequeño con info del modo
    panel_h = 40
    total_h = h + panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (w, total_h))

    # Detectar si hay masks (SAM2) o solo bboxes (DINO fallback)
    has_masks = any(
        det.mask is not None
        for frame_dets in tracked_per_frame
        for det in frame_dets
    )
    mode_label = "DINO + SAM2 ✅" if has_masks else "DINO only (SAM2 failed) ❌"
    mode_color = (80, 255, 130) if has_masks else (60, 80, 255)

    print(f"  Mode: {mode_label}")
    print(f"  Rendering {len(frames)} frames → {out_path.name}...")

    for i, (frame, fidx) in enumerate(zip(frames, indices)):
        dets = tracked_per_frame[i] if i < len(tracked_per_frame) else []
        annotated = draw_detections(frame, dets, fidx, fps, target)

        # Panel inferior: modo + frame count
        panel = __import__('numpy').zeros((panel_h, w, 3), dtype=__import__('numpy').uint8)
        panel[:] = (20, 20, 30)
        cv2.putText(panel, f"  {mode_label}   |   frame {i+1}/{len(frames)}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1, cv2.LINE_AA)

        combined = __import__('numpy').vstack([annotated, panel])
        writer.write(combined)

        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(frames)}")

    writer.release()
    print(f"  ✓ Video: {out_path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Validación visual de DINO + SAM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video",   required=True,  help="Path al video .mp4")
    p.add_argument("--annot",   required=True,  help="Path al CSV de anotaciones")
    p.add_argument("--output",  default="./test_output", help="Directorio de salida")
    p.add_argument("--no-sam2", action="store_true", help="Forzar DINO-only (para comparar)")
    p.add_argument("--padding", type=float, default=3.0,
                   help="Segundos de contexto alrededor del mistake (default: 3.0)")
    p.add_argument("--max-frames", type=int, default=300,
                   help="Máximo de frames a extraer (default: 300, más rápido para test)")
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in [(args.video, "Video"), (args.annot, "CSV")]:
        if not Path(path).exists():
            print(f"ERROR: {label} no encontrado: {path}")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*54}")
    print(f"  PERCEIVE VALIDATION TEST")
    print(f"  Device: {device.upper()}")
    print(f"  Mode:   {'DINO only (--no-sam2)' if args.no_sam2 else 'DINO + SAM2'}")
    print(f"{'='*54}\n")

    # LOAD
    print("[1/4] Loading video + annotations...")
    video   = probe_video(args.video)
    actions = load_annotations(args.annot)
    target  = select_target(actions, "auto")
    print(f"  Video:  {video}")
    print(f"  Target: {target.describe(video.fps)}")

    # EXTRACT FRAMES
    print("\n[2/4] Extracting frames...")
    padding   = video.time_to_frame(args.padding)
    start     = max(0, target.start - padding)
    end       = min(video.total_frames, target.end + padding)
    print(f"  Window: {start/video.fps:.1f}s → {end/video.fps:.1f}s")

    frames, indices = extract_frames(
        video.path, start, end, video.fps,
        max_frames=args.max_frames,
        use_cache=False,   # siempre fresco en test
    )
    print(f"  Extracted: {len(frames)} frames")

    # DINO en mistake onset
    print("\n[3/4] Running DINO + SAM2...")
    dino_frame_idx = 0
    for i, abs_idx in enumerate(indices):
        if abs_idx >= target.start:
            dino_frame_idx = i
            break

    print(f"  DINO frame: {indices[dino_frame_idx]} "
          f"(t={indices[dino_frame_idx]/video.fps:.1f}s)")
    init_dets = run_dino(frames[dino_frame_idx], target.obj1, target.obj2, device)

    if not init_dets:
        print("  ⚠  DINO no detectó objetos — prueba bajar el threshold en run_dino()")

    # SAM2 o fallback
    use_sam2 = not args.no_sam2
    if use_sam2 and init_dets:
        try:
            frames_for_sam  = frames[dino_frame_idx:]
            indices_for_sam = indices[dino_frame_idx:]
            tracked_sam = run_sam2(frames_for_sam, indices_for_sam, init_dets, device)
            tracked = [[] for _ in range(dino_frame_idx)] + tracked_sam
        except Exception as e:
            print(f"  SAM2 failed ({type(e).__name__}: {e})")
            print("  → Falling back to DINO-only (static bboxes)")
            tracked = dino_only_tracking(frames, init_dets)
    else:
        tracked = dino_only_tracking(frames, init_dets)

    # REPORTE
    print_tracking_report(tracked, indices, video.fps)

    # VIDEO
    print("[4/4] Compiling test video...")
    suffix = "dino_only" if args.no_sam2 else "dino_sam2"
    out_path = out_dir / f"test_perceive_{suffix}.mp4"
    compile_test_video(frames, indices, tracked, target, video.fps, out_path)

    print(f"\n{'='*54}")
    print(f"✓  TEST COMPLETE")
    print(f"   Video: {out_path}")
    print(f"   Abre el video y verifica:")
    print(f"   ✅ SAM2 OK:  máscaras de color que SE MUEVEN")
    print(f"   ❌ SAM2 KO:  bboxes congelados en misma posición")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()