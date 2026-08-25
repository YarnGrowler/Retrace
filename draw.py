from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from drawing.order import assign_stages_and_order
from drawing.preprocess import load_and_extract_ink, skeletonize
from drawing.render import render_animation
from drawing.trace import (
    clean_and_merge_paths,
    estimate_thickness,
    path_length,
    trace_skeleton,
)


OUTPUT_DIR = Path("output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn black-and-white line art into a local speed-drawing video."
    )
    parser.add_argument("image", type=Path, help="Input PNG or other OpenCV-readable image")
    parser.add_argument(
        "--duration",
        default="25",
        help="Target seconds, or 'auto' to give every stroke visible pen time",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: output/<input-name>.mp4)",
    )
    parser.add_argument(
        "--threshold", default="auto", help="'auto' (Otsu) or a grayscale value from 0 to 255"
    )
    parser.add_argument(
        "--cursor", choices=("none", "dot", "pencil"), default="pencil"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Print extra processing details")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live window (useful on headless machines)",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = OUTPUT_DIR / f"{args.image.stem}.mp4"
    if str(args.duration).lower() == "auto":
        args.duration = "auto"
    else:
        try:
            args.duration = float(args.duration)
        except ValueError:
            parser.error("--duration must be a positive number or 'auto'")
        if args.duration <= 0:
            parser.error("--duration must be greater than zero")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.threshold != "auto":
        try:
            value = int(args.threshold)
        except ValueError:
            parser.error("--threshold must be 'auto' or an integer from 0 to 255")
        if not 0 <= value <= 255:
            parser.error("--threshold must be from 0 to 255")
        args.threshold = value
    return args


def save_stroke_debug(
    path: Path, strokes: list[dict], width: int, height: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), 255, np.uint8)
    colors = rng.integers(30, 225, size=(max(1, len(strokes)), 3), dtype=np.uint8)
    for index, stroke in enumerate(strokes):
        points = np.asarray(stroke["points"], np.int32)
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, colors[index].tolist(), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def save_order_debug(path: Path, strokes: list[dict], width: int, height: int) -> None:
    canvas = np.full((height, width, 3), 255, np.uint8)
    count = max(1, len(strokes) - 1)
    label_every = max(20, len(strokes) // 24)
    for index, stroke in enumerate(strokes):
        hue = int(120 * (1.0 - index / count))
        hsv = np.uint8([[[hue, 210, 205]]])
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        points = np.asarray(stroke["points"], np.int32)
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        if index % label_every == 0 and len(points):
            x, y = (int(v) for v in points[0])
            cv2.circle(canvas, (x, y), 3, (25, 25, 25), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index),
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)



def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        print(f"ERROR: Input image not found: {args.image}", file=sys.stderr)
        return 2

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("DRAWING PROTOTYPE\n")
    print(f"Input: {args.image}")

    print("\n[1/7] Thresholding...")
    gray, ink = load_and_extract_ink(args.image, args.threshold)
    height, width = gray.shape
    ink_coverage = 100.0 * float(np.count_nonzero(ink)) / ink.size
    print(f"      resolution: {width} x {height}")
    print(f"      ink coverage: {ink_coverage:.1f}%")
    if ink_coverage > 30.0:
        print("      WARNING: Image contains large filled regions.")
        print("      This prototype works best with line art.")
    cv2.imwrite(str(output_dir / "01_binary.png"), ink)

    print("\n[2/7] Skeletonizing...")
    skeleton, hatch_mask, shaded_pixels = skeletonize(ink)
    print(f"      skeleton pixels: {np.count_nonzero(skeleton):,}")
    if shaded_pixels:
        print(f"      shaded pixels converted to hatching: {shaded_pixels:,}")
    cv2.imwrite(str(output_dir / "02_skeleton.png"), skeleton)

    print("\n[3/7] Tracing...")
    raw_paths = trace_skeleton(skeleton)
    print(f"      raw strokes: {len(raw_paths):,}")

    print("\n[4/7] Cleaning...")
    diagonal = float(np.hypot(width, height))
    paths, stats = clean_and_merge_paths(raw_paths, diagonal)
    distance_map = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    hatch_region_mask = cv2.dilate(
        hatch_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    _, hatch_region_labels = cv2.connectedComponents(hatch_region_mask, 8)
    strokes: list[dict] = []
    for points in paths:
        length = path_length(points)
        hatch_hits = sum(
            bool(hatch_mask[min(height - 1, max(0, y)), min(width - 1, max(0, x))])
            for x, y in points
        )
        is_hatch = bool(points) and hatch_hits / len(points) >= 0.5
        shade_group = 0
        if is_hatch:
            point_labels = [
                int(hatch_region_labels[min(height - 1, max(0, y)), min(width - 1, max(0, x))])
                for x, y in points
            ]
            positive_labels = [label for label in point_labels if label > 0]
            if positive_labels:
                shade_group = int(np.bincount(positive_labels).argmax())
        strokes.append(
            {
                "id": len(strokes),
                "points": [[int(x), int(y)] for x, y in points],
                "length": round(length, 3),
                "thickness": 1 if is_hatch else estimate_thickness(points, distance_map),
                "kind": "hatch" if is_hatch else "line",
                "shade_group": shade_group if is_hatch else None,
            }
        )
    print(f"      removed tiny strokes: {stats['removed']:,}")
    print(f"      merged continuations: {stats['merged']:,}")
    print(f"      final strokes: {len(strokes):,}")
    if not strokes:
        print("ERROR: No usable strokes were found in the image.", file=sys.stderr)
        return 1
    save_stroke_debug(output_dir / "03_strokes.png", strokes, width, height, args.seed)

    print("\n[5/7] Ordering...")
    ordered, stage_counts = assign_stages_and_order(strokes, width, height)
    for stage in ("structural", "medium", "detail"):
        print(f"      {stage}: {stage_counts[stage]:,}")
    save_order_debug(output_dir / "04_order.png", ordered, width, height)

    payload = {"width": width, "height": height, "strokes": ordered}
    with (output_dir / "strokes.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print("\n[6/7] Rendering...")
    total_length = sum(float(stroke["length"]) for stroke in ordered)
    automatic_duration = args.duration == "auto"
    if automatic_duration:
        # Ten percent is reserved for pen-up movement and the clean final hold;
        # every actual draw event receives at least one full output-frame interval.
        calculated = len(ordered) / (args.fps * 0.88) + 0.5
        target_duration = max(5.0, np.ceil(calculated * args.fps) / args.fps)
    else:
        target_duration = float(args.duration)
    print(f"      total path length: {total_length:,.0f} px")
    print(f"      target duration: {target_duration:g} sec" + (" (auto)" if automatic_duration else ""))
    available_frames = int(round(target_duration * args.fps))
    if not automatic_duration and available_frames < len(ordered):
        print(
            f"      WARNING: {len(ordered):,} strokes share only {available_frames:,} frames; "
            "use --duration auto to show every micro-stroke under the pencil."
        )
    result = render_animation(
        ordered,
        width,
        height,
        args.output,
        duration=target_duration,
        fps=args.fps,
        cursor=args.cursor,
        seed=args.seed,
        preview=not args.no_preview,
        appearance_gray=gray,
        ink_mask=ink,
        difference_path=output_dir / "05_final_difference.png",
        ensure_stroke_visibility=automatic_duration,
    )

    print("\n\n[7/7] Complete" if not result.cancelled else "\n\n[7/7] Cancelled")
    print(f"\n{args.output}")
    print(f"Foreground pixels reconstructed: {result.foreground_revealed_percent:.2f}%")
    print(f"Final visual reconstruction: {result.reconstruction_percent:.3f}%")
    if args.debug:
        print(f"Rendered frames: {result.frames:,}")
        print(f"Calculated drawing speed: {result.draw_speed:,.0f} px/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
