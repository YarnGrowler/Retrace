from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass
class RenderResult:
    frames: int
    draw_speed: float
    cancelled: bool
    reconstruction_percent: float
    foreground_revealed_percent: float


@dataclass
class RevealOwnership:
    pixels_by_stroke: list[np.ndarray]
    progress_by_stroke: list[np.ndarray]
    residual_pixels: np.ndarray
    foreground_pixels: np.ndarray


def _path_metrics(points: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    return array, np.concatenate(([0.0], np.cumsum(lengths)))


def _point_at(array: np.ndarray, cumulative: np.ndarray, distance: float) -> tuple[np.ndarray, np.ndarray]:
    if len(array) < 2 or distance <= 0:
        tangent = array[min(1, len(array) - 1)] - array[0]
        return array[0], tangent
    if distance >= cumulative[-1]:
        return array[-1], array[-1] - array[-2]
    segment = min(len(array) - 2, int(np.searchsorted(cumulative, distance, side="right") - 1))
    span = cumulative[segment + 1] - cumulative[segment]
    fraction = 0.0 if span == 0 else (distance - cumulative[segment]) / span
    point = array[segment] + fraction * (array[segment + 1] - array[segment])
    return point, array[segment + 1] - array[segment]


def _build_reveal_ownership(
    strokes: list[dict],
    ink_mask: np.ndarray,
    appearance_gray: np.ndarray,
    render_width: int,
) -> RevealOwnership:
    height, width = ink_mask.shape
    path_owner = np.zeros((height, width), np.int32)
    path_progress = np.zeros((height, width), np.float32)

    # Original city renderer: rasterize the thin traced paths only to establish
    # where and when the pen travels. The source grayscale pixels remain the
    # visible artwork, preserving their real thickness and texture.
    for stroke_index, stroke in enumerate(strokes):
        array, cumulative = _path_metrics(stroke["points"])
        for segment in range(len(array) - 1):
            start, finish = array[segment], array[segment + 1]
            steps = max(1, int(np.max(np.abs(finish - start))))
            fractions = np.linspace(0.0, 1.0, steps + 1)
            xs = np.rint(start[0] + fractions * (finish[0] - start[0])).astype(np.int32)
            ys = np.rint(start[1] + fractions * (finish[1] - start[1])).astype(np.int32)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            xs, ys, fractions = xs[valid], ys[valid], fractions[valid]
            path_owner[ys, xs] = stroke_index + 1
            path_progress[ys, xs] = (
                cumulative[segment]
                + fractions * (cumulative[segment + 1] - cumulative[segment])
            )

    guide_pixels = path_owner > 0
    if not np.any(guide_pixels):
        empty = [np.empty(0, np.int64) for _ in strokes]
        empty_progress = [np.empty(0, np.float32) for _ in strokes]
        return RevealOwnership(empty, empty_progress, np.empty(0, np.int64), np.empty(0, np.int64))

    distance_source = np.where(guide_pixels, 0, 255).astype(np.uint8)
    guide_distance, nearest_labels = cv2.distanceTransformWithLabels(
        distance_source,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    guide_labels = nearest_labels[guide_pixels]
    label_count = int(nearest_labels.max()) + 1
    label_to_owner = np.zeros(label_count, np.int32)
    label_to_progress = np.zeros(label_count, np.float32)
    label_to_owner[guide_labels] = path_owner[guide_pixels]
    label_to_progress[guide_labels] = path_progress[guide_pixels]
    nearest_owner = label_to_owner[nearest_labels]
    nearest_progress = label_to_progress[nearest_labels]

    foreground = (ink_mask > 0) | (appearance_gray < 254)
    assigned = foreground & (nearest_owner > 0)

    ys, xs = np.nonzero(assigned)
    padded_indices = ys.astype(np.int64) * render_width + xs
    owners = nearest_owner[ys, xs] - 1
    progresses = nearest_progress[ys, xs]
    sort_order = np.lexsort((progresses, owners))
    padded_indices = padded_indices[sort_order]
    owners = owners[sort_order]
    progresses = progresses[sort_order]
    counts = np.bincount(owners, minlength=len(strokes))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    pixels_by_stroke = [
        padded_indices[offsets[index] : offsets[index + 1]] for index in range(len(strokes))
    ]
    progress_by_stroke = [
        progresses[offsets[index] : offsets[index + 1]] for index in range(len(strokes))
    ]

    residual_y, residual_x = np.nonzero(foreground & ~assigned)
    residual_pixels = residual_y.astype(np.int64) * render_width + residual_x
    if len(residual_pixels):
        residual_order = np.argsort(guide_distance[residual_y, residual_x], kind="stable")
        residual_pixels = residual_pixels[residual_order]
    foreground_y, foreground_x = np.nonzero(foreground)
    foreground_pixels = foreground_y.astype(np.int64) * render_width + foreground_x
    return RevealOwnership(
        pixels_by_stroke,
        progress_by_stroke,
        residual_pixels,
        foreground_pixels,
    )


def _draw_cursor(canvas: np.ndarray, point: np.ndarray, pose_axis: np.ndarray, cursor: str) -> None:
    tip = np.asarray(point, np.float64)
    if cursor == "none":
        return
    if cursor == "dot":
        cv2.circle(canvas, tuple(np.rint(tip).astype(int)), 4, (35, 35, 35), -1, cv2.LINE_AA)
        return

    norm = float(np.linalg.norm(pose_axis))
    axis = pose_axis / norm if norm else np.asarray((0.72, -0.69))
    side = np.asarray((-axis[1], axis[0]))
    scale = float(np.clip(np.hypot(canvas.shape[1], canvas.shape[0]) / 1800.0, 0.65, 1.15))

    def at(along: float, across: float = 0.0) -> np.ndarray:
        return tip + axis * (along * scale) + side * (across * scale)

    def polygon(points: list[np.ndarray]) -> np.ndarray:
        return np.rint(np.asarray(points)).astype(np.int32)

    # A soft offset shadow gives the pencil depth above the paper.
    shadow = canvas.copy()
    shadow_offset = np.asarray((5.0, 7.0)) * scale
    shadow_body = polygon(
        [at(11, 5.5) + shadow_offset, at(84, 7) + shadow_offset,
         at(84, -7) + shadow_offset, at(11, -5.5) + shadow_offset]
    )
    cv2.fillConvexPoly(shadow, shadow_body, (80, 80, 80), cv2.LINE_AA)
    cv2.addWeighted(shadow, 0.16, canvas, 0.84, 0, canvas)

    # Foreshortened hexagonal pencil with three differently lit lacquer faces.
    body_upper = polygon([at(12, 5.5), at(66, 7), at(66, 1.5), at(12, 1.2)])
    body_center = polygon([at(12, 1.2), at(66, 1.5), at(66, -2.2), at(12, -1.8)])
    body_lower = polygon([at(12, -1.8), at(66, -2.2), at(66, -7), at(12, -5.5)])
    cv2.fillConvexPoly(canvas, body_upper, (72, 196, 247), cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, body_center, (98, 221, 255), cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, body_lower, (34, 145, 214), cv2.LINE_AA)
    body_outline = polygon([at(12, 5.5), at(66, 7), at(66, -7), at(12, -5.5)])
    cv2.polylines(canvas, [body_outline], True, (52, 78, 92), max(1, round(scale)), cv2.LINE_AA)
    cv2.line(
        canvas,
        tuple(np.rint(at(14, 2.3)).astype(int)),
        tuple(np.rint(at(63, 3.0)).astype(int)),
        (185, 242, 255),
        max(1, round(scale)),
        cv2.LINE_AA,
    )

    # Exposed wood and graphite; the graphite point lands exactly on the path.
    wood = polygon([tip, at(12, 5.5), at(12, -5.5)])
    cv2.fillConvexPoly(canvas, wood, (174, 204, 226), cv2.LINE_AA)
    cv2.polylines(canvas, [wood], True, (69, 79, 84), max(1, round(scale)), cv2.LINE_AA)
    graphite = polygon([tip, at(4.5, 1.55), at(4.5, -1.55)])
    cv2.fillConvexPoly(canvas, graphite, (27, 29, 31), cv2.LINE_AA)

    # Metallic ferrule and eraser remain visible beyond the grip.
    ferrule = polygon([at(66, 7), at(76, 7.2), at(76, -7.2), at(66, -7)])
    cv2.fillConvexPoly(canvas, ferrule, (180, 188, 193), cv2.LINE_AA)
    for along in (68.5, 72.0, 75.0):
        cv2.line(
            canvas,
            tuple(np.rint(at(along, 6.7)).astype(int)),
            tuple(np.rint(at(along, -6.7)).astype(int)),
            (105, 112, 118),
            max(1, round(scale)),
            cv2.LINE_AA,
        )
    eraser = polygon([at(76, 7.2), at(84, 6.3), at(84, -6.3), at(76, -7.2)])
    cv2.fillConvexPoly(canvas, eraser, (137, 157, 225), cv2.LINE_AA)
    cv2.polylines(canvas, [eraser], True, (77, 91, 126), max(1, round(scale)), cv2.LINE_AA)



def _make_schedule(
    strokes: list[dict],
    duration: float,
    fps: int,
    seed: int,
    ensure_stroke_visibility: bool = False,
) -> tuple[list[dict], float]:
    rng = np.random.default_rng(seed)
    final_hold = min(0.5, max(1.0 / fps, duration * 0.04))
    working_duration = max(duration - final_hold, duration * 0.5)
    overhead: list[tuple[float, float, float]] = []
    draw_weights: list[float] = []
    previous: np.ndarray | None = None
    total_distance = 0.0
    for stroke in strokes:
        points = np.asarray(stroke["points"], np.float64)
        length = float(stroke["length"])
        total_distance += length
        stage_factor = {"structural": 0.92, "medium": 1.0, "detail": 1.12}[stroke["stage"]]
        speed_factor = stage_factor * float(rng.uniform(0.9, 1.1))
        draw_weights.append(length / speed_factor)
        begin_pause = float(rng.uniform(0.02, 0.08))
        end_pause = float(rng.uniform(0.02, 0.10))
        if previous is None:
            move_time = 0.0
        else:
            travel = float(np.linalg.norm(points[0] - previous))
            move_frames = int(np.clip(2 + travel / 250.0, 2, 5))
            move_time = move_frames / fps
        overhead.append((move_time, begin_pause, end_pause))
        previous = points[-1]

    desired_overhead = sum(sum(values) for values in overhead)
    overhead_fraction = 0.10 if ensure_stroke_visibility else 0.24
    overhead_budget = min(desired_overhead, working_duration * overhead_fraction)
    overhead_scale = overhead_budget / desired_overhead if desired_overhead else 0.0
    draw_budget = max(working_duration - overhead_budget, working_duration * 0.5)
    weight_total = sum(draw_weights) or 1.0

    schedule: list[dict] = []
    time = 0.0
    previous = None
    minimum_draw_time = 1.0 / fps if ensure_stroke_visibility else 0.0
    minimum_draw_total = minimum_draw_time * len(strokes)
    weighted_draw_budget = max(0.0, draw_budget - minimum_draw_total)
    for index, stroke in enumerate(strokes):
        array, cumulative = _path_metrics(stroke["points"])
        move_time, begin_pause, end_pause = (value * overhead_scale for value in overhead[index])
        if previous is not None and move_time > 0:
            schedule.append({"kind": "move", "start": time, "end": time + move_time, "from": previous, "to": array[0]})
            time += move_time
        if begin_pause > 0:
            schedule.append({"kind": "pause", "start": time, "end": time + begin_pause, "point": array[0]})
            time += begin_pause
        draw_time = minimum_draw_time + weighted_draw_budget * draw_weights[index] / weight_total
        schedule.append(
            {
                "kind": "draw",
                "start": time,
                "end": time + draw_time,
                "array": array,
                "cumulative": cumulative,
                "stroke_index": index,
                "thickness": int(stroke["thickness"]),
                "color": (55, 55, 55) if stroke.get("kind") == "hatch" else (25, 25, 25),
            }
        )
        time += draw_time
        if end_pause > 0:
            schedule.append({"kind": "pause", "start": time, "end": time + end_pause, "point": array[-1]})
            time += end_pause
        previous = array[-1]

    draw_speed = total_distance / draw_budget if draw_budget else 0.0
    return schedule, draw_speed


def render_animation(
    strokes: list[dict],
    width: int,
    height: int,
    output_path: Path,
    duration: float = 25.0,
    fps: int = 30,
    cursor: str = "pencil",
    seed: int = 42,
    preview: bool = True,
    appearance_gray: np.ndarray | None = None,
    ink_mask: np.ndarray | None = None,
    difference_path: Path | None = None,
    ensure_stroke_visibility: bool = False,
) -> RenderResult:
    render_width = width + width % 2
    render_height = height + height % 2
    if appearance_gray is None:
        appearance_gray = np.full((height, width), 255, np.uint8)
    if ink_mask is None:
        ink_mask = np.zeros((height, width), np.uint8)
    appearance = np.full((render_height, render_width, 3), 255, np.uint8)
    appearance[:height, :width] = cv2.cvtColor(appearance_gray, cv2.COLOR_GRAY2BGR)
    appearance_flat = appearance.reshape((-1, 3))
    ownership = _build_reveal_ownership(strokes, ink_mask, appearance_gray, render_width)
    permanent_revealed = np.zeros(render_height * render_width, dtype=bool)
    schedule, draw_speed = _make_schedule(
        strokes, duration, fps, seed, ensure_stroke_visibility
    )
    frame_count = max(1, int(round(duration * fps)))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_mpeg = output_path.suffix.lower() in {".mpg", ".mpeg"}
    codec = "mpeg2video" if is_mpeg else "libx264"
    output_params = [] if is_mpeg else ["-movflags", "+faststart"]
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (render_width, render_height),
        fps=fps,
        codec=codec,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=9,
        macro_block_size=1,
        ffmpeg_log_level="warning",
        output_params=output_params,
    )
    writer.send(None)
    event_index = 0
    cancelled = False
    preview_available = preview
    last_percent = -1
    last_visible = permanent_revealed.copy()
    schedule_end = schedule[-1]["end"] if schedule else 0.0
    pencil_axis = np.asarray((0.67, -0.74), np.float64)
    base_pencil_angle = float(np.arctan2(pencil_axis[1], pencil_axis[0]))
    try:
        for frame_index in range(frame_count):
            time = duration * frame_index / max(1, frame_count - 1)
            while event_index < len(schedule) and time >= schedule[event_index]["end"]:
                event = schedule[event_index]
                if event["kind"] == "draw":
                    stroke_pixels = ownership.pixels_by_stroke[event["stroke_index"]]
                    permanent_revealed[stroke_pixels] = True
                event_index += 1

            visible = permanent_revealed.copy()
            cursor_point = np.asarray((0.0, 0.0))
            cursor_tangent = np.asarray((1.0, 1.0))
            drawing_now = False
            if event_index < len(schedule):
                event = schedule[event_index]
                span = max(1e-9, event["end"] - event["start"])
                progress = float(np.clip((time - event["start"]) / span, 0.0, 1.0))
                if event["kind"] == "draw":
                    drawing_now = True
                    distance = progress * event["cumulative"][-1]
                    cursor_point, cursor_tangent = _point_at(
                        event["array"], event["cumulative"], distance
                    )
                    stroke_index = event["stroke_index"]
                    reveal_count = int(
                        np.searchsorted(
                            ownership.progress_by_stroke[stroke_index], distance, side="right"
                        )
                    )
                    visible[ownership.pixels_by_stroke[stroke_index][:reveal_count]] = True
                elif event["kind"] == "move":
                    cursor_point = event["from"] + progress * (event["to"] - event["from"])
                    cursor_tangent = event["to"] - event["from"]
                else:
                    cursor_point = event["point"]
            elif strokes:
                cursor_point = np.asarray(strokes[-1]["points"][-1], np.float64)

            # Pixels too far from any trajectory fill outward from their nearest
            # guide during the clean final hold. Thus no source detail stays lost.
            if event_index >= len(schedule) and len(ownership.residual_pixels):
                residual_span = max(1e-9, duration - schedule_end)
                residual_progress = float(np.clip((time - schedule_end) / residual_span, 0.0, 1.0))
                residual_count = int(round(residual_progress * len(ownership.residual_pixels)))
                visible[ownership.residual_pixels[:residual_count]] = True

            frame = np.full((render_height, render_width, 3), 255, np.uint8)
            frame.reshape((-1, 3))[visible] = appearance_flat[visible]
            if event_index < len(schedule):
                if drawing_now:
                    tangent_norm = float(np.linalg.norm(cursor_tangent))
                    if tangent_norm:
                        tangent_unit = cursor_tangent / tangent_norm
                        influence = float(np.clip(0.72 * tangent_unit[0] + 0.28 * tangent_unit[1], -1.0, 1.0))
                        target_angle = base_pencil_angle + np.deg2rad(7.0) * influence
                        target_axis = np.asarray((np.cos(target_angle), np.sin(target_angle)))
                        pencil_axis = 0.88 * pencil_axis + 0.12 * target_axis
                        pencil_axis /= np.linalg.norm(pencil_axis)
                _draw_cursor(frame, cursor_point, pencil_axis, cursor)
            last_visible = visible
            writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())

            percent = int((frame_index + 1) * 100 / frame_count)
            if percent != last_percent:
                filled = percent // 5
                print(f"\r      [{'#' * filled}{'-' * (20 - filled)}] {percent:3d}%", end="", flush=True)
                last_percent = percent

            if preview_available:
                try:
                    cv2.imshow("Drawing Preview — Q to stop", frame)
                    # Pace the visible preview at the requested video rate. Headless
                    # rendering remains unthrottled and can finish much faster.
                    key = cv2.waitKey(max(1, round(1000 / fps))) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        cancelled = True
                        break
                    if key == ord(" "):
                        while True:
                            paused_key = cv2.waitKey(50) & 0xFF
                            if paused_key == ord(" "):
                                break
                            if paused_key in (ord("q"), ord("Q")):
                                cancelled = True
                                break
                        if cancelled:
                            break
                except cv2.error:
                    preview_available = False
                    print("\n      Preview unavailable; continuing headlessly.")
    except KeyboardInterrupt:
        cancelled = True
        print("\n      Rendering interrupted; finalizing the partial video.")
    finally:
        writer.close()
        if preview_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    rendered_frames = frame_index + 1 if frame_count else 0
    final_gray = np.full((render_height, render_width), 255, np.uint8)
    appearance_gray_padded = cv2.cvtColor(appearance, cv2.COLOR_BGR2GRAY)
    final_gray.reshape(-1)[last_visible] = appearance_gray_padded.reshape(-1)[last_visible]
    difference = cv2.absdiff(appearance_gray_padded[:height, :width], final_gray[:height, :width])
    if difference_path is not None:
        difference_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(difference_path), difference)
    reconstruction = 100.0 * (1.0 - float(np.mean(difference)) / 255.0)
    if len(ownership.foreground_pixels):
        foreground_revealed = 100.0 * float(
            np.mean(last_visible[ownership.foreground_pixels])
        )
    else:
        foreground_revealed = 100.0
    return RenderResult(
        rendered_frames,
        draw_speed,
        cancelled,
        reconstruction,
        foreground_revealed,
    )
