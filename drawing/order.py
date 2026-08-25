from __future__ import annotations

import numpy as np


STAGES = ("structural", "medium", "detail")


def _is_closed(points: list[list[int]]) -> bool:
    return len(points) > 2 and points[0] == points[-1]


def _orient_from(
    points: list[list[int]], current: np.ndarray
) -> tuple[list[list[int]], np.ndarray, float]:
    array = np.asarray(points, dtype=np.float64)
    if _is_closed(points):
        core = array[:-1]
        distances = np.linalg.norm(core - current, axis=1)
        start_index = int(np.argmin(distances))
        rotated = np.concatenate(
            (core[start_index:], core[:start_index], core[start_index : start_index + 1])
        )
        return rotated.astype(int).tolist(), rotated[0], float(distances[start_index])

    start_distance = float(np.linalg.norm(array[0] - current))
    end_distance = float(np.linalg.norm(array[-1] - current))
    if end_distance < start_distance:
        reversed_points = list(reversed(points))
        return reversed_points, np.asarray(reversed_points[0], np.float64), end_distance
    return points, array[0], start_distance


def _order_stage(
    candidates: list[dict], current: np.ndarray
) -> tuple[list[dict], np.ndarray]:
    if not candidates:
        return [], current

    remaining = list(candidates)
    starts = np.asarray(
        [stroke["points"][0] for stroke in remaining], dtype=np.float64
    )
    ends = np.asarray(
        [stroke["points"][-1] for stroke in remaining], dtype=np.float64
    )
    active = np.ones(len(remaining), dtype=bool)
    ordered: list[dict] = []

    for _ in range(len(remaining)):
        start_distances = np.linalg.norm(starts - current, axis=1)
        end_distances = np.linalg.norm(ends - current, axis=1)
        use_end = end_distances < start_distances
        candidate_starts = np.where(use_end[:, None], ends, starts)
        travel = np.minimum(start_distances, end_distances)
        upward_backtrack = np.maximum(0.0, current[1] - candidate_starts[:, 1])
        left_backtrack = np.maximum(0.0, current[0] - candidate_starts[:, 0])
        costs = travel + 0.07 * upward_backtrack + 0.025 * left_backtrack
        costs += 0.002 * candidate_starts[:, 1] + 0.001 * candidate_starts[:, 0]
        costs[~active] = np.inf
        best_index = int(np.argmin(costs))

        stroke = remaining[best_index]
        best_points, _, _ = _orient_from(stroke["points"], current)
        stroke["points"] = best_points
        ordered.append(stroke)
        active[best_index] = False
        current = np.asarray(stroke["points"][-1], np.float64)

    return ordered, current


def _order_hatch_groups(
    candidates: list[dict], current: np.ndarray
) -> tuple[list[dict], np.ndarray]:
    groups: dict[int, list[dict]] = {}
    unique_fallback = -1
    for stroke in candidates:
        group = int(stroke.get("shade_group") or 0)
        if group == 0:
            group = unique_fallback
            unique_fallback -= 1
        groups.setdefault(group, []).append(stroke)

    ordered: list[dict] = []
    while groups:
        best_group = min(
            groups,
            key=lambda group: min(
                min(
                    np.linalg.norm(
                        np.asarray(stroke["points"][0], np.float64) - current
                    ),
                    np.linalg.norm(
                        np.asarray(stroke["points"][-1], np.float64) - current
                    ),
                )
                for stroke in groups[group]
            ),
        )
        ordered_group, current = _order_stage(groups.pop(best_group), current)
        ordered.extend(ordered_group)
    return ordered, current


def assign_stages_and_order(
    strokes: list[dict], width: int, height: int
) -> tuple[list[dict], dict[str, int]]:
    line_indices = [
        index
        for index, stroke in enumerate(strokes)
        if stroke.get("kind") != "hatch"
    ]
    ranked = sorted(
        line_indices, key=lambda index: (-strokes[index]["length"], index)
    )
    count = len(line_indices)
    structural_count = int(round(count * 0.30))
    medium_count = int(round(count * 0.40))
    structural = set(ranked[:structural_count])
    medium = set(ranked[structural_count : structural_count + medium_count])

    buckets = {stage: [] for stage in STAGES}
    for index, original in enumerate(strokes):
        stroke = dict(original)
        stroke["points"] = [list(point) for point in original["points"]]
        if original.get("kind") == "hatch":
            stage = "detail"
        elif index in structural:
            stage = "structural"
        elif index in medium:
            stage = "medium"
        else:
            stage = "detail"
        stroke["stage"] = stage
        buckets[stage].append(stroke)

    result: list[dict] = []
    current = np.asarray((width * 0.5, height * 0.5), dtype=np.float64)
    for stage in STAGES:
        lines = [
            stroke for stroke in buckets[stage] if stroke.get("kind") != "hatch"
        ]
        hatches = [
            stroke for stroke in buckets[stage] if stroke.get("kind") == "hatch"
        ]
        ordered_lines, current = _order_stage(lines, current)
        result.extend(ordered_lines)
        ordered_hatches, current = _order_hatch_groups(hatches, current)
        result.extend(ordered_hatches)

    for order, stroke in enumerate(result):
        stroke["order"] = order
        stroke["id"] = order
    return result, {stage: len(buckets[stage]) for stage in STAGES}
