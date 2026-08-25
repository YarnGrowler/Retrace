from __future__ import annotations

from collections import defaultdict
from math import acos, degrees

import cv2
import numpy as np


Pixel = tuple[int, int]  # (y, x)
Point = tuple[int, int]  # (x, y)
Edge = tuple[Pixel, Pixel]

NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _edge(a: Pixel, b: Pixel) -> Edge:
    return (a, b) if a <= b else (b, a)


def _neighbors(pixel: Pixel, pixels: set[Pixel]) -> list[Pixel]:
    y, x = pixel
    neighbors: list[Pixel] = []
    for dy, dx in NEIGHBOR_OFFSETS:
        candidate = (y + dy, x + dx)
        if candidate not in pixels:
            continue
        if dy and dx:
            # Keep genuine diagonal connections, but discard a diagonal edge
            # when an orthogonal pixel already links the same two pixels. The
            # redundant edge creates degree-3 triangles along ordinary digital
            # curves and fragments them into hundreds of fake junction strokes.
            if (y, x + dx) in pixels or (y + dy, x) in pixels:
                continue
        neighbors.append(candidate)
    return neighbors


def trace_skeleton(skeleton: np.ndarray) -> list[list[Point]]:
    ys, xs = np.nonzero(skeleton)
    pixels: set[Pixel] = set(zip(ys.tolist(), xs.tolist()))
    if not pixels:
        return []

    adjacency = {pixel: _neighbors(pixel, pixels) for pixel in pixels}
    degree = {pixel: len(neighbors) for pixel, neighbors in adjacency.items()}
    visited: set[Edge] = set()
    paths: list[list[Point]] = []

    def walk(start: Pixel, first: Pixel) -> list[Pixel]:
        path = [start, first]
        visited.add(_edge(start, first))
        previous, current = start, first
        while degree[current] == 2:
            candidates = [neighbor for neighbor in adjacency[current] if neighbor != previous]
            if not candidates:
                break
            following = candidates[0]
            connection = _edge(current, following)
            if connection in visited:
                break
            visited.add(connection)
            path.append(following)
            previous, current = current, following
        return path

    # Trace every branch between endpoints and junctions.
    for pixel in sorted(pixels):
        if degree[pixel] == 2:
            continue
        for neighbor in adjacency[pixel]:
            if _edge(pixel, neighbor) not in visited:
                path = walk(pixel, neighbor)
                paths.append([(x, y) for y, x in path])

    # Any unvisited edge belongs to a closed loop with no endpoints or junctions.
    for pixel in sorted(pixels):
        for neighbor in adjacency[pixel]:
            if _edge(pixel, neighbor) in visited:
                continue
            loop = walk(pixel, neighbor)
            if loop[-1] != pixel and pixel in adjacency[loop[-1]]:
                visited.add(_edge(loop[-1], pixel))
                loop.append(pixel)
            paths.append([(x, y) for y, x in loop])

    return paths


def path_length(points: list[Point]) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def _endpoint_direction(points: list[Point], side: int, reach: float = 8.0) -> np.ndarray:
    oriented = points if side == 0 else list(reversed(points))
    origin = np.asarray(oriented[0], np.float64)
    target = np.asarray(oriented[-1], np.float64)
    travelled = 0.0
    for first, second in zip(oriented, oriented[1:]):
        segment = float(np.hypot(second[0] - first[0], second[1] - first[1]))
        travelled += segment
        target = np.asarray(second, np.float64)
        if travelled >= reach:
            break
    vector = target - origin
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _continuation_connections(
    paths: list[list[Point]], tolerance_degrees: float = 25.0, junction_radius: float = 4.0
) -> dict[tuple[int, int], tuple[int, int]]:
    endpoints: list[tuple[Point, tuple[int, int]]] = []
    for index, path in enumerate(paths):
        if len(path) >= 2:
            endpoints.append((path[0], (index, 0)))
            endpoints.append((path[-1], (index, 1)))

    connections: dict[tuple[int, int], tuple[int, int]] = {}
    minimum_angle = 180.0 - tolerance_degrees
    cell_size = junction_radius
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for endpoint_index, (point, _) in enumerate(endpoints):
        cells[(int(point[0] // cell_size), int(point[1] // cell_size))].append(endpoint_index)

    candidates: list[tuple[float, float, tuple[int, int], tuple[int, int]]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for left_index, (left_point, left) in enumerate(endpoints):
        cell_x, cell_y = int(left_point[0] // cell_size), int(left_point[1] // cell_size)
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for right_index in cells.get((cell_x + offset_x, cell_y + offset_y), []):
                    pair = (min(left_index, right_index), max(left_index, right_index))
                    if left_index == right_index or pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    right_point, right = endpoints[right_index]
                    if left[0] == right[0]:
                        continue
                    gap = float(np.hypot(right_point[0] - left_point[0], right_point[1] - left_point[1]))
                    if gap > junction_radius:
                        continue
                    a = _endpoint_direction(paths[left[0]], left[1])
                    b = _endpoint_direction(paths[right[0]], right[1])
                    if not np.any(a) or not np.any(b):
                        continue
                    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
                    angle = degrees(acos(cosine))
                    if angle >= minimum_angle:
                        candidates.append((angle, gap, left, right))

    # Continue the straightest lines first, using gap only as a tie-breaker.
    for _, _, left, right in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if left in connections or right in connections:
            continue
        connections[left] = right
        connections[right] = left
    return connections


def _merge_connected_paths(paths: list[list[Point]]) -> tuple[list[list[Point]], int]:
    connections = _continuation_connections(paths)
    used: set[int] = set()
    merged_paths: list[list[Point]] = []

    def consume(start_index: int, entry_side: int) -> list[Point]:
        result: list[Point] = []
        current_index, current_entry = start_index, entry_side
        while current_index not in used:
            used.add(current_index)
            segment = paths[current_index]
            if current_entry == 1:
                segment = list(reversed(segment))
            result.extend(segment if not result else segment[1:])
            exit_side = 1 - current_entry
            following = connections.get((current_index, exit_side))
            if following is None or following[0] in used:
                break
            current_index, current_entry = following
        return result

    # Begin with chains that have at least one genuinely free end.
    for index in range(len(paths)):
        if index in used:
            continue
        if (index, 0) not in connections:
            merged_paths.append(consume(index, 0))
        elif (index, 1) not in connections:
            merged_paths.append(consume(index, 1))

    # Remaining components are closed continuation cycles.
    for index in range(len(paths)):
        if index not in used:
            merged_paths.append(consume(index, 0))

    return merged_paths, len(paths) - len(merged_paths)


def _simplify(points: list[Point], epsilon: float = 1.0) -> list[Point]:
    if len(points) <= 2:
        return points
    closed = points[0] == points[-1]
    array = np.asarray(points, np.float32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(array, epsilon, closed).reshape((-1, 2))
    result = [(int(round(x)), int(round(y))) for x, y in simplified]
    if closed and result and result[0] != result[-1]:
        result.append(result[0])
    return result


def clean_and_merge_paths(
    raw_paths: list[list[Point]], image_diagonal: float
) -> tuple[list[list[Point]], dict[str, int]]:
    minimum_length = max(3.0, image_diagonal * 0.002)
    retained = [path for path in raw_paths if path_length(path) >= minimum_length]
    removed = len(raw_paths) - len(retained)
    merged, merge_count = _merge_connected_paths(retained)
    simplified = [_simplify(path) for path in merged]
    simplified = [path for path in simplified if len(path) >= 2 and path_length(path) > 0]
    return simplified, {"removed": removed, "merged": merge_count}


def estimate_thickness(points: list[Point], distance_map: np.ndarray) -> int:
    height, width = distance_map.shape
    samples: list[float] = []

    # Simplification often leaves only junction/corner vertices. Measuring only
    # those points overestimates the whole path, so sample every segment densely.
    for start, finish in zip(points, points[1:]):
        x0, y0 = start
        x1, y1 = finish
        segment_length = float(np.hypot(x1 - x0, y1 - y0))
        sample_count = max(1, int(np.ceil(segment_length)))
        for step in range(sample_count):
            fraction = step / sample_count
            x = int(round(x0 + fraction * (x1 - x0)))
            y = int(round(y0 + fraction * (y1 - y0)))
            samples.append(
                float(distance_map[min(height - 1, max(0, y)), min(width - 1, max(0, x))])
            )
    if points:
        x, y = points[-1]
        samples.append(float(distance_map[min(height - 1, max(0, y)), min(width - 1, max(0, x))]))

    # In a discrete mask a one-pixel line has radius 1, hence 2*r-1. A modest
    # cap keeps dense/filled areas from turning into marker-like black masses.
    median_radius = float(np.median(samples)) if samples else 1.0
    apparent_width = max(1.0, 2.0 * median_radius - 1.0)
    return int(np.clip(round(apparent_width), 1, 2))
