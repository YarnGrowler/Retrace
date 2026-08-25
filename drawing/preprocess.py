from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _composite_over_white(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(color * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    return image[:, :, :3]


def load_and_extract_ink(
    image_path: Path, threshold: str | int = "auto", max_dimension: int = 1600
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")
    image = _composite_over_white(image)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape
    longest = max(height, width)
    if longest > max_dimension:
        scale = max_dimension / longest
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    if threshold == "auto":
        _, ink = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
    else:
        _, ink = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)

    # Remove only truly isolated connected components. Thin real details are retained.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    min_area = max(2, int(np.hypot(*ink.shape) * 0.00035))
    cleaned = np.zeros_like(ink)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    # A tiny close repairs one-pixel breaks without rounding architectural corners.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gray, cleaned


def _hatch_pattern(shape: tuple[int, int], spacing: int, direction: int) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    phase = np.rint(yy + direction * 0.22 * xx).astype(np.int32)
    return ((phase % spacing) == 0).astype(np.uint8) * 255


def skeletonize(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "thinning"):
        raise RuntimeError(
            "cv2.ximgproc.thinning is unavailable. Install opencv-contrib-python, "
            "not opencv-python."
        )
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)

    # Ordinary linework should be centerlined. Broad dark regions should not:
    # their medial axes create cellular, vein-like artifacts. Detect only cores
    # that are broad in both dimensions so a long heavy pen line stays a line.
    core = (distance >= 4.0).astype(np.uint8) * 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    broad_core = np.zeros_like(core)
    for label in range(1, component_count):
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area >= 12 and min(component_width, component_height) >= 3:
            broad_core[labels == label] = 255

    if not np.any(broad_core):
        thinned = cv2.ximgproc.thinning(ink, cv2.ximgproc.THINNING_ZHANGSUEN)
        return thinned, np.zeros_like(ink), 0

    expansion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shade_region = cv2.bitwise_and(cv2.dilate(broad_core, expansion), ink)
    shade_interior = cv2.erode(
        shade_region, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1
    )

    # Leave a one-pixel shell for the original region boundary and thin the
    # remaining normal ink. The broad interior is represented with hatching.
    line_ink = ink.copy()
    line_ink[shade_interior > 0] = 0
    line_skeleton = cv2.ximgproc.thinning(
        line_ink, cv2.ximgproc.THINNING_ZHANGSUEN
    )

    shade_count, shade_labels, shade_stats, _ = cv2.connectedComponentsWithStats(
        shade_interior, 8
    )
    positive = _hatch_pattern(ink.shape, spacing=5, direction=1)
    negative = _hatch_pattern(ink.shape, spacing=5, direction=-1)
    hatch = np.zeros_like(ink)
    for label in range(1, shade_count):
        if shade_stats[label, cv2.CC_STAT_AREA] < 8:
            continue
        pattern = positive if label % 2 else negative
        hatch[(shade_labels == label) & (pattern > 0)] = 255

    # Very deep black cores receive restrained cross-hatching. Keep hatch marks
    # one pixel away from linework so graph tracing does not create new junctions.
    deep = (distance >= 10.0) & (shade_interior > 0)
    hatch[deep & (negative > 0)] = 255
    line_buffer = cv2.dilate(
        line_skeleton, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1
    )
    hatch[line_buffer > 0] = 0
    guide = cv2.bitwise_or(line_skeleton, hatch)
    return guide, hatch, int(np.count_nonzero(shade_interior))
