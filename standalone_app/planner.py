import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from utils import (
    CIRCLE_RADIUS,
    PlanResult,
    binarize_mask,
    fill_small_holes,
    inscribed_center,
    largest_connected_component,
)


def compute_faz_center(faz_mask: np.ndarray) -> Tuple[int, int]:
    faz_bin = binarize_mask(faz_mask)
    faz_lcc = largest_connected_component(faz_bin)
    return inscribed_center(faz_lcc)


def generate_ring_points(
    center: Tuple[int, int],
    radius: float,
    min_distance: int = 50,
) -> np.ndarray:
    if radius <= 0:
        return np.empty((0, 2), dtype=np.int32)
    if radius * 2 < min_distance:
        return np.empty((0, 2), dtype=np.int32)
    angle_step = 2 * math.asin(min_distance / (2 * radius))
    if angle_step <= 0:
        return np.empty((0, 2), dtype=np.int32)
    num_points = int(math.floor(2 * math.pi / angle_step))
    if num_points < 1:
        return np.empty((0, 2), dtype=np.int32)
    angles = np.linspace(0, 2 * math.pi, num=num_points, endpoint=False)
    cx, cy = center
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    coords = np.stack([xs, ys], axis=1)
    coords_int = np.rint(coords).astype(np.int32)
    return coords_int


def plan_surgery(
    image: Image.Image,
    mask: np.ndarray,
    faz_center: Tuple[int, int],
    radius_step: int = 50,
) -> PlanResult:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    mask_bin = binarize_mask(mask)
    mask_bin = fill_small_holes(mask_bin, area_threshold=200)
    h, w = mask_bin.shape
    cx, cy = faz_center
    max_radius = int(
        max(
            math.hypot(cx, cy),
            math.hypot(cx, h - 1 - cy),
            math.hypot(w - 1 - cx, cy),
            math.hypot(w - 1 - cx, h - 1 - cy),
        )
    )

    all_curve_points: List[np.ndarray] = []
    all_centers: List[Tuple[int, int]] = []

    radius = radius_step
    while radius <= max_radius:
        ring_points = generate_ring_points(faz_center, radius, min_distance=50)
        if len(ring_points) > 0:
            all_curve_points.append(ring_points)
        for x, y in ring_points:
            if 0 <= x < w and 0 <= y < h and mask_bin[y, x] > 0:
                all_centers.append((int(x), int(y)))
        radius += radius_step

    for center in all_centers:
        x, y = center
        draw.ellipse(
            (x - CIRCLE_RADIUS, y - CIRCLE_RADIUS, x + CIRCLE_RADIUS, y + CIRCLE_RADIUS),
            outline=(0, 0, 255),
            width=2,
        )

    return PlanResult(overlay=overlay, curve_points=all_curve_points, circle_centers=all_centers)
