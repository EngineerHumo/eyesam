import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from utils import PlanResult, binarize_mask, inscribed_center, largest_connected_component


def compute_faz_center(faz_mask: np.ndarray) -> Tuple[int, int]:
    faz_bin = binarize_mask(faz_mask)
    faz_lcc = largest_connected_component(faz_bin)
    return inscribed_center(faz_lcc)


def extract_arc_segments(mask: np.ndarray, center: Tuple[int, int], radius: float) -> List[np.ndarray]:
    h, w = mask.shape
    cx, cy = center
    angles = np.linspace(0, 2 * math.pi, num=720, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    coords = np.stack([xs, ys], axis=1)
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < w)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < h)
    )
    coords = coords[valid]
    coords_int = np.round(coords).astype(int)
    inside = mask[coords_int[:, 1], coords_int[:, 0]] > 0

    segments: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []
    for idx, is_inside in enumerate(inside):
        if is_inside:
            current.append((int(coords_int[idx, 0]), int(coords_int[idx, 1])))
        else:
            if current:
                segments.append(current)
                current = []
    if current:
        segments.append(current)

    return [np.array(seg, dtype=np.int32) for seg in segments if len(seg) > 1]


def place_circles_on_arc(arc: np.ndarray, min_distance: int = 50) -> List[Tuple[int, int]]:
    centers: List[Tuple[int, int]] = []
    if len(arc) < 2:
        return centers
    cum_dist = 0.0
    last_point = arc[0]
    centers.append((int(last_point[0]), int(last_point[1])))
    for point in arc[1:]:
        dist = math.hypot(point[0] - last_point[0], point[1] - last_point[1])
        cum_dist += dist
        if cum_dist >= min_distance:
            centers.append((int(point[0]), int(point[1])))
            cum_dist = 0.0
        last_point = point
    return centers


def plan_surgery(
    image: Image.Image,
    mask: np.ndarray,
    faz_center: Tuple[int, int],
    radius_step: int = 50,
) -> PlanResult:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    mask_bin = binarize_mask(mask)
    h, w = mask_bin.shape
    max_radius = int(min(w, h) / 2)

    all_curve_points: List[np.ndarray] = []
    all_centers: List[Tuple[int, int]] = []

    radius = radius_step
    while radius <= max_radius:
        segments = extract_arc_segments(mask_bin, faz_center, radius)
        for segment in segments:
            all_curve_points.append(segment)
            centers = place_circles_on_arc(segment, min_distance=50)
            all_centers.extend(centers)
        radius += radius_step

    for curve in all_curve_points:
        draw.line([tuple(pt) for pt in curve], fill=(255, 0, 0), width=2)

    circle_radius = 12
    for center in all_centers:
        x, y = center
        draw.ellipse(
            (x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius),
            outline=(0, 0, 255),
            width=2,
        )

    return PlanResult(overlay=overlay, curve_points=all_curve_points, circle_centers=all_centers)
