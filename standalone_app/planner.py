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
    coords_int[:, 0] = np.clip(coords_int[:, 0], 0, w - 1)
    coords_int[:, 1] = np.clip(coords_int[:, 1], 0, h - 1)
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
    arc_points = arc.astype(np.float32)
    deltas = np.diff(arc_points, axis=0)
    seg_lengths = np.sqrt((deltas**2).sum(axis=1))
    cum_dist = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    total_length = cum_dist[-1]
    if total_length < min_distance:
        first = arc_points[0]
        return [(int(first[0]), int(first[1]))]

    num_points = max(1, int(math.ceil(total_length / min_distance)))
    targets = np.linspace(0.0, total_length, num_points + 1, endpoint=False)
    for target in targets:
        idx = int(np.searchsorted(cum_dist, target, side="right") - 1)
        idx = max(0, min(idx, len(arc_points) - 1))
        point = arc_points[idx]
        centers.append((int(point[0]), int(point[1])))
    filtered: List[Tuple[int, int]] = []
    min_dist_sq = min_distance * min_distance
    for candidate in centers:
        if all(
            (candidate[0] - cx) ** 2 + (candidate[1] - cy) ** 2 > min_dist_sq
            for cx, cy in filtered
        ):
            filtered.append(candidate)
    return filtered


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
        segments = extract_arc_segments(mask_bin, faz_center, radius)
        for segment in segments:
            all_curve_points.append(segment)
            centers = place_circles_on_arc(segment, min_distance=50)
            all_centers.extend(centers)
        radius += radius_step

    circle_radius = 12
    for center in all_centers:
        x, y = center
        draw.ellipse(
            (x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius),
            outline=(0, 0, 255),
            width=2,
        )

    return PlanResult(overlay=overlay, curve_points=all_curve_points, circle_centers=all_centers)
