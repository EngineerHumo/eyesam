import random
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils import PlanResult, binarize_mask, fill_small_holes


def _contour_points(contour: np.ndarray, spacing: float) -> List[Tuple[int, int]]:
    if contour.ndim != 3 or contour.shape[1] != 1 or contour.shape[2] != 2:
        return []
    coords = contour[:, 0, :].astype(np.float32)
    if len(coords) == 0:
        return []
    if len(coords) == 1:
        return [(int(coords[0, 0]), int(coords[0, 1]))]
    coords = np.vstack([coords, coords[0:1]])
    seg_vec = coords[1:] - coords[:-1]
    seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
    total_len = float(seg_len.sum())
    if total_len <= 0:
        return [(int(coords[0, 0]), int(coords[0, 1]))]
    step = max(spacing, 1.0)
    num_points = max(int(total_len // step), 1)
    distances = np.linspace(0, total_len, num=num_points, endpoint=False)
    samples: List[Tuple[int, int]] = []
    seg_index = 0
    accum = 0.0
    for dist in distances:
        while seg_index < len(seg_len) and accum + seg_len[seg_index] < dist:
            accum += seg_len[seg_index]
            seg_index += 1
        if seg_index >= len(seg_len):
            break
        ratio = 0.0 if seg_len[seg_index] == 0 else (dist - accum) / seg_len[seg_index]
        point = coords[seg_index] + ratio * seg_vec[seg_index]
        samples.append((int(round(point[0])), int(round(point[1]))))
    return samples


def _circle_mask(shape: Tuple[int, int], centers: Iterable[Tuple[int, int]], radius: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y in centers:
        cv2.circle(mask, (int(x), int(y)), radius, 1, thickness=-1)
    return mask


def _collect_component_masks(mask: np.ndarray) -> List[np.ndarray]:
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if component.any():
            components.append(component)
    return components


def _remove_overlaps(
    centers: List[Tuple[int, int]],
    min_center_distance: float,
) -> List[Tuple[int, int]]:
    if len(centers) < 2:
        return centers
    remaining = centers[:]
    min_dist_sq = float(min_center_distance) ** 2
    changed = True
    while changed:
        changed = False
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                dx = remaining[i][0] - remaining[j][0]
                dy = remaining[i][1] - remaining[j][1]
                if dx * dx + dy * dy < min_dist_sq:
                    remove_idx = random.choice([i, j])
                    remaining.pop(remove_idx)
                    changed = True
                    break
            if changed:
                break
    return remaining


def _component_centroid(mask: np.ndarray) -> Tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _push_outward(
    point: Tuple[int, int],
    center: Tuple[float, float],
    existing: List[Tuple[int, int]],
    min_center_distance: float,
    bounds: Tuple[int, int],
    max_steps: int = 200,
) -> Tuple[int, int]:
    x, y = point
    cx, cy = center
    dx = x - cx
    dy = y - cy
    norm = float(np.hypot(dx, dy))
    if norm == 0:
        dx, dy = 1.0, 0.0
        norm = 1.0
    step_x = dx / norm
    step_y = dy / norm
    h, w = bounds
    for _ in range(max_steps):
        if not existing:
            break
        distances = np.sum((np.array(existing) - np.array([x, y])) ** 2, axis=1)
        if np.all(distances >= min_center_distance**2):
            break
        x = int(round(x + step_x))
        y = int(round(y + step_y))
        if x < 0 or y < 0 or x >= w or y >= h:
            break
    return x, y


def plan_surgery(
    image: Image.Image,
    mask: np.ndarray,
    area_mask: np.ndarray | None,
    spot_diameter: int,
    spot_distance: int,
    max_layers: int,
) -> PlanResult:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    radius = max(int(round(spot_diameter / 2)), 1)
    min_center_distance = float(spot_diameter + spot_distance)

    mask_bin = binarize_mask(mask)
    mask_bin = fill_small_holes(mask_bin, area_threshold=200)
    h, w = mask_bin.shape

    all_centers: List[Tuple[int, int]] = []
    all_curve_points: List[np.ndarray] = []

    components = _collect_component_masks(mask_bin)
    for component in components:
        component_centers: List[Tuple[int, int]] = []
        curve_mask = component.copy()
        component_center = _component_centroid(component)
        layer_count = 0
        while True:
            if layer_count >= max_layers:
                break
            if layer_count == 0:
                dilate_radius = radius
            else:
                dilate_radius = radius + spot_distance
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1)
            )
            dilated = cv2.dilate(curve_mask, kernel)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                break
            layer_added = []
            for contour in contours:
                points = _contour_points(contour, spacing=min_center_distance * 0.85)
                if points:
                    all_curve_points.append(np.array(points, dtype=np.int32))
                for x, y in points:
                    if not (0 <= x < w and 0 <= y < h):
                        continue
                    if mask_bin[y, x] == 1:
                        continue
                    adjusted = _push_outward(
                        (int(x), int(y)),
                        component_center,
                        component_centers,
                        min_center_distance,
                        (h, w),
                    )
                    ax, ay = adjusted
                    if not (0 <= ax < w and 0 <= ay < h):
                        continue
                    layer_added.append((ax, ay))
            verified_layer: List[Tuple[int, int]] = []
            for x, y in layer_added:
                if verified_layer:
                    distances = np.sum(
                        (np.array(verified_layer) - np.array([x, y])) ** 2, axis=1
                    )
                    if np.any(distances < min_center_distance**2):
                        continue
                verified_layer.append((x, y))
            if verified_layer:
                component_centers.extend(verified_layer)
            layer_mask = _circle_mask(curve_mask.shape, verified_layer, radius)
            curve_mask = layer_mask if verified_layer else dilated
            layer_count += 1
            if layer_count > 100:
                break
        all_centers.extend(component_centers)

    all_centers = _remove_overlaps(all_centers, min_center_distance=min_center_distance)

    for x, y in all_centers:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(0, 0, 255),
            width=2,
        )

    return PlanResult(overlay=overlay, curve_points=all_curve_points, circle_centers=all_centers)
