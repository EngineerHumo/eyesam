import logging
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelImage:
    original_pil: Image.Image
    original_np: np.ndarray
    resized_np: np.ndarray
    scale_x: float
    scale_y: float


@dataclass
class Click:
    x: float
    y: float
    label: int


@dataclass
class PlanResult:
    overlay: Image.Image
    curve_points: List[np.ndarray]
    circle_centers: List[Tuple[int, int]]


SUPPORTED_CHINESE_FONTS = (
    "Noto Sans CJK SC",
    "Noto Sans CJK",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
)

CIRCLE_RADIUS = 12
DEFAULT_SPOT_DIAMETER = 25
DEFAULT_SPOT_DISTANCE = 25
MIN_SPOT_DIAMETER = 10
MAX_SPOT_DIAMETER = 100
MIN_SPOT_DISTANCE = 3
MAX_SPOT_DISTANCE = 30


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def prepare_image_for_model(image: Image.Image, target_size: Tuple[int, int]) -> ModelImage:
    resized = image.resize(target_size, Image.BILINEAR)
    original_np = np.array(image)
    resized_np = np.array(resized)
    scale_x = target_size[0] / image.width
    scale_y = target_size[1] / image.height
    return ModelImage(
        original_pil=image,
        original_np=original_np,
        resized_np=resized_np,
        scale_x=scale_x,
        scale_y=scale_y,
    )


def normalize_image(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return img


def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (mask >= threshold).astype(np.uint8)


def fill_small_holes(mask: np.ndarray, area_threshold: int = 200) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    binary = (mask > 0).astype(np.uint8)
    inverted = (1 - binary).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    h, w = binary.shape
    filled = binary.copy()
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area >= area_threshold:
            continue
        touches_border = x == 0 or y == 0 or x + width == w or y + height == h
        if touches_border:
            continue
        filled[labels == label] = 1
    return filled


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    filtered = binary.copy()
    for label in range(1, num_labels):
        area = stats[label][cv2.CC_STAT_AREA]
        if area < min_size:
            filtered[labels == label] = 0
    return filtered


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = [(labels == i).sum() for i in range(1, num_labels)]
    largest_index = int(np.argmax(areas)) + 1
    return (labels == largest_index).astype(np.uint8)


def connected_component_centroid(mask: np.ndarray) -> Tuple[int, int]:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if mask.max() == 0:
        h, w = mask.shape
        LOGGER.warning("Mask is empty, fallback to image center")
        return w // 2, h // 2
    largest = largest_connected_component((mask > 0).astype(np.uint8))
    ys, xs = np.where(largest > 0)
    if len(xs) == 0:
        h, w = mask.shape
        LOGGER.warning("Largest component is empty, fallback to image center")
        return w // 2, h // 2

    x = int(np.rint(xs.mean()))
    y = int(np.rint(ys.mean()))
    if largest[y, x] > 0:
        return x, y

    deltas_x = xs.astype(np.int64) - x
    deltas_y = ys.astype(np.int64) - y
    distances = deltas_x * deltas_x + deltas_y * deltas_y
    nearest_idx = int(np.argmin(distances))
    return int(xs[nearest_idx]), int(ys[nearest_idx])


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)


def log_clicks(clicks: Iterable[Click], prefix: str) -> None:
    click_list = list(clicks)
    coords = np.array([[c.x, c.y] for c in click_list], dtype=np.float32)
    batched = coords[None, ...]
    LOGGER.info(
        "%s clicks=%s shape=%s",
        prefix,
        [f"({c.x:.1f},{c.y:.1f}) label={c.label}" for c in click_list],
        batched.shape,
    )
    for idx, click in enumerate(click_list):
        LOGGER.info("%s click[%d]=(%0.1f,%0.1f) label=%d", prefix, idx, click.x, click.y, click.label)
