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


def inscribed_center(mask: np.ndarray) -> Tuple[int, int]:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if mask.max() == 0:
        h, w = mask.shape
        LOGGER.warning("Mask is empty, fallback to image center")
        return w // 2, h // 2
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 0)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


def scale_point(point: Tuple[float, float], scale_x: float, scale_y: float) -> Tuple[float, float]:
    return point[0] * scale_x, point[1] * scale_y


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
