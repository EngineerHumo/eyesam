from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .onnx_utils import resolve_onnx_providers

CHANNEL_TO_LABEL = np.array([0, 3, 2, 1], dtype=np.uint8)


@dataclass
class PreprocessResult:
    labels: np.ndarray
    faz_mask: np.ndarray
    area_mask: np.ndarray


def _pad_to_shape(
    image: np.ndarray, target_shape: Sequence[int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    target_h, target_w = target_shape
    h, w = image.shape[:2]
    if h > target_h or w > target_w:
        raise ValueError(
            f"图像尺寸 {h}x{w} 超过模型期望尺寸 {target_h}x{target_w}，无法直接填充。"
        )

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def _prepare_model_input(
    image: np.ndarray, target_shape: Sequence[int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int]]:
    target_h, target_w = target_shape
    height, width = image.shape[:2]

    if height > target_h or width > target_w:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    processed_shape = resized.shape[:2]
    padded, pads = _pad_to_shape(resized, target_shape)
    input_array = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(input_array, axis=0), pads, processed_shape


def _infer_labels(
    session: ort.InferenceSession,
    model_input: np.ndarray,
    pads: Tuple[int, int, int, int],
    output_shape: Tuple[int, int],
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: model_input})[0]
    if output.ndim != 4:
        raise ValueError("ONNX 模型输出维度不符，期望形状为 (N, C, H, W)")
    prediction = output[0]

    pad_top, pad_bottom, pad_left, pad_right = pads
    _, padded_h, padded_w = prediction.shape
    h, w = output_shape

    start_h = pad_top
    end_h = padded_h - pad_bottom
    start_w = pad_left
    end_w = padded_w - pad_right

    cropped = prediction[:, start_h:end_h, start_w:end_w]
    if cropped.shape[1:] != (h, w):
        raise ValueError("裁剪后的尺寸与原始图像不匹配")

    channel_indices = np.argmax(cropped, axis=0).astype(np.uint8)
    labels = CHANNEL_TO_LABEL[channel_indices]
    return labels


def connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    num, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return num, comp


def boundary_band(labels: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    band = np.zeros_like(labels, dtype=bool)
    for value in range(4):
        mask = (labels == value).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        band |= dilated != eroded
    return band


def majority_filter_on_band(labels: np.ndarray, band: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    counts = []
    for value in range(4):
        mask = (labels == value).astype(np.float32)
        count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        counts.append(count)
    stacked = np.stack(counts, axis=-1)
    modes = np.argmax(stacked, axis=-1).astype(np.uint8)
    smoothed = labels.copy()
    smoothed[band] = modes[band]
    return smoothed


def dilate_red(labels: np.ndarray) -> np.ndarray:
    red_mask = labels == 3
    if not np.any(red_mask):
        return labels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(red_mask.astype(np.uint8), kernel)
    result = labels.copy()
    result[dilated.astype(bool)] = 3
    return result


def replace_component_with_neighbors(labels: np.ndarray, component_mask: np.ndarray, value: int) -> None:
    if not np.any(component_mask):
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    border = cv2.dilate(component_mask.astype(np.uint8), kernel).astype(bool)
    border &= ~component_mask
    if not np.any(border):
        labels[component_mask] = 0
        return
    neighbors = labels[border]
    counts = np.bincount(neighbors, minlength=4)
    counts[value] = 0
    new_value = int(np.argmax(counts))
    labels[component_mask] = new_value


def clean_green_components(labels: np.ndarray) -> None:
    green_mask = labels == 1
    total = int(np.sum(green_mask))
    if total == 0:
        return
    threshold = max(int(total * 0.1), 1)
    num, comp = connected_components(green_mask)
    for idx in range(1, num):
        component_mask = comp == idx
        if int(np.sum(component_mask)) < threshold:
            replace_component_with_neighbors(labels, component_mask, 1)


def keep_largest_component(labels: np.ndarray, value: int) -> None:
    mask = labels == value
    if not np.any(mask):
        return
    num, comp = connected_components(mask)
    areas = [np.sum(comp == idx) for idx in range(1, num)]
    if not areas:
        return
    largest_idx = int(np.argmax(areas)) + 1
    for idx in range(1, num):
        if idx == largest_idx:
            continue
        component_mask = comp == idx
        replace_component_with_neighbors(labels, component_mask, value)


def fill_removed_regions(labels: np.ndarray, removed_mask: np.ndarray, target_value: int) -> None:
    if not np.any(removed_mask):
        return
    binary = (labels == target_value).astype(np.uint8)
    if np.all(binary == 1):
        labels[removed_mask] = target_value
        return
    _, indices = cv2.distanceTransformWithLabels(
        binary, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    zero_coords = np.column_stack(np.where(binary == 0))
    target_indices = indices[removed_mask] - 1
    target_indices = np.clip(target_indices, 0, len(zero_coords) - 1)
    nearest_coords = zero_coords[target_indices]
    new_values = labels[nearest_coords[:, 0], nearest_coords[:, 1]]
    labels[removed_mask] = new_values


def opening_and_refill(labels: np.ndarray, value: int, radius: int = 4) -> None:
    mask = labels == value
    if not np.any(mask):
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    removed = mask & ~opened
    labels[mask] = value
    fill_removed_regions(labels, removed, value)


def area_preserving_rethreshold(labels: np.ndarray) -> None:
    original_black = labels == 0
    original_red = labels == 3
    yellow_mask = labels == 2
    green_mask = labels == 1
    yellow_green = yellow_mask | green_mask
    if not np.any(yellow_green):
        return

    protect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    protected = cv2.dilate((original_black | original_red).astype(np.uint8), protect_kernel).astype(bool)
    movable = yellow_green & ~protected
    if not np.any(movable):
        labels[original_black] = 0
        labels[original_red] = 3
        return

    kernel_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    dilated_y = cv2.dilate(yellow_mask.astype(np.uint8), kernel_band).astype(bool)
    dilated_g = cv2.dilate(green_mask.astype(np.uint8), kernel_band).astype(bool)
    band = movable & dilated_y & dilated_g
    if not np.any(band):
        band = movable

    yellow_fixed = yellow_mask & ~band
    yellow_target = int(np.sum(yellow_mask))
    yellow_to_allocate = yellow_target - int(np.sum(yellow_fixed))
    if yellow_to_allocate <= 0:
        labels[band] = 1
        labels[yellow_fixed] = 2
        labels[original_black] = 0
        labels[original_red] = 3
        return

    band_size = int(np.sum(band))
    if yellow_to_allocate >= band_size:
        labels[band] = 2
        labels[yellow_fixed] = 2
        labels[original_black] = 0
        labels[original_red] = 3
        return

    blur = cv2.GaussianBlur(
        yellow_mask.astype(np.float32),
        (0, 0),
        sigmaX=12.0,
        sigmaY=12.0,
        borderType=cv2.BORDER_REPLICATE,
    )
    values = blur[band]
    flat_band_indices = np.flatnonzero(band)

    partition_index = len(values) - yellow_to_allocate
    threshold_value = np.partition(values, partition_index)[partition_index]
    larger = values > threshold_value
    selected_count = int(np.sum(larger))

    equals = np.where(values == threshold_value)[0]
    need = yellow_to_allocate - selected_count
    if need > 0:
        tie_indices = equals[:need]
    else:
        tie_indices = np.array([], dtype=int)

    new_yellow_mask = np.zeros_like(yellow_mask, dtype=bool)
    if selected_count > 0:
        selected_idx = np.flatnonzero(larger)
        new_yellow_mask.flat[flat_band_indices[selected_idx]] = True
    if need > 0:
        new_yellow_mask.flat[flat_band_indices[tie_indices]] = True

    labels[band] = 1
    labels[new_yellow_mask] = 2
    labels[yellow_fixed] = 2
    labels[original_black] = 0
    labels[original_red] = 3


def clean_yellow_components(labels: np.ndarray) -> None:
    yellow_mask = labels == 2
    total = int(np.sum(yellow_mask))
    if total == 0:
        return

    threshold = max(int(total * 0.1), 1)
    num, comp = connected_components(yellow_mask)
    if num <= 1:
        return

    for idx in range(1, num):
        component_mask = comp == idx
        if int(np.sum(component_mask)) < threshold:
            replace_component_with_neighbors(labels, component_mask, 2)


def remove_small_components(labels: np.ndarray, min_size: int) -> None:
    for value in range(4):
        mask = labels == value
        if not np.any(mask):
            continue
        num, comp = connected_components(mask)
        if num <= 1:
            continue
        for idx in range(1, num):
            component_mask = comp == idx
            if int(np.sum(component_mask)) < min_size:
                replace_component_with_neighbors(labels, component_mask, value)


def process_label(labels: np.ndarray) -> np.ndarray:
    band = boundary_band(labels)
    labels = majority_filter_on_band(labels, band)
    labels = dilate_red(labels)
    clean_green_components(labels)
    keep_largest_component(labels, 3)
    keep_largest_component(labels, 0)
    opening_and_refill(labels, 1, radius=4)
    opening_and_refill(labels, 2, radius=4)
    area_preserving_rethreshold(labels)
    clean_yellow_components(labels)
    remove_small_components(labels, min_size=1000)
    return labels


class PreSegmentation:
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (1024, 1024),
        prefer_gpu: bool = True,
    ) -> None:
        providers = resolve_onnx_providers(prefer_gpu=prefer_gpu)
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_size = input_size

    def infer(self, image: np.ndarray) -> PreprocessResult:
        model_input, pads, processed_shape = _prepare_model_input(image, self.input_size)
        labels = _infer_labels(self.session, model_input, pads, processed_shape)
        if processed_shape != image.shape[:2]:
            labels = cv2.resize(
                labels.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        labels = process_label(labels)
        faz_mask = (labels == 3).astype(np.uint8)
        area_mask = (labels == 1).astype(np.uint8)
        return PreprocessResult(labels=labels, faz_mask=faz_mask, area_mask=area_mask)
