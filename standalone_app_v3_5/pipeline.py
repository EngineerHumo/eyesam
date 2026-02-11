import logging
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from inference import InferenceResult, OnnxModel
from planner import compute_faz_center
from preprocess import PreSegmentation
from utils import (
    Click,
    ModelImage,
    binarize_mask,
    fill_small_holes,
    connected_component_centroid,
    largest_connected_component,
    prepare_image_for_model,
    remove_small_components,
    resize_mask,
)

LOGGER = logging.getLogger(__name__)


class SurgicalPipeline:
    def __init__(self, onnx_dir: str):
        self.pre_model = PreSegmentation(f"{onnx_dir}/pre.onnx")
        self.first_model = OnnxModel(f"{onnx_dir}/first.onnx")
        self.iteration_model = OnnxModel(f"{onnx_dir}/iteration.onnx")

    def _prepare_click(self, point: Tuple[int, int], label: int = 1) -> List[Click]:
        click = Click(x=float(point[0]), y=float(point[1]), label=label)
        LOGGER.info("auto_click=(%d,%d)", point[0], point[1])
        return [click]

    def _postprocess_first_mask(self, mask: np.ndarray) -> np.ndarray:
        cleaned = remove_small_components(mask, min_size=600)
        filled = fill_small_holes(cleaned, area_threshold=400)
        return filled

    def _apply_area_constraint(self, mask: np.ndarray, area_mask: np.ndarray) -> np.ndarray:
        area_bin = binarize_mask(area_mask)
        return (mask > 0).astype(np.uint8) * area_bin

    def _mask_area(self, mask: np.ndarray) -> int:
        return int(np.sum(mask > 0))

    def _is_valid_scheme(self, candidate: np.ndarray, existing: List[np.ndarray]) -> bool:
        candidate_area = self._mask_area(candidate)
        if candidate_area == 0:
            return False
        for mask in existing:
            existing_area = self._mask_area(mask)
            if existing_area == 0:
                continue
            intersection = int(np.sum((candidate > 0) & (mask > 0)))
            if intersection >= existing_area / 2:
                return False
            if intersection >= candidate_area / 2:
                return False
            ratio = existing_area / candidate_area
            if ratio <= 0.2 or ratio >= 5:
                return False
        return True

    def _run_first_with_click(
        self,
        first_image: ModelImage,
        resized_hw: Tuple[int, int],
        click: Tuple[int, int],
    ) -> Tuple[InferenceResult, Click]:
        click_list = self._prepare_click(click, label=1)
        result = self.first_model.infer(
            first_image.resized_np,
            resized_hw=resized_hw,
            orig_hw=(first_image.original_np.shape[0], first_image.original_np.shape[1]),
            clicks=click_list,
        )
        return result, click_list[0]

    def run_initial(
        self,
        image_pil,
        image_size: Tuple[int, int],
        progress_callback: Optional[Callable[[int], None]] = None,
    ):
        first_size = self.first_model.image_input_size(image_size)

        first_image = prepare_image_for_model(image_pil, first_size)

        pre_result = self.pre_model.infer(np.array(image_pil))
        faz_display_mask = pre_result.faz_mask
        area_display_mask = pre_result.area_mask

        area_bin = binarize_mask(area_display_mask)
        area_lcc = largest_connected_component(area_bin)
        click0 = connected_component_centroid(area_lcc)
        LOGGER.info("auto_click0=(%d,%d)", click0[0], click0[1])

        resized_hw = (first_image.resized_np.shape[0], first_image.resized_np.shape[1])
        first_result, last_auto_click = self._run_first_with_click(first_image, resized_hw, click0)
        run_count = 1
        if progress_callback:
            progress_callback(run_count)

        current_result = first_result
        current_click = click0
        for idx in range(4):
            prev_bin = binarize_mask(current_result.mask)
            prev_lcc = largest_connected_component(prev_bin)
            current_click_raw = connected_component_centroid(prev_lcc)
            mask_h, mask_w = current_result.mask.shape
            scale_x_first = first_image.original_pil.width / mask_w
            scale_y_first = first_image.original_pil.height / mask_h
            current_click = (
                int(current_click_raw[0] * scale_x_first),
                int(current_click_raw[1] * scale_y_first),
            )
            LOGGER.info("auto_click%d=(%d,%d)", idx + 1, current_click[0], current_click[1])
            current_result, last_auto_click = self._run_first_with_click(
                first_image, resized_hw, current_click
            )
            run_count += 1
            if progress_callback:
                progress_callback(run_count)

        faz_center = compute_faz_center(faz_display_mask)
        display_mask = resize_mask(
            current_result.mask,
            (image_pil.width, image_pil.height),
        )
        display_mask = self._postprocess_first_mask(display_mask)
        display_mask = self._apply_area_constraint(display_mask, area_display_mask)

        area_total = self._mask_area(area_bin)
        scheme_masks = [display_mask]
        scheme_logits = [current_result.logits]
        scheme_clicks = [last_auto_click]
        scheme_union = display_mask.copy()
        coverage = self._mask_area(scheme_union) / area_total if area_total > 0 else 0.0
        if coverage >= 0.9:
            scheme_masks[0] = area_bin
            scheme_union = area_bin.copy()
            coverage = 1.0
        else:
            rejected_union = np.zeros_like(area_bin, dtype=np.uint8)
            while run_count < 15 and coverage < 0.9:
                remaining = area_bin * (1 - scheme_union) * (1 - rejected_union)
                if self._mask_area(remaining) == 0:
                    break
                new_center = connected_component_centroid(remaining)
                LOGGER.info("auto_scheme_click=(%d,%d)", new_center[0], new_center[1])
                candidate_result, candidate_click = self._run_first_with_click(
                    first_image, resized_hw, new_center
                )
                run_count += 1
                if progress_callback:
                    progress_callback(run_count)
                candidate_mask = resize_mask(
                    candidate_result.mask,
                    (image_pil.width, image_pil.height),
                )
                candidate_mask = self._postprocess_first_mask(candidate_mask)
                candidate_mask = self._apply_area_constraint(candidate_mask, area_display_mask)
                if self._is_valid_scheme(candidate_mask, scheme_masks):
                    scheme_masks.append(candidate_mask)
                    scheme_logits.append(candidate_result.logits)
                    scheme_clicks.append(candidate_click)
                    scheme_union = np.maximum(scheme_union, candidate_mask)
                    coverage = (
                        self._mask_area(scheme_union) / area_total if area_total > 0 else coverage
                    )
                else:
                    rejected_union = np.maximum(rejected_union, candidate_mask)
        return (
            scheme_masks,
            scheme_logits,
            scheme_clicks,
            current_result.logits,
            last_auto_click,
            current_click,
            faz_center,
            area_display_mask,
            faz_display_mask,
        )

    def run_iteration(
        self,
        image: ModelImage,
        clicks: List[Click],
        prev_logits: np.ndarray,
    ) -> InferenceResult:
        resized_hw = (image.resized_np.shape[0], image.resized_np.shape[1])
        mask_input_shape = None
        for shape in self.iteration_model.io.input_shapes.values():
            if len(shape) == 4 and shape[1] == 1:
                if shape[2] in (-1, None) or shape[3] in (-1, None):
                    mask_input_shape = None
                else:
                    mask_input_shape = (shape[2], shape[3])
                break
        if mask_input_shape is None:
            mask_input_shape = (resized_hw[0] // 4, resized_hw[1] // 4)

        mask_input = cv2.resize(
            prev_logits.astype(np.float32),
            (mask_input_shape[1], mask_input_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )[None, None, ...]

        return self.iteration_model.infer(
            image.resized_np,
            resized_hw=resized_hw,
            orig_hw=(image.original_np.shape[0], image.original_np.shape[1]),
            clicks=clicks,
            mask_input=mask_input,
        )
