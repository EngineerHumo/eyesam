import logging
from typing import List, Tuple

import cv2
import numpy as np

from inference import InferenceResult, OnnxModel
from planner import compute_faz_center, plan_surgery
from preprocess import PreSegmentation
from utils import (
    Click,
    ModelImage,
    binarize_mask,
    inscribed_center,
    largest_connected_component,
    prepare_image_for_model,
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

    def run_initial(self, image_pil, image_size: Tuple[int, int]):
        first_size = self.first_model.image_input_size(image_size)

        first_image = prepare_image_for_model(image_pil, first_size)

        pre_result = self.pre_model.infer(np.array(image_pil))
        faz_display_mask = pre_result.faz_mask
        area_display_mask = pre_result.area_mask

        area_bin = binarize_mask(area_display_mask)
        area_lcc = largest_connected_component(area_bin)
        click0 = inscribed_center(area_lcc)
        LOGGER.info("auto_click0=(%d,%d)", click0[0], click0[1])

        first_clicks = self._prepare_click(click0, label=1)
        last_auto_click = first_clicks[0]
        resized_hw = (first_image.resized_np.shape[0], first_image.resized_np.shape[1])
        first_result = self.first_model.infer(
            first_image.resized_np,
            resized_hw=resized_hw,
            orig_hw=(first_image.original_np.shape[0], first_image.original_np.shape[1]),
            clicks=first_clicks,
        )

        current_result = first_result
        current_click = click0
        for idx in range(4):
            prev_bin = binarize_mask(current_result.mask)
            prev_lcc = largest_connected_component(prev_bin)
            current_click_raw = inscribed_center(prev_lcc)
            mask_h, mask_w = current_result.mask.shape
            scale_x_first = first_image.original_pil.width / mask_w
            scale_y_first = first_image.original_pil.height / mask_h
            current_click = (
                int(current_click_raw[0] * scale_x_first),
                int(current_click_raw[1] * scale_y_first),
            )
            LOGGER.info("auto_click%d=(%d,%d)", idx + 1, current_click[0], current_click[1])
            click_list = self._prepare_click(current_click, label=1)
            last_auto_click = click_list[0]
            current_result = self.first_model.infer(
                first_image.resized_np,
                resized_hw=resized_hw,
                orig_hw=(first_image.original_np.shape[0], first_image.original_np.shape[1]),
                clicks=click_list,
            )

        faz_center = compute_faz_center(faz_display_mask)
        display_mask = resize_mask(
            current_result.mask,
            (image_pil.width, image_pil.height),
        )
        LOGGER.info("planning_with_initial_plan=%s", True)
        plan = plan_surgery(image_pil, display_mask, faz_center, area_mask=area_display_mask)
        return (
            display_mask,
            current_result.logits,
            last_auto_click,
            current_click,
            faz_center,
            plan,
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
