import logging
from typing import List, Tuple

import cv2
import numpy as np

from inference import InferenceResult, OnnxModel
from preprocess import PreSegmentation
from utils import Click, ModelImage

LOGGER = logging.getLogger(__name__)


class SurgicalPipeline:
    def __init__(self, onnx_dir: str):
        self.pre_model = PreSegmentation(f"{onnx_dir}/pre.onnx")
        self.first_model = OnnxModel(f"{onnx_dir}/first.onnx")
        self.iteration_model = OnnxModel(f"{onnx_dir}/iteration.onnx")

    def run_presegmentation(self, image_pil) -> Tuple[np.ndarray, np.ndarray]:
        pre_result = self.pre_model.infer(np.array(image_pil))
        return pre_result.area_mask, pre_result.faz_mask

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
