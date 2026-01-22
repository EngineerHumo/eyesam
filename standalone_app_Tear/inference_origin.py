import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from utils import Click, binarize_mask, log_clicks, normalize_image, sigmoid

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelIO:
    input_shapes: Dict[str, Tuple[int, ...]]
    input_names: List[str]
    output_names: List[str]


@dataclass
class InferenceResult:
    mask: np.ndarray
    logits: np.ndarray


class OnnxModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.io = self._inspect_io()

    def _inspect_io(self) -> ModelIO:
        input_shapes = {i.name: tuple(i.shape) for i in self.session.get_inputs()}
        input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        return ModelIO(input_shapes=input_shapes, input_names=input_names, output_names=output_names)

    def image_input_size(self, fallback: Tuple[int, int]) -> Tuple[int, int]:
        for shape in self.io.input_shapes.values():
            if len(shape) == 4:
                if shape[1] in (1, 3):
                    h, w = shape[2], shape[3]
                elif shape[-1] in (1, 3):
                    h, w = shape[1], shape[2]
                else:
                    continue
                if h in (-1, None) or w in (-1, None):
                    return fallback
                return int(w), int(h)
        return fallback

    def _resolve_image_input(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        for name, shape in self.io.input_shapes.items():
            if len(shape) == 4:
                channels_first = shape[1] in (1, 3)
                channels_last = shape[-1] in (1, 3)
                if channels_first or channels_last:
                    img = normalize_image(image)
                    if channels_first:
                        img = img.transpose(2, 0, 1)
                    img = img.astype(np.float32)[None, ...]
                    return {name: img}
        raise ValueError("No valid image input found in ONNX model")

    def _resolve_points_inputs(
        self,
        clicks: Optional[List[Click]],
        resized_hw: Tuple[int, int],
        orig_hw: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        if not clicks:
            return {}
        points = np.array([[c.x, c.y] for c in clicks], dtype=np.float32)
        labels = np.array([c.label for c in clicks], dtype=np.int64)
        log_clicks(clicks, "prompt")

        target_h, target_w = resized_hw
        scale_x = target_w / orig_hw[1]
        scale_y = target_h / orig_hw[0]
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y

        points = points[None, ...]
        labels = labels[None, ...]

        inputs = {}
        for name, shape in self.io.input_shapes.items():
            if len(shape) == 3 and shape[-1] == 2:
                inputs[name] = points.astype(np.float32)
            if "point_labels" in name:
                inputs[name] = labels.astype(np.int64)
            if "label" in name and name not in inputs:
                inputs[name] = labels.astype(np.int64)
            if "point_coords" in name:
                inputs[name] = points.astype(np.float32)
        return inputs

    def _resolve_mask_inputs(
        self, mask_input: Optional[np.ndarray], resized_hw: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        inputs: Dict[str, np.ndarray] = {}
        for name, shape in self.io.input_shapes.items():
            if "has_mask" in name:
                mask_value = 1 if mask_input is not None else 0
                shape_dims = []
                for dim in shape:
                    if dim in (-1, None):
                        shape_dims.append(1)
                        continue
                    try:
                        shape_dims.append(int(dim))
                    except (TypeError, ValueError):
                        shape_dims.append(1)
                if len(shape_dims) == 0:
                    inputs[name] = np.array(mask_value, dtype=np.float32)
                else:
                    inputs[name] = np.full(shape_dims, mask_value, dtype=np.float32)
        for name, shape in self.io.input_shapes.items():
            if "has_mask" in name:
                continue
            if "mask_input" in name or "mask_inputs" in name or (
                len(shape) == 4 and shape[1] == 1
            ):
                if mask_input is not None:
                    inputs[name] = mask_input.astype(np.float32)
                    continue
                shape_h = shape[2]
                shape_w = shape[3]
                if shape_h in (-1, None) or shape_w in (-1, None):
                    shape_h = resized_hw[0] // 4
                    shape_w = resized_hw[1] // 4
                inputs[name] = np.zeros((1, 1, int(shape_h), int(shape_w)), dtype=np.float32)
        return inputs

    def _resolve_orig_size_inputs(self, orig_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
        inputs = {}
        for name, shape in self.io.input_shapes.items():
            if len(shape) == 2 and shape[-1] == 2 and "orig" in name:
                inputs[name] = np.array([orig_hw], dtype=np.float32)
        return inputs

    def infer(
        self,
        image: np.ndarray,
        resized_hw: Tuple[int, int],
        orig_hw: Tuple[int, int],
        clicks: Optional[List[Click]] = None,
        mask_input: Optional[np.ndarray] = None,
    ) -> InferenceResult:
        feed = {}
        feed.update(self._resolve_image_input(image))
        feed.update(self._resolve_points_inputs(clicks, resized_hw, orig_hw))
        feed.update(self._resolve_mask_inputs(mask_input, resized_hw))
        feed.update(self._resolve_orig_size_inputs(orig_hw))

        outputs = self.session.run(self.io.output_names, feed)
        if not outputs:
            raise RuntimeError("ONNX model returned no outputs")
        logits = outputs[0]
        if logits.ndim == 4:
            logits = logits[0, 0]
        elif logits.ndim == 3:
            logits = logits[0]
        mask = sigmoid(logits) if logits.max() > 1 or logits.min() < 0 else logits
        mask = binarize_mask(mask)
        return InferenceResult(mask=mask, logits=logits)
