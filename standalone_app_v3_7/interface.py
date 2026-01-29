import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline import SurgicalPipeline
from utils import Click, ModelImage, load_image, prepare_image_for_model, resize_mask

LOGGER = logging.getLogger(__name__)

MASK_TEMPLATE = "mask_{index}.npy"
LOGITS_TEMPLATE = "logits_{index}.npy"
CLICKS_TEMPLATE = "clicks_{index}.json"
CURRENT_MASK_FILE = "current_mask.npy"
CURRENT_LOGITS_FILE = "current_logits.npy"
CURRENT_CLICKS_FILE = "current_clicks.json"
PLANNING_CENTER_FILE = "planning_center.json"
PREVIEW_MASK_FILE = "preview_mask.npy"


@dataclass
class PlanArtifacts:
    masks: List[Path]
    logits: List[Path]
    clicks: List[Path]
    planning_center: Path


class SurgicalInterface:
    def __init__(self, onnx_dir: str, prefer_gpu: bool = True) -> None:
        self._ensure_onnx_files(onnx_dir)
        self.pipeline = SurgicalPipeline(onnx_dir)
        self.prefer_gpu = prefer_gpu
        self.image_path: Optional[Path] = None
        self.image_pil = None
        self.model_image: Optional[ModelImage] = None
        self.area_mask: Optional[np.ndarray] = None
        self.faz_mask: Optional[np.ndarray] = None
        self.faz_center: Optional[Tuple[int, int]] = None
        self.output_dir: Optional[Path] = None
        self.current_index: Optional[int] = None

    def _ensure_onnx_files(self, onnx_dir: str) -> None:
        required = ["pre.onnx", "first.onnx", "iteration.onnx"]
        missing = [name for name in required if not os.path.exists(os.path.join(onnx_dir, name))]
        if missing:
            raise FileNotFoundError(f"缺少 ONNX 文件: {', '.join(missing)}，请确认 {onnx_dir} 中包含模型。")

    def _set_image(self, image_path: str) -> None:
        self.image_path = Path(image_path)
        self.image_pil = load_image(image_path)
        first_size = self.pipeline.first_model.image_input_size(
            (self.image_pil.width, self.image_pil.height)
        )
        self.model_image = prepare_image_for_model(self.image_pil, first_size)

    def _ensure_output_dir(self, output_dir: str) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_path
        return output_path

    def _mask_path(self, index: int) -> Path:
        return self.output_dir / MASK_TEMPLATE.format(index=index)

    def _logits_path(self, index: int) -> Path:
        return self.output_dir / LOGITS_TEMPLATE.format(index=index)

    def _clicks_path(self, index: int) -> Path:
        return self.output_dir / CLICKS_TEMPLATE.format(index=index)

    def _current_mask_path(self) -> Path:
        return self.output_dir / CURRENT_MASK_FILE

    def _current_logits_path(self) -> Path:
        return self.output_dir / CURRENT_LOGITS_FILE

    def _current_clicks_path(self) -> Path:
        return self.output_dir / CURRENT_CLICKS_FILE

    def _planning_center_path(self) -> Path:
        return self.output_dir / PLANNING_CENTER_FILE

    def _preview_mask_path(self) -> Path:
        return self.output_dir / PREVIEW_MASK_FILE

    def _save_clicks(self, clicks: List[Click], path: Path) -> None:
        payload = {
            "points": [
                {
                    "x": float(click.x),
                    "y": float(click.y),
                    "label": int(click.label),
                }
                for click in clicks
            ]
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_clicks(self, path: Path) -> List[Click]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_points = payload.get("points", payload)
        clicks: List[Click] = []
        for point in raw_points:
            clicks.append(
                Click(
                    x=float(point["x"]),
                    y=float(point["y"]),
                    label=int(point.get("label", 1)),
                )
            )
        return clicks

    def _save_planning_center(self, center: Tuple[int, int]) -> Path:
        payload = {"x": int(center[0]), "y": int(center[1])}
        path = self._planning_center_path()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _resolve_current_paths(self) -> Tuple[Path, Path, Path]:
        if self.output_dir is None:
            raise RuntimeError("尚未设置输出目录，请先调用初始规划或提供输出路径。")
        if self.current_index is None:
            return self._current_mask_path(), self._current_logits_path(), self._current_clicks_path()
        return (
            self._mask_path(self.current_index),
            self._logits_path(self.current_index),
            self._clicks_path(self.current_index),
        )

    def initial_plan(self, image_path: str, output_dir: str) -> PlanArtifacts:
        self._set_image(image_path)
        output_path = self._ensure_output_dir(output_dir)
        result = self.pipeline.run_initial(
            self.image_pil,
            (self.image_pil.width, self.image_pil.height),
        )
        (
            scheme_masks,
            scheme_logits,
            scheme_clicks,
            _current_logits,
            _last_auto_click,
            _current_click,
            faz_center,
            area_mask,
            faz_mask,
        ) = result

        self.area_mask = area_mask
        self.faz_mask = faz_mask
        self.faz_center = faz_center
        self.current_index = None

        masks: List[Path] = []
        logits: List[Path] = []
        clicks: List[Path] = []
        for idx, (mask, logit, click) in enumerate(zip(scheme_masks, scheme_logits, scheme_clicks)):
            mask_path = output_path / MASK_TEMPLATE.format(index=idx)
            logits_path = output_path / LOGITS_TEMPLATE.format(index=idx)
            clicks_path = output_path / CLICKS_TEMPLATE.format(index=idx)
            np.save(mask_path, mask.astype(np.uint8))
            np.save(logits_path, logit.astype(np.float32))
            self._save_clicks([click], clicks_path)
            masks.append(mask_path)
            logits.append(logits_path)
            clicks.append(clicks_path)

        center_path = self._save_planning_center(faz_center)
        return PlanArtifacts(masks=masks, logits=logits, clicks=clicks, planning_center=center_path)

    def select_initial_scheme(self, mask_index: int) -> None:
        if self.output_dir is None:
            raise RuntimeError("尚未设置输出目录，请先调用初始规划。")
        target_mask = self._mask_path(mask_index)
        target_logits = self._logits_path(mask_index)
        target_clicks = self._clicks_path(mask_index)
        if not target_mask.exists() or not target_logits.exists() or not target_clicks.exists():
            raise FileNotFoundError("指定的 mask/logits/clicks 文件不存在，请检查 mask 编号。")

        for mask_path in self.output_dir.glob("mask_*.npy"):
            if mask_path == target_mask:
                continue
            mask_path.unlink()
        for logits_path in self.output_dir.glob("logits_*.npy"):
            if logits_path == target_logits:
                continue
            logits_path.unlink()
        for clicks_path in self.output_dir.glob("clicks_*.json"):
            if clicks_path == target_clicks:
                continue
            clicks_path.unlink()

        self.current_index = mask_index

    def apply_clicks(self, clicks_json: str) -> Dict[str, Path]:
        if self.image_pil is None or self.model_image is None:
            raise RuntimeError("尚未加载图像，请先调用初始规划。")
        if self.output_dir is None:
            raise RuntimeError("尚未设置输出目录，请先调用初始规划。")
        if self.area_mask is None:
            raise RuntimeError("缺少区域掩码，请先执行初始规划。")

        new_clicks = self._load_clicks(Path(clicks_json))
        mask_path, logits_path, clicks_path = self._resolve_current_paths()
        existing_clicks: List[Click] = []
        if clicks_path.exists():
            existing_clicks = self._load_clicks(clicks_path)
        merged_clicks = existing_clicks + new_clicks

        if logits_path.exists() and mask_path.exists():
            prev_logits = np.load(logits_path)
            result = self.pipeline.run_iteration(self.model_image, merged_clicks, prev_logits)
            display_mask = resize_mask(result.mask, (self.image_pil.width, self.image_pil.height))
            display_mask = self.pipeline._apply_area_constraint(display_mask, self.area_mask)
        else:
            result = self.pipeline.first_model.infer(
                self.model_image.resized_np,
                resized_hw=(
                    self.model_image.resized_np.shape[0],
                    self.model_image.resized_np.shape[1],
                ),
                orig_hw=(
                    self.model_image.original_np.shape[0],
                    self.model_image.original_np.shape[1],
                ),
                clicks=merged_clicks,
            )
            display_mask = resize_mask(result.mask, (self.image_pil.width, self.image_pil.height))
            display_mask = self.pipeline._postprocess_first_mask(display_mask)
            display_mask = self.pipeline._apply_area_constraint(display_mask, self.area_mask)

        np.save(mask_path, display_mask.astype(np.uint8))
        np.save(logits_path, result.logits.astype(np.float32))
        self._save_clicks(merged_clicks, clicks_path)
        return {"mask": mask_path, "logits": logits_path, "clicks": clicks_path}

    def preview_clicks(self, clicks_json: str) -> Path:
        if self.image_pil is None or self.model_image is None:
            raise RuntimeError("尚未加载图像，请先调用初始规划。")
        if self.output_dir is None:
            raise RuntimeError("尚未设置输出目录，请先调用初始规划。")
        if self.area_mask is None:
            raise RuntimeError("缺少区域掩码，请先执行初始规划。")

        new_clicks = self._load_clicks(Path(clicks_json))
        mask_path, logits_path, clicks_path = self._resolve_current_paths()
        existing_clicks: List[Click] = []
        if clicks_path.exists():
            existing_clicks = self._load_clicks(clicks_path)
        merged_clicks = existing_clicks + new_clicks

        if logits_path.exists() and mask_path.exists():
            prev_logits = np.load(logits_path)
            result = self.pipeline.run_iteration(self.model_image, merged_clicks, prev_logits)
            display_mask = resize_mask(result.mask, (self.image_pil.width, self.image_pil.height))
            display_mask = self.pipeline._apply_area_constraint(display_mask, self.area_mask)
        else:
            result = self.pipeline.first_model.infer(
                self.model_image.resized_np,
                resized_hw=(
                    self.model_image.resized_np.shape[0],
                    self.model_image.resized_np.shape[1],
                ),
                orig_hw=(
                    self.model_image.original_np.shape[0],
                    self.model_image.original_np.shape[1],
                ),
                clicks=merged_clicks,
            )
            display_mask = resize_mask(result.mask, (self.image_pil.width, self.image_pil.height))
            display_mask = self.pipeline._postprocess_first_mask(display_mask)
            display_mask = self.pipeline._apply_area_constraint(display_mask, self.area_mask)

        preview_path = self._preview_mask_path()
        np.save(preview_path, display_mask.astype(np.uint8))
        return preview_path

    def clear_current_plan(self) -> None:
        if self.output_dir is None:
            return
        mask_path, logits_path, clicks_path = self._resolve_current_paths()
        for path in (mask_path, logits_path, clicks_path):
            if path.exists():
                path.unlink()
        self.current_index = None
