from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

from pipeline import SurgicalPipeline
from utils import (
    Click,
    binarize_mask,
    fill_small_holes,
    load_image,
    prepare_image_for_model,
    remove_small_components,
    resize_mask,
)

STATE_FILE = "state.json"
AREA_MASK_FILE = "area_mask.npy"
FAZ_MASK_FILE = "faz_mask.npy"
PLANNING_CENTER_FILE = "planning_center.json"
CURRENT_MASK_FILE = "current_mask.npy"
CURRENT_LOGITS_FILE = "current_logits.npy"
CURRENT_CLICKS_FILE = "current_clicks.json"
SCHEME_MASK_TEMPLATE = "scheme_{index}_mask.npy"
SCHEME_LOGITS_TEMPLATE = "scheme_{index}_logits.npy"
SCHEME_CLICKS_TEMPLATE = "scheme_{index}_clicks.json"


class SurgicalPlannerAPI:
    def __init__(self, onnx_dir: str) -> None:
        self.pipeline = SurgicalPipeline(onnx_dir)

    def initial_plan(self, image_path: str, work_dir: str) -> Dict[str, Any]:
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        self._clear_scheme_files(work_path)
        self._clear_current_files(work_path)

        image = load_image(image_path)
        model_size = self.pipeline.iteration_model.image_input_size((image.width, image.height))
        (
            scheme_masks,
            scheme_logits,
            scheme_clicks,
            _logits,
            _last_auto_click,
            _last_click,
            faz_center,
            area_mask,
            faz_mask,
        ) = self.pipeline.run_initial(image, model_size)

        scheme_outputs = []
        for idx, mask in enumerate(scheme_masks):
            mask_path = work_path / SCHEME_MASK_TEMPLATE.format(index=idx)
            logits_path = work_path / SCHEME_LOGITS_TEMPLATE.format(index=idx)
            clicks_path = work_path / SCHEME_CLICKS_TEMPLATE.format(index=idx)
            np.save(mask_path, mask.astype(np.uint8))
            np.save(logits_path, scheme_logits[idx].astype(np.float32))
            self._save_clicks(clicks_path, [scheme_clicks[idx]] if scheme_clicks[idx] else [])
            scheme_outputs.append(
                {
                    "mask": str(mask_path),
                    "logits": str(logits_path),
                    "clicks": str(clicks_path),
                }
            )

        if area_mask is not None:
            np.save(work_path / AREA_MASK_FILE, area_mask.astype(np.uint8))
        if faz_mask is not None:
            np.save(work_path / FAZ_MASK_FILE, faz_mask.astype(np.uint8))

        planning_center_path = work_path / PLANNING_CENTER_FILE
        self._save_json(planning_center_path, {"x": int(faz_center[0]), "y": int(faz_center[1])})

        self._save_json(
            work_path / STATE_FILE,
            {
                "image_path": str(Path(image_path).resolve()),
                "image_width": image.width,
                "image_height": image.height,
            },
        )

        return {
            "planning_center": str(planning_center_path),
            "schemes": scheme_outputs,
        }

    def select_initial_scheme(self, work_dir: str, scheme_index: int) -> None:
        work_path = Path(work_dir)
        mask_path = work_path / SCHEME_MASK_TEMPLATE.format(index=scheme_index)
        logits_path = work_path / SCHEME_LOGITS_TEMPLATE.format(index=scheme_index)
        clicks_path = work_path / SCHEME_CLICKS_TEMPLATE.format(index=scheme_index)
        if not mask_path.exists() or not logits_path.exists():
            raise FileNotFoundError(f"未找到编号为 {scheme_index} 的初始方案")

        for idx in self._list_scheme_indices(work_path):
            if idx == scheme_index:
                continue
            (work_path / SCHEME_MASK_TEMPLATE.format(index=idx)).unlink(missing_ok=True)
            (work_path / SCHEME_LOGITS_TEMPLATE.format(index=idx)).unlink(missing_ok=True)
            (work_path / SCHEME_CLICKS_TEMPLATE.format(index=idx)).unlink(missing_ok=True)

        shutil.copyfile(mask_path, work_path / CURRENT_MASK_FILE)
        shutil.copyfile(logits_path, work_path / CURRENT_LOGITS_FILE)
        if clicks_path.exists():
            shutil.copyfile(clicks_path, work_path / CURRENT_CLICKS_FILE)
        else:
            self._save_clicks(work_path / CURRENT_CLICKS_FILE, [])

    def iterate_with_clicks(self, work_dir: str, click_json_path: str) -> Dict[str, str]:
        work_path = Path(work_dir)
        state = self._load_state(work_path)
        image = load_image(state["image_path"])
        model_size = self.pipeline.iteration_model.image_input_size((image.width, image.height))
        model_image = prepare_image_for_model(image, model_size)

        new_clicks = self._load_clicks(Path(click_json_path), require_label=True)
        existing_clicks = self._load_clicks(work_path / CURRENT_CLICKS_FILE, require_label=False)
        combined_clicks = existing_clicks + new_clicks

        area_mask = self._load_area_mask(work_path)
        if (work_path / CURRENT_LOGITS_FILE).exists():
            prev_logits = np.load(work_path / CURRENT_LOGITS_FILE)
            result = self.pipeline.run_iteration(model_image, combined_clicks, prev_logits)
            mask = resize_mask(result.mask, (image.width, image.height))
            mask = self._apply_area_constraint(mask, area_mask)
        else:
            first_size = self.pipeline.first_model.image_input_size((image.width, image.height))
            first_image = prepare_image_for_model(image, first_size)
            result = self.pipeline.first_model.infer(
                first_image.resized_np,
                resized_hw=(first_image.resized_np.shape[0], first_image.resized_np.shape[1]),
                orig_hw=(first_image.original_np.shape[0], first_image.original_np.shape[1]),
                clicks=combined_clicks,
            )
            mask = resize_mask(result.mask, (image.width, image.height))
            mask = self._postprocess_first_mask(mask, area_mask)

        mask_path = work_path / CURRENT_MASK_FILE
        logits_path = work_path / CURRENT_LOGITS_FILE
        clicks_path = work_path / CURRENT_CLICKS_FILE
        np.save(mask_path, mask.astype(np.uint8))
        np.save(logits_path, result.logits.astype(np.float32))
        self._save_clicks(clicks_path, combined_clicks)

        return {
            "mask": str(mask_path),
            "logits": str(logits_path),
            "clicks": str(clicks_path),
        }

    def add_click_point(self, work_dir: str, click_json_path: str) -> str:
        return self._update_logits_with_point(work_dir, click_json_path, value=1)

    def remove_click_point(self, work_dir: str, click_json_path: str) -> str:
        return self._update_logits_with_point(work_dir, click_json_path, value=0)

    def add_area(self, work_dir: str, polygon_json_path: str) -> str:
        return self._update_logits_with_polygon(work_dir, polygon_json_path, add=True)

    def remove_area(self, work_dir: str, polygon_json_path: str) -> str:
        return self._update_logits_with_polygon(work_dir, polygon_json_path, add=False)

    def clear_current(self, work_dir: str) -> None:
        self._clear_current_files(Path(work_dir))

    def _apply_area_constraint(
        self, mask: np.ndarray, area_mask: np.ndarray | None
    ) -> np.ndarray:
        if area_mask is None:
            return mask
        area_bin = binarize_mask(area_mask)
        return (mask > 0).astype(np.uint8) * area_bin

    def _postprocess_first_mask(
        self, mask: np.ndarray, area_mask: np.ndarray | None
    ) -> np.ndarray:
        cleaned = remove_small_components(mask, min_size=400)
        filled = fill_small_holes(cleaned, area_threshold=400)
        return self._apply_area_constraint(filled, area_mask)

    def _update_logits_with_point(self, work_dir: str, click_json_path: str, value: int) -> str:
        work_path = Path(work_dir)
        state = self._load_state(work_path)
        logits_path = work_path / CURRENT_LOGITS_FILE
        if not logits_path.exists():
            raise FileNotFoundError("缺少 current_logits.npy，无法修改 logits")
        logits = np.load(logits_path)
        point = self._load_single_point(Path(click_json_path))
        x, y = self._scale_point_to_logits(point, logits.shape, state)
        self._apply_square(logits, (x, y), size=30, value=value)
        np.save(logits_path, logits.astype(np.float32))
        return str(logits_path)

    def _update_logits_with_polygon(self, work_dir: str, polygon_json_path: str, add: bool) -> str:
        work_path = Path(work_dir)
        state = self._load_state(work_path)
        logits_path = work_path / CURRENT_LOGITS_FILE
        if not logits_path.exists():
            raise FileNotFoundError("缺少 current_logits.npy，无法修改 logits")
        logits = np.load(logits_path)
        points = self._load_polygon(Path(polygon_json_path))
        if len(points) < 3:
            raise ValueError("多边形至少需要 3 个点")
        mask = np.zeros_like(logits, dtype=np.uint8)
        scaled = self._scale_points(points, logits.shape, state)
        cv2.fillPoly(mask, [scaled], 1)
        if add:
            logits[mask == 1] = 1
        else:
            logits[mask == 1] = 0
        np.save(logits_path, logits.astype(np.float32))
        return str(logits_path)

    def _apply_square(self, logits: np.ndarray, center: Tuple[int, int], size: int, value: int) -> None:
        half = size // 2
        x, y = center
        h, w = logits.shape
        x0 = max(0, x - half)
        x1 = min(w, x + half)
        y0 = max(0, y - half)
        y1 = min(h, y + half)
        logits[y0:y1, x0:x1] = value

    def _scale_points(
        self, points: Iterable[Tuple[int, int]], logits_shape: Tuple[int, int], state: Dict[str, Any]
    ) -> np.ndarray:
        scale_x = logits_shape[1] / state["image_width"]
        scale_y = logits_shape[0] / state["image_height"]
        scaled = np.array(points, dtype=np.float32)
        scaled[:, 0] = scaled[:, 0] * scale_x
        scaled[:, 1] = scaled[:, 1] * scale_y
        return scaled.astype(np.int32)

    def _scale_point_to_logits(
        self, point: Tuple[int, int], logits_shape: Tuple[int, int], state: Dict[str, Any]
    ) -> Tuple[int, int]:
        scale_x = logits_shape[1] / state["image_width"]
        scale_y = logits_shape[0] / state["image_height"]
        return int(round(point[0] * scale_x)), int(round(point[1] * scale_y))

    def _load_area_mask(self, work_path: Path) -> np.ndarray | None:
        area_path = work_path / AREA_MASK_FILE
        if area_path.exists():
            return np.load(area_path)
        return None

    def _load_state(self, work_path: Path) -> Dict[str, Any]:
        state_path = work_path / STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError("缺少 state.json，请先执行初始规划")
        return self._load_json(state_path)

    def _load_single_point(self, path: Path) -> Tuple[int, int]:
        data = self._load_json(path)
        if isinstance(data, dict) and "x" in data and "y" in data:
            return int(data["x"]), int(data["y"])
        if isinstance(data, dict) and "point" in data:
            point = data["point"]
            return int(point["x"]), int(point["y"])
        raise ValueError("点击点 JSON 需包含 x/y 或 point.x/point.y")

    def _load_polygon(self, path: Path) -> List[Tuple[int, int]]:
        data = self._load_json(path)
        if isinstance(data, dict):
            data = data.get("points", data.get("polygon", data))
        if not isinstance(data, list):
            raise ValueError("多边形 JSON 格式无效")
        points = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("多边形点应为对象列表")
            points.append((int(item["x"]), int(item["y"])))
        return points

    def _load_clicks(self, path: Path, require_label: bool) -> List[Click]:
        if not path.exists():
            return []
        data = self._load_json(path)
        if isinstance(data, dict):
            data = data.get("clicks", data.get("points", data))
        if not isinstance(data, list):
            raise ValueError("点击点 JSON 格式无效")
        clicks = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("点击点应为对象列表")
            if require_label and "label" not in item:
                raise ValueError("点击点缺少 label")
            label = int(item.get("label", 1))
            clicks.append(Click(x=float(item["x"]), y=float(item["y"]), label=label))
        return clicks

    def _save_clicks(self, path: Path, clicks: Iterable[Click]) -> None:
        payload = {"clicks": [asdict(click) for click in clicks]}
        self._save_json(path, payload)

    def _save_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_json(self, path: Path) -> Dict[str, Any] | List[Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _list_scheme_indices(self, work_path: Path) -> List[int]:
        indices = []
        for item in work_path.glob("scheme_*_mask.npy"):
            name = item.stem
            parts = name.split("_")
            if len(parts) < 3:
                continue
            try:
                indices.append(int(parts[1]))
            except ValueError:
                continue
        return sorted(indices)

    def _clear_scheme_files(self, work_path: Path) -> None:
        for item in work_path.glob("scheme_*_mask.npy"):
            idx = item.stem.split("_")[1]
            (work_path / SCHEME_MASK_TEMPLATE.format(index=idx)).unlink(missing_ok=True)
            (work_path / SCHEME_LOGITS_TEMPLATE.format(index=idx)).unlink(missing_ok=True)
            (work_path / SCHEME_CLICKS_TEMPLATE.format(index=idx)).unlink(missing_ok=True)

    def _clear_current_files(self, work_path: Path) -> None:
        (work_path / CURRENT_MASK_FILE).unlink(missing_ok=True)
        (work_path / CURRENT_LOGITS_FILE).unlink(missing_ok=True)
        (work_path / CURRENT_CLICKS_FILE).unlink(missing_ok=True)
