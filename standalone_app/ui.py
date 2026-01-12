import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from typing import List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk

from pipeline import SurgicalPipeline
from planner import plan_surgery
from utils import (
    Click,
    CIRCLE_RADIUS,
    ModelImage,
    PlanResult,
    SUPPORTED_CHINESE_FONTS,
    load_image,
    prepare_image_for_model,
    resize_mask,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class AppState:
    has_plan: bool = False
    current_mask: Optional[np.ndarray] = None
    current_logits: Optional[np.ndarray] = None
    clicks: Optional[List[Click]] = None
    auto_click: Optional[Click] = None
    mode: str = "none"


class MainWindow:
    def __init__(self, root: tk.Tk, pipeline: SurgicalPipeline):
        self.root = root
        self.pipeline = pipeline
        self.state = AppState(clicks=[], auto_click=None)
        self.current_image: Optional[ModelImage] = None
        self.original_pil: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.plan: Optional[PlanResult] = None
        self.faz_center: Optional[tuple[int, int]] = None
        self.last_auto_click: Optional[tuple[int, int]] = None
        self.display_size = (640, 640)
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0
        self.preview_job: Optional[str] = None
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        self.mouse_over_canvas = False
        self.drawing_points: List[Tuple[int, int]] = []
        self.drawn_line: Optional[int] = None

        self._setup_fonts()
        self._setup_ui()

    def _setup_fonts(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        available = set(tkfont.families())
        for candidate in SUPPORTED_CHINESE_FONTS:
            if candidate in available:
                default_font.configure(family=candidate, size=11)
                LOGGER.info("Using font: %s", candidate)
                return
        LOGGER.warning("No preferred Chinese font found, using default font")

    def _setup_ui(self) -> None:
        self.root.title("手术方案规划工具")
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开", command=self.open_image)
        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        self.h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", self.on_canvas_leave)
        self.canvas.bind("<Enter>", self.on_canvas_enter)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        top_button_frame = tk.Frame(button_frame)
        top_button_frame.pack(side=tk.TOP, fill=tk.X)
        bottom_button_frame = tk.Frame(button_frame)
        bottom_button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_positive = tk.Button(
            top_button_frame, text="正向点击点", width=12, command=self.toggle_positive
        )
        self.btn_negative = tk.Button(
            top_button_frame, text="负向点击点", width=12, command=self.toggle_negative
        )
        self.lbl_circle_legend = tk.Label(
            top_button_frame,
            text="绿色圆圈：新增激光点\n蓝色圆圈：已有激光点\n红色圆圈：去除激光点",
            justify=tk.LEFT,
        )
        self.btn_add_point = tk.Button(
            top_button_frame, text="添加激光点", width=14, command=self.toggle_add_point
        )
        self.btn_remove_point = tk.Button(
            top_button_frame, text="删除激光点", width=14, command=self.toggle_remove_point
        )
        self.btn_add_area = tk.Button(
            top_button_frame, text="添加手术区域", width=14, command=self.toggle_add_area
        )
        self.btn_remove_area = tk.Button(
            top_button_frame, text="删除手术区域", width=14, command=self.toggle_remove_area
        )
        self.btn_clear = tk.Button(
            bottom_button_frame, text="清空当前手术方案", width=16, command=self.clear_plan
        )
        self.btn_confirm = tk.Button(
            bottom_button_frame, text="确定手术方案", width=16, command=self.confirm_plan
        )

        self.btn_positive.pack(pady=8)
        self.btn_negative.pack(pady=8)
        self.lbl_circle_legend.pack(padx=6, pady=4, anchor="w")
        self.btn_add_point.pack(pady=6)
        self.btn_remove_point.pack(pady=6)
        self.btn_add_area.pack(pady=6)
        self.btn_remove_area.pack(pady=6)
        self.btn_confirm.pack(pady=6)
        self.btn_clear.pack(pady=12)

        self._update_button_states(initial=True)

    def _update_button_states(self, initial: bool = False) -> None:
        if initial:
            self.btn_negative.config(state=tk.DISABLED)
            self.btn_confirm.config(state=tk.DISABLED)
            self.btn_add_point.config(state=tk.DISABLED)
            self.btn_remove_point.config(state=tk.DISABLED)
            self.btn_add_area.config(state=tk.DISABLED)
            self.btn_remove_area.config(state=tk.DISABLED)
            return

        if self.state.has_plan:
            self.btn_negative.config(state=tk.NORMAL)
            self.btn_confirm.config(state=tk.NORMAL)
            self.btn_add_point.config(state=tk.NORMAL)
            self.btn_remove_point.config(state=tk.NORMAL)
            self.btn_add_area.config(state=tk.NORMAL)
            self.btn_remove_area.config(state=tk.NORMAL)
        else:
            self.btn_negative.config(state=tk.DISABLED)
            self.btn_confirm.config(state=tk.DISABLED)
            self.btn_add_point.config(state=tk.DISABLED)
            self.btn_remove_point.config(state=tk.DISABLED)
            self.btn_add_area.config(state=tk.DISABLED)
            self.btn_remove_area.config(state=tk.DISABLED)

    def _set_mode(self, mode: str) -> None:
        if self.state.mode == mode:
            self.state.mode = "none"
        else:
            self.state.mode = mode
        self._refresh_toggle_buttons()
        if self.state.mode in {"add_positive", "add_negative"}:
            self._start_preview_loop()
        else:
            self._stop_preview_loop()

    def _refresh_toggle_buttons(self) -> None:
        def set_relief(button: tk.Button, active: bool) -> None:
            button.config(relief=tk.SUNKEN if active else tk.RAISED)

        set_relief(self.btn_positive, self.state.mode == "add_positive")
        set_relief(self.btn_negative, self.state.mode == "add_negative")
        set_relief(self.btn_add_point, self.state.mode == "add_point")
        set_relief(self.btn_remove_point, self.state.mode == "remove_point")
        set_relief(self.btn_add_area, self.state.mode == "add_area")
        set_relief(self.btn_remove_area, self.state.mode == "remove_area")

    def toggle_positive(self) -> None:
        self._set_mode("add_positive")
        if self.state.mode == "add_positive":
            self.btn_negative.config(relief=tk.RAISED)
            self.btn_add_point.config(relief=tk.RAISED)
            self.btn_remove_point.config(relief=tk.RAISED)
            self.btn_add_area.config(relief=tk.RAISED)
            self.btn_remove_area.config(relief=tk.RAISED)

    def toggle_negative(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("add_negative")
        if self.state.mode == "add_negative":
            self.btn_positive.config(relief=tk.RAISED)
            self.btn_add_point.config(relief=tk.RAISED)
            self.btn_remove_point.config(relief=tk.RAISED)
            self.btn_add_area.config(relief=tk.RAISED)
            self.btn_remove_area.config(relief=tk.RAISED)

    def toggle_add_point(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("add_point")
        if self.state.mode == "add_point":
            self.btn_positive.config(relief=tk.RAISED)
            self.btn_negative.config(relief=tk.RAISED)
            self.btn_remove_point.config(relief=tk.RAISED)
            self.btn_add_area.config(relief=tk.RAISED)
            self.btn_remove_area.config(relief=tk.RAISED)

    def toggle_remove_point(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("remove_point")
        if self.state.mode == "remove_point":
            self.btn_positive.config(relief=tk.RAISED)
            self.btn_negative.config(relief=tk.RAISED)
            self.btn_add_point.config(relief=tk.RAISED)
            self.btn_add_area.config(relief=tk.RAISED)
            self.btn_remove_area.config(relief=tk.RAISED)

    def toggle_add_area(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("add_area")
        if self.state.mode == "add_area":
            self.btn_positive.config(relief=tk.RAISED)
            self.btn_negative.config(relief=tk.RAISED)
            self.btn_add_point.config(relief=tk.RAISED)
            self.btn_remove_point.config(relief=tk.RAISED)
            self.btn_remove_area.config(relief=tk.RAISED)

    def toggle_remove_area(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("remove_area")
        if self.state.mode == "remove_area":
            self.btn_positive.config(relief=tk.RAISED)
            self.btn_negative.config(relief=tk.RAISED)
            self.btn_add_point.config(relief=tk.RAISED)
            self.btn_remove_point.config(relief=tk.RAISED)
            self.btn_add_area.config(relief=tk.RAISED)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
        )
        if not path:
            return
        image = load_image(path)
        self.original_pil = image
        model_size = self.pipeline.iteration_model.image_input_size((image.width, image.height))
        self.display_scale_x = image.width / self.display_size[0]
        self.display_scale_y = image.height / self.display_size[1]
        self.current_image = ModelImage(
            original_pil=image,
            original_np=np.array(image),
            resized_np=np.array(image.resize(model_size, Image.BILINEAR)),
            scale_x=model_size[0] / image.width,
            scale_y=model_size[1] / image.height,
        )

        (
            mask,
            logits,
            last_auto_click,
            last_click,
            faz_center,
            plan,
            _area_mask,
            _faz_mask,
        ) = self.pipeline.run_initial(image, model_size)
        self.state.current_mask = mask
        self.state.current_logits = logits
        self.state.auto_click = last_auto_click
        self.state.clicks = [last_auto_click]
        self.state.has_plan = True
        self.state.mode = "none"
        self._update_button_states()
        self._refresh_toggle_buttons()
        self.plan = plan
        self.faz_center = faz_center
        self.last_auto_click = last_click
        LOGGER.info("initial_plan_clicks=%d", 1 if last_auto_click else 0)
        self._render_overlay(plan.overlay)

    def _render_overlay(self, overlay: Image.Image) -> None:
        display_overlay = overlay.resize(self.display_size, Image.BILINEAR)
        self.display_image = ImageTk.PhotoImage(display_overlay)
        self.canvas.delete("all")
        self.canvas.config(width=self.display_size[0], height=self.display_size[1])
        self.canvas.create_image(0, 0, image=self.display_image, anchor=tk.NW)
        self.canvas.configure(scrollregion=(0, 0, self.display_size[0], self.display_size[1]))

    def _render_current_plan(self) -> None:
        if self.plan:
            self._render_overlay(self.plan.overlay)
        elif self.original_pil:
            self._render_overlay(self.original_pil)

    def _canvas_coords(self, event) -> Tuple[float, float]:
        return (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _display_to_original(self, point: Tuple[float, float]) -> Tuple[int, int]:
        return (int(point[0] * self.display_scale_x), int(point[1] * self.display_scale_y))

    def on_canvas_press(self, event) -> None:
        if self.state.mode in {"add_area", "remove_area"}:
            self._start_polygon(event)
            return
        self.on_canvas_click(event)

    def on_canvas_release(self, event) -> None:
        if self.state.mode in {"add_area", "remove_area"}:
            self._finish_polygon()

    def on_canvas_drag(self, event) -> None:
        if self.state.mode in {"add_area", "remove_area"}:
            self._extend_polygon(event)

    def on_canvas_motion(self, event) -> None:
        if self.state.mode not in {"add_positive", "add_negative"}:
            return
        coords = self._canvas_coords(event)
        if self._is_inside_display(coords):
            self.last_mouse_pos = self._display_to_original(coords)

    def on_canvas_leave(self, _event) -> None:
        self.mouse_over_canvas = False
        self._render_current_plan()

    def on_canvas_enter(self, _event) -> None:
        self.mouse_over_canvas = True

    def on_canvas_click(self, event) -> None:
        if self.state.mode == "none":
            return
        if self.current_image is None:
            messagebox.showinfo("提示", "请先打开图像")
            return

        canvas_pos = self._canvas_coords(event)
        if not self._is_inside_display(canvas_pos):
            return
        orig_pos = self._display_to_original(canvas_pos)
        click = Click(x=float(orig_pos[0]), y=float(orig_pos[1]), label=1)
        if self.state.mode == "add_negative":
            click.label = 0
        LOGGER.info("user_click=(%d,%d) label=%d", orig_pos[0], orig_pos[1], click.label)

        if not self.state.has_plan and click.label == 1:
            self.state.clicks = [click]
            self.state.auto_click = None
            first_size = self.pipeline.first_model.image_input_size(
                (self.original_pil.width, self.original_pil.height)
            )
            first_image = prepare_image_for_model(self.original_pil, first_size)
            result = self.pipeline.first_model.infer(
                first_image.resized_np,
                resized_hw=(first_image.resized_np.shape[0], first_image.resized_np.shape[1]),
                orig_hw=(first_image.original_np.shape[0], first_image.original_np.shape[1]),
                clicks=self.state.clicks,
            )
            if self.faz_center is None:
                self.faz_center = (self.original_pil.width // 2, self.original_pil.height // 2)
            display_mask = resize_mask(result.mask, (self.original_pil.width, self.original_pil.height))
            LOGGER.info("planning_with_initial_plan=%s", False)
            plan = plan_surgery(self.original_pil, display_mask, self.faz_center)
            self._apply_plan(result, plan, display_mask)
            self.state.has_plan = True
            self._update_button_states()
            self._start_preview_loop()
            return

        if not self.state.has_plan:
            return

        if self.state.mode == "add_point":
            self._apply_point_modification(orig_pos, add=True)
            return
        if self.state.mode == "remove_point":
            self._apply_point_modification(orig_pos, add=False)
            return

        self.state.clicks.append(click)
        if self.state.current_logits is None:
            messagebox.showerror("错误", "缺少上一轮 logits，无法迭代")
            return
        auto_clicks = [self.state.auto_click] if self.state.auto_click else []
        user_clicks = max(len(self.state.clicks) - len(auto_clicks), 0)
        LOGGER.info(
            "iteration_inputs auto_clicks=%d user_clicks=%d total_clicks=%d",
            len(auto_clicks),
            user_clicks,
            len(self.state.clicks),
        )
        result = self.pipeline.run_iteration(
            self.current_image,
            self.state.clicks,
            self.state.current_logits,
        )
        if self.faz_center is None:
            self.faz_center = (self.original_pil.width // 2, self.original_pil.height // 2)
        display_mask = resize_mask(result.mask, (self.original_pil.width, self.original_pil.height))
        LOGGER.info("planning_with_initial_plan=%s", bool(auto_clicks))
        plan = plan_surgery(self.original_pil, display_mask, self.faz_center)
        self._apply_plan(result, plan, display_mask)
        self._start_preview_loop()

    def _apply_plan(self, result, plan: PlanResult, display_mask: np.ndarray) -> None:
        self.state.current_mask = display_mask
        self.state.current_logits = result.logits
        self.plan = plan
        self._render_overlay(plan.overlay)

    def clear_plan(self) -> None:
        if not self.original_pil:
            return
        self.state = AppState(clicks=[], auto_click=None)
        self.plan = None
        self.last_mouse_pos = None
        self._stop_preview_loop()
        self._update_button_states()
        self._refresh_toggle_buttons()
        self._render_overlay(self.original_pil)

    def confirm_plan(self) -> None:
        self.btn_positive.config(state=tk.DISABLED)
        self.btn_negative.config(state=tk.DISABLED)
        self.btn_add_point.config(state=tk.DISABLED)
        self.btn_remove_point.config(state=tk.DISABLED)
        self.btn_add_area.config(state=tk.DISABLED)
        self.btn_remove_area.config(state=tk.DISABLED)
        self.btn_clear.config(state=tk.DISABLED)
        self.btn_confirm.config(state=tk.DISABLED)
        self.state.mode = "none"
        self._stop_preview_loop()

    def _is_inside_display(self, point: Tuple[float, float]) -> bool:
        return 0 <= point[0] < self.display_size[0] and 0 <= point[1] < self.display_size[1]

    def _start_preview_loop(self) -> None:
        if self.preview_job:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None
        self.preview_job = self.root.after(200, self._preview_step)

    def _stop_preview_loop(self) -> None:
        if self.preview_job:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None
        self._render_current_plan()

    def _preview_step(self) -> None:
        if self.state.mode not in {"add_positive", "add_negative"}:
            return
        if not self.mouse_over_canvas or self.last_mouse_pos is None:
            self.preview_job = self.root.after(200, self._preview_step)
            return
        if self.current_image is None or self.original_pil is None:
            self.preview_job = self.root.after(200, self._preview_step)
            return

        preview_click = Click(
            x=float(self.last_mouse_pos[0]),
            y=float(self.last_mouse_pos[1]),
            label=1 if self.state.mode == "add_positive" else 0,
        )

        if self.state.current_mask is None or self.state.current_logits is None or not self.state.has_plan:
            first_size = self.pipeline.first_model.image_input_size(
                (self.original_pil.width, self.original_pil.height)
            )
            first_image = prepare_image_for_model(self.original_pil, first_size)
            result = self.pipeline.first_model.infer(
                first_image.resized_np,
                resized_hw=(first_image.resized_np.shape[0], first_image.resized_np.shape[1]),
                orig_hw=(first_image.original_np.shape[0], first_image.original_np.shape[1]),
                clicks=[preview_click],
            )
        else:
            preview_clicks = list(self.state.clicks) + [preview_click]
            result = self.pipeline.run_iteration(
                self.current_image,
                preview_clicks,
                self.state.current_logits,
            )

        display_mask = resize_mask(result.mask, (self.original_pil.width, self.original_pil.height))
        if self.faz_center is None:
            self.faz_center = (self.original_pil.width // 2, self.original_pil.height // 2)
        preview_plan = plan_surgery(self.original_pil, display_mask, self.faz_center)
        overlay = self._build_preview_overlay(preview_plan)
        self._render_overlay(overlay)
        self.preview_job = self.root.after(200, self._preview_step)

    def _build_preview_overlay(self, preview_plan: PlanResult) -> Image.Image:
        base = self.original_pil.convert("RGBA")
        draw = ImageDraw.Draw(base, "RGBA")
        current_centers = set(self.plan.circle_centers) if self.plan else set()
        preview_centers = set(preview_plan.circle_centers)
        stay_centers = current_centers & preview_centers
        add_centers = preview_centers - current_centers
        remove_centers = current_centers - preview_centers

        for x, y in stay_centers:
            draw.ellipse(
                (x - CIRCLE_RADIUS, y - CIRCLE_RADIUS, x + CIRCLE_RADIUS, y + CIRCLE_RADIUS),
                outline=(0, 0, 255, 255),
                width=2,
            )
        for x, y in add_centers:
            draw.ellipse(
                (x - CIRCLE_RADIUS, y - CIRCLE_RADIUS, x + CIRCLE_RADIUS, y + CIRCLE_RADIUS),
                outline=(0, 255, 0, 200),
                fill=(0, 255, 0, 128),
                width=2,
            )
        for x, y in remove_centers:
            draw.ellipse(
                (x - CIRCLE_RADIUS, y - CIRCLE_RADIUS, x + CIRCLE_RADIUS, y + CIRCLE_RADIUS),
                outline=(255, 0, 0, 200),
                fill=(255, 0, 0, 128),
                width=2,
            )
        return base.convert("RGB")

    def _start_polygon(self, event) -> None:
        if self.current_image is None:
            return
        self.drawing_points = []
        self.canvas.delete("draw_preview")
        coords = self._canvas_coords(event)
        if self._is_inside_display(coords):
            self.drawing_points.append(coords)

    def _extend_polygon(self, event) -> None:
        coords = self._canvas_coords(event)
        if not self._is_inside_display(coords):
            return
        self.drawing_points.append(coords)
        if len(self.drawing_points) > 1:
            points = [p for xy in self.drawing_points for p in xy]
            self.canvas.delete("draw_preview")
            self.canvas.create_line(
                points, fill="yellow", width=2, tags="draw_preview", smooth=True
            )

    def _finish_polygon(self) -> None:
        self.canvas.delete("draw_preview")
        if len(self.drawing_points) < 3:
            self.drawing_points = []
            return
        polygon = [self._display_to_original(p) for p in self.drawing_points]
        if self.state.mode == "add_area":
            self._apply_area_modification(polygon, add=True)
        elif self.state.mode == "remove_area":
            self._apply_area_modification(polygon, add=False)
        self.drawing_points = []

    def _apply_point_modification(self, point: Tuple[int, int], add: bool) -> None:
        if not self.plan or self.current_image is None or self.original_pil is None:
            return
        if self.state.current_mask is None or self.state.current_logits is None:
            messagebox.showerror("错误", "缺少 mask 或 logits，无法修改")
            return
        nearest = self._find_nearest_circle(point)
        if nearest is None:
            messagebox.showinfo("提示", "未找到可用圆圈")
            return
        center, is_displayed = nearest
        if add and is_displayed:
            return
        if not add and not is_displayed:
            return
        mask_value = 1 if add else 0
        self._update_circle_mask_logits(center, mask_value)
        if self.faz_center is None:
            self.faz_center = (self.original_pil.width // 2, self.original_pil.height // 2)
        plan = plan_surgery(self.original_pil, self.state.current_mask, self.faz_center)
        self.plan = plan
        self._render_overlay(plan.overlay)

    def _apply_area_modification(self, polygon: List[Tuple[int, int]], add: bool) -> None:
        if self.current_image is None or self.original_pil is None:
            return
        if self.state.current_mask is None or self.state.current_logits is None:
            messagebox.showerror("错误", "缺少 mask 或 logits，无法修改")
            return
        poly_array = np.array([polygon], dtype=np.int32)
        mask_update = np.zeros_like(self.state.current_mask, dtype=np.uint8)
        cv2.fillPoly(mask_update, poly_array, 1)
        if add:
            self.state.current_mask = np.maximum(self.state.current_mask, mask_update)
        else:
            self.state.current_mask = self.state.current_mask * (1 - mask_update)

        self._update_logits_with_polygon(poly_array, add)
        if self.faz_center is None:
            self.faz_center = (self.original_pil.width // 2, self.original_pil.height // 2)
        plan = plan_surgery(self.original_pil, self.state.current_mask, self.faz_center)
        self.plan = plan
        self._render_overlay(plan.overlay)

    def _find_nearest_circle(self, point: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], bool]]:
        if not self.plan or self.original_pil is None:
            return None
        all_points = []
        for ring in self.plan.curve_points:
            if ring.size == 0:
                continue
            for x, y in ring:
                if 0 <= x < self.original_pil.width and 0 <= y < self.original_pil.height:
                    all_points.append((int(x), int(y)))
        if not all_points:
            return None
        points_np = np.array(all_points, dtype=np.int32)
        target = np.array(point, dtype=np.int32)
        distances = np.sum((points_np - target) ** 2, axis=1)
        idx = int(np.argmin(distances))
        nearest = (int(points_np[idx][0]), int(points_np[idx][1]))
        is_displayed = nearest in set(self.plan.circle_centers)
        return nearest, is_displayed

    def _update_circle_mask_logits(self, center: Tuple[int, int], value: int) -> None:
        if self.state.current_mask is None or self.state.current_logits is None:
            return
        h, w = self.state.current_mask.shape
        mask_circle = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_circle, center, CIRCLE_RADIUS, 1, thickness=-1)
        if value == 1:
            self.state.current_mask = np.maximum(self.state.current_mask, mask_circle)
        else:
            self.state.current_mask = self.state.current_mask * (1 - mask_circle)

        logits_h, logits_w = self.state.current_logits.shape
        scale_x = logits_w / self.original_pil.width
        scale_y = logits_h / self.original_pil.height
        logits_center = (int(center[0] * scale_x), int(center[1] * scale_y))
        logits_radius = int(max(1, round(CIRCLE_RADIUS * (scale_x + scale_y) / 2)))
        logits_circle = np.zeros((logits_h, logits_w), dtype=np.uint8)
        cv2.circle(logits_circle, logits_center, logits_radius, 1, thickness=-1)
        if value == 1:
            self.state.current_logits[logits_circle == 1] = 1
        else:
            self.state.current_logits[logits_circle == 1] = 0

    def _update_logits_with_polygon(self, polygon: np.ndarray, add: bool) -> None:
        if self.state.current_logits is None or self.original_pil is None:
            return
        logits_h, logits_w = self.state.current_logits.shape
        scale_x = logits_w / self.original_pil.width
        scale_y = logits_h / self.original_pil.height
        scaled = polygon.astype(np.float32)
        scaled[:, :, 0] = scaled[:, :, 0] * scale_x
        scaled[:, :, 1] = scaled[:, :, 1] * scale_y
        scaled = scaled.astype(np.int32)
        mask_logits = np.zeros((logits_h, logits_w), dtype=np.uint8)
        cv2.fillPoly(mask_logits, scaled, 1)
        if add:
            self.state.current_logits[mask_logits == 1] = 1
        else:
            self.state.current_logits[mask_logits == 1] = 0
