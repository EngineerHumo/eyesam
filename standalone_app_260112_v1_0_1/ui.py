import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from typing import List, Optional

import numpy as np
from PIL import Image, ImageTk

from pipeline import SurgicalPipeline
from planner import plan_surgery
from utils import (
    Click,
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
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.btn_positive = tk.Button(
            button_frame, text="正向点击点", width=12, command=self.toggle_positive
        )
        self.btn_negative = tk.Button(
            button_frame, text="负向点击点", width=12, command=self.toggle_negative
        )
        self.btn_clear = tk.Button(
            button_frame, text="清空当前手术方案", width=16, command=self.clear_plan
        )
        self.btn_confirm = tk.Button(
            button_frame, text="确定手术方案", width=16, command=self.confirm_plan
        )

        self.btn_positive.pack(pady=8)
        self.btn_negative.pack(pady=8)
        self.btn_clear.pack(pady=30)
        self.btn_confirm.pack(pady=8)

        self._update_button_states(initial=True)

    def _update_button_states(self, initial: bool = False) -> None:
        if initial:
            self.btn_negative.config(state=tk.DISABLED)
            self.btn_confirm.config(state=tk.DISABLED)
            return

        if self.state.has_plan:
            self.btn_negative.config(state=tk.NORMAL)
            self.btn_confirm.config(state=tk.NORMAL)
        else:
            self.btn_negative.config(state=tk.DISABLED)
            self.btn_confirm.config(state=tk.DISABLED)

    def _set_mode(self, mode: str) -> None:
        if self.state.mode == mode:
            self.state.mode = "none"
        else:
            self.state.mode = mode
        self._refresh_toggle_buttons()

    def _refresh_toggle_buttons(self) -> None:
        def set_relief(button: tk.Button, active: bool) -> None:
            button.config(relief=tk.SUNKEN if active else tk.RAISED)

        set_relief(self.btn_positive, self.state.mode == "add_positive")
        set_relief(self.btn_negative, self.state.mode == "add_negative")

    def toggle_positive(self) -> None:
        self._set_mode("add_positive")
        if self.state.mode == "add_positive":
            self.btn_negative.config(relief=tk.RAISED)

    def toggle_negative(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("add_negative")
        if self.state.mode == "add_negative":
            self.btn_positive.config(relief=tk.RAISED)

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
        self.display_image = ImageTk.PhotoImage(overlay)
        self.canvas.delete("all")
        self.canvas.config(width=overlay.width, height=overlay.height)
        self.canvas.create_image(0, 0, image=self.display_image, anchor=tk.NW)
        self.canvas.configure(scrollregion=(0, 0, overlay.width, overlay.height))

    def on_canvas_click(self, event) -> None:
        if self.state.mode == "none":
            return
        if self.current_image is None:
            messagebox.showinfo("提示", "请先打开图像")
            return

        click = Click(x=float(event.x), y=float(event.y), label=1)
        if self.state.mode == "add_negative":
            click.label = 0
        LOGGER.info("user_click=(%d,%d) label=%d", event.x, event.y, click.label)

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
            return

        if not self.state.has_plan:
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
        self._update_button_states()
        self._refresh_toggle_buttons()
        self._render_overlay(self.original_pil)

    def confirm_plan(self) -> None:
        self.btn_positive.config(state=tk.DISABLED)
        self.btn_negative.config(state=tk.DISABLED)
        self.btn_clear.config(state=tk.DISABLED)
        self.btn_confirm.config(state=tk.DISABLED)
        self.state.mode = "none"
