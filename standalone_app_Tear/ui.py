import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from pipeline import SurgicalPipeline
from planner import plan_surgery
from utils import (
    Click,
    DEFAULT_SPOT_DIAMETER,
    DEFAULT_SPOT_DISTANCE,
    ModelImage,
    PlanResult,
    SUPPORTED_CHINESE_FONTS,
    fill_small_holes,
    load_image,
    prepare_image_for_model,
    remove_small_components,
    resize_mask,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class AppState:
    has_plan: bool = False
    current_mask: Optional[np.ndarray] = None
    current_logits: Optional[np.ndarray] = None
    clicks: Optional[List[Click]] = None
    mode: str = "none"


class MainWindow:
    def __init__(self, root: tk.Tk, pipeline: SurgicalPipeline):
        self.root = root
        self.pipeline = pipeline
        self.state = AppState(clicks=[])
        self.current_image: Optional[ModelImage] = None
        self.original_pil: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.plan: Optional[PlanResult] = None
        self.area_mask: Optional[np.ndarray] = None
        self.manual_centers: List[Tuple[int, int]] = []
        self.hidden_centers: Set[Tuple[int, int]] = set()
        self.display_size = (640, 640)
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0
        self.preview_job: Optional[str] = None
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        self.mouse_over_canvas = False
        self.spot_diameter_var = tk.IntVar(value=DEFAULT_SPOT_DIAMETER)
        self.spot_distance_var = tk.IntVar(value=DEFAULT_SPOT_DISTANCE)
        self.spot_layers_var = tk.IntVar(value=3)
        self.exposure_time_var = tk.IntVar(value=50)

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
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", self.on_canvas_leave)
        self.canvas.bind("<Enter>", self.on_canvas_enter)

        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = tk.Frame(right_panel)
        button_frame.pack(side=tk.LEFT, fill=tk.Y)

        ai_tool_frame = tk.Frame(button_frame, relief=tk.GROOVE, borderwidth=2)
        ai_tool_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Label(ai_tool_frame, text="AI工具", font=tkfont.Font(weight="bold")).pack(
            pady=(6, 4)
        )

        self.btn_positive = tk.Button(
            ai_tool_frame, text="正向点击点", width=12, command=self.toggle_positive
        )
        self.btn_negative = tk.Button(
            ai_tool_frame, text="负向点击点", width=12, command=self.toggle_negative
        )
        legend_frame = tk.Frame(ai_tool_frame)
        legend_frame.pack(padx=6, pady=4, anchor="w")
        self.icon_green = self._create_circle_icon("green")
        self.icon_blue = self._create_circle_icon("blue")
        self.icon_red = self._create_circle_icon("red")
        self._add_legend_row(legend_frame, self.icon_green, "新增激光点")
        self._add_legend_row(legend_frame, self.icon_blue, "已有激光点")
        self._add_legend_row(legend_frame, self.icon_red, "去除激光点")
        tk.Label(
            ai_tool_frame,
            text="建议先使用AI工具再用传统工具修改手术规划",
            wraplength=180,
            justify=tk.LEFT,
        ).pack(padx=6, pady=(4, 6), anchor="w")

        traditional_tool_frame = tk.Frame(button_frame, relief=tk.GROOVE, borderwidth=2)
        traditional_tool_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        tk.Label(traditional_tool_frame, text="传统工具", font=tkfont.Font(weight="bold")).pack(
            pady=(6, 4)
        )
        self.btn_add_point = tk.Button(
            traditional_tool_frame, text="添加激光点", width=14, command=self.toggle_add_point
        )
        self.btn_remove_point = tk.Button(
            traditional_tool_frame, text="删除激光点", width=14, command=self.toggle_remove_point
        )

        action_frame = tk.Frame(button_frame)
        action_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)
        self.btn_clear = tk.Button(
            action_frame, text="清空当前手术方案", width=16, command=self.clear_plan
        )
        self.btn_confirm = tk.Button(
            action_frame, text="确定手术方案", width=16, command=self.confirm_plan
        )

        self.btn_positive.pack(pady=6)
        self.btn_negative.pack(pady=6)
        self.btn_add_point.pack(pady=6)
        self.btn_remove_point.pack(pady=6)
        self.btn_clear.pack(pady=6)
        self.btn_confirm.pack(pady=6)

        slider_frame = tk.Frame(right_panel, relief=tk.GROOVE, borderwidth=2)
        slider_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        tk.Label(slider_frame, text="参数设置", font=tkfont.Font(weight="bold")).pack(
            pady=(6, 10)
        )
        self._add_spinbox(
            slider_frame,
            label="光斑层数",
            var=self.spot_layers_var,
            from_=1,
            to=6,
            command=self._on_spot_params_change,
        )
        self._add_slider(
            slider_frame,
            label="光斑直径",
            var=self.spot_diameter_var,
            from_=4,
            to=40,
            command=self._on_spot_params_change,
        )
        self._add_slider(
            slider_frame,
            label="光斑距离",
            var=self.spot_distance_var,
            from_=3,
            to=30,
            command=self._on_spot_params_change,
        )
        self._add_slider(
            slider_frame,
            label="曝光时间",
            var=self.exposure_time_var,
            from_=1,
            to=100,
            command=None,
        )

        self._update_button_states(initial=True)

    def _create_circle_icon(self, color: str, size: int = 12) -> ImageTk.PhotoImage:
        radius = size // 2 - 1
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (size // 2 - radius, size // 2 - radius, size // 2 + radius, size // 2 + radius),
            fill=color,
            outline="black",
        )
        return ImageTk.PhotoImage(image)

    def _add_legend_row(
        self, parent: tk.Frame, icon: ImageTk.PhotoImage, text: str
    ) -> None:
        row = tk.Frame(parent)
        row.pack(anchor="w")
        tk.Label(row, image=icon).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(row, text=text).pack(side=tk.LEFT)

    def _add_slider(
        self,
        parent: tk.Frame,
        label: str,
        var: tk.IntVar,
        from_: int,
        to: int,
        command: Optional[Callable[[], None]],
    ) -> None:
        frame = tk.Frame(parent)
        frame.pack(pady=6)
        tk.Label(frame, text=label).pack(pady=(0, 6))

        def _on_change(_value: str) -> None:
            if command:
                command()

        tk.Scale(
            frame,
            from_=from_,
            to=to,
            orient=tk.VERTICAL,
            showvalue=False,
            variable=var,
            command=_on_change,
            length=160,
        ).pack()

    def _add_spinbox(
        self,
        parent: tk.Frame,
        label: str,
        var: tk.IntVar,
        from_: int,
        to: int,
        command: Optional[Callable[[], None]],
    ) -> None:
        frame = tk.Frame(parent)
        frame.pack(pady=6)
        tk.Label(frame, text=label).pack(pady=(0, 6))
        spinbox = tk.Spinbox(
            frame,
            from_=from_,
            to=to,
            textvariable=var,
            width=6,
            command=command,
        )
        spinbox.bind(
            "<KeyRelease>",
            lambda _event: command() if command else None,
        )
        spinbox.pack()

    def _update_button_states(self, initial: bool = False) -> None:
        if initial:
            self.btn_negative.config(state=tk.DISABLED)
            self.btn_confirm.config(state=tk.DISABLED)
            self.btn_add_point.config(state=tk.DISABLED)
            self.btn_remove_point.config(state=tk.DISABLED)
            return

        if self.state.has_plan:
            self.btn_negative.config(state=tk.NORMAL)
            self.btn_confirm.config(state=tk.NORMAL)
            self.btn_add_point.config(state=tk.NORMAL)
            self.btn_remove_point.config(state=tk.NORMAL)
        else:
            self.btn_negative.config(state=tk.DISABLED)
            self.btn_confirm.config(state=tk.DISABLED)
            self.btn_add_point.config(state=tk.DISABLED)
            self.btn_remove_point.config(state=tk.DISABLED)

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

    def toggle_positive(self) -> None:
        self._set_mode("add_positive")
        if self.state.mode == "add_positive":
            self.btn_negative.config(relief=tk.RAISED)
            self.btn_add_point.config(relief=tk.RAISED)
            self.btn_remove_point.config(relief=tk.RAISED)

    def toggle_negative(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("add_negative")
        if self.state.mode == "add_negative":
            self.btn_positive.config(relief=tk.RAISED)
            self.btn_add_point.config(relief=tk.RAISED)
            self.btn_remove_point.config(relief=tk.RAISED)

    def toggle_add_point(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("add_point")
        if self.state.mode == "add_point":
            self.btn_positive.config(relief=tk.RAISED)
            self.btn_negative.config(relief=tk.RAISED)
            self.btn_remove_point.config(relief=tk.RAISED)

    def toggle_remove_point(self) -> None:
        if not self.state.has_plan:
            messagebox.showinfo("提示", "请先生成手术方案")
            return
        self._set_mode("remove_point")
        if self.state.mode == "remove_point":
            self.btn_positive.config(relief=tk.RAISED)
            self.btn_negative.config(relief=tk.RAISED)
            self.btn_add_point.config(relief=tk.RAISED)

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
        self.area_mask = self.pipeline.run_presegmentation(image)
        self.state = AppState(clicks=[])
        self.plan = None
        self.manual_centers = []
        self.hidden_centers = set()
        self.last_mouse_pos = None
        self.state.has_plan = False
        self.state.mode = "none"
        self._update_button_states()
        self._refresh_toggle_buttons()
        self._render_overlay(self.original_pil)

    def _render_overlay(self, overlay: Image.Image) -> None:
        display_overlay = overlay.resize(self.display_size, Image.BILINEAR)
        self.display_image = ImageTk.PhotoImage(display_overlay)
        self.canvas.delete("all")
        self.canvas.config(width=self.display_size[0], height=self.display_size[1])
        self.canvas.create_image(0, 0, image=self.display_image, anchor=tk.NW)
        self.canvas.configure(scrollregion=(0, 0, self.display_size[0], self.display_size[1]))

    def _render_current_plan(self) -> None:
        if self.plan and self.original_pil:
            overlay = self._build_plan_overlay()
            self._render_overlay(overlay)
        elif self.original_pil:
            self._render_overlay(self.original_pil)

    def _build_plan_overlay(self) -> Image.Image:
        base = self.original_pil.convert("RGBA")
        draw = ImageDraw.Draw(base, "RGBA")
        radius = self._spot_radius()
        centers = self._visible_centers(self.plan.circle_centers if self.plan else [])
        for x, y in centers:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(0, 0, 255, 255),
                width=2,
            )
        for x, y in self.manual_centers:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(0, 0, 255, 255),
                width=2,
            )
        return base.convert("RGB")

    def _canvas_coords(self, event) -> Tuple[float, float]:
        return (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _display_to_original(self, point: Tuple[float, float]) -> Tuple[int, int]:
        return (int(point[0] * self.display_scale_x), int(point[1] * self.display_scale_y))

    def on_canvas_press(self, event) -> None:
        self.on_canvas_click(event)

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
            display_mask = resize_mask(result.mask, (self.original_pil.width, self.original_pil.height))
            display_mask = self._postprocess_first_mask(display_mask)
            self.spot_diameter_var.set(DEFAULT_SPOT_DIAMETER)
            self.spot_distance_var.set(DEFAULT_SPOT_DISTANCE)
            plan = plan_surgery(
                self.original_pil,
                display_mask,
                self.area_mask,
                spot_diameter=self.spot_diameter_var.get(),
                spot_distance=self.spot_distance_var.get(),
                max_layers=self.spot_layers_var.get(),
            )
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
        LOGGER.info(
            "iteration_inputs user_clicks=%d total_clicks=%d",
            len(self.state.clicks),
            len(self.state.clicks),
        )
        result = self.pipeline.run_iteration(
            self.current_image,
            self.state.clicks,
            self.state.current_logits,
        )
        display_mask = resize_mask(result.mask, (self.original_pil.width, self.original_pil.height))
        display_mask = self._postprocess_mask(display_mask)
        plan = plan_surgery(
            self.original_pil,
            display_mask,
            self.area_mask,
            spot_diameter=self.spot_diameter_var.get(),
            spot_distance=self.spot_distance_var.get(),
            max_layers=self.spot_layers_var.get(),
        )
        self._apply_plan(result, plan, display_mask)
        self._start_preview_loop()

    def _apply_plan(self, result, plan: PlanResult, display_mask: np.ndarray) -> None:
        self.state.current_mask = display_mask
        self.state.current_logits = result.logits
        self.plan = plan
        self.hidden_centers = {center for center in self.hidden_centers if center in plan.circle_centers}
        self._filter_manual_centers()
        self._render_current_plan()

    def clear_plan(self) -> None:
        if not self.original_pil:
            return
        self.state = AppState(clicks=[])
        self.plan = None
        self.manual_centers = []
        self.hidden_centers = set()
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
            display_mask = resize_mask(result.mask, (self.original_pil.width, self.original_pil.height))
            display_mask = self._postprocess_first_mask(display_mask)
        else:
            preview_clicks = list(self.state.clicks) + [preview_click]
            result = self.pipeline.run_iteration(
                self.current_image,
                preview_clicks,
                self.state.current_logits,
            )
            display_mask = resize_mask(result.mask, (self.original_pil.width, self.original_pil.height))
            display_mask = self._postprocess_mask(display_mask)
        preview_plan = plan_surgery(
            self.original_pil,
            display_mask,
            self.area_mask,
            spot_diameter=self.spot_diameter_var.get(),
            spot_distance=self.spot_distance_var.get(),
            max_layers=self.spot_layers_var.get(),
        )
        overlay = self._build_preview_overlay(preview_plan)
        self._render_overlay(overlay)
        self.preview_job = self.root.after(200, self._preview_step)

    def _build_preview_overlay(self, preview_plan: PlanResult) -> Image.Image:
        base = self.original_pil.convert("RGBA")
        draw = ImageDraw.Draw(base, "RGBA")
        radius = self._spot_radius()
        current_centers = set(self._visible_centers(self.plan.circle_centers)) if self.plan else set()
        preview_centers = set(self._visible_centers(preview_plan.circle_centers))
        stay_centers = current_centers & preview_centers
        add_centers = preview_centers - current_centers
        remove_centers = current_centers - preview_centers

        for x, y in stay_centers:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(0, 0, 255, 255),
                width=2,
            )
        for x, y in add_centers:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(0, 255, 0, 200),
                fill=(0, 255, 0, 128),
                width=2,
            )
        for x, y in remove_centers:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(255, 0, 0, 200),
                fill=(255, 0, 0, 128),
                width=2,
            )
        for x, y in self.manual_centers:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(0, 0, 255, 255),
                width=2,
            )
        return base.convert("RGB")

    def _apply_point_modification(self, point: Tuple[int, int], add: bool) -> None:
        if not self.plan or self.original_pil is None:
            return
        radius = self._spot_radius()
        min_distance = self._min_center_distance()
        plan_centers = self._visible_centers(self.plan.circle_centers)
        if add:
            if not self._is_inside_area_mask(point):
                return
            nearest_distance = self._distance_to_nearest_center(point, plan_centers + self.manual_centers)
            if nearest_distance is not None and nearest_distance < min_distance:
                return
            self.manual_centers.append(point)
        else:
            removed = self._remove_circle_at_point(point, radius)
            if not removed:
                return
        self._render_current_plan()

    def _postprocess_first_mask(self, mask: np.ndarray) -> np.ndarray:
        cleaned = remove_small_components(mask, min_size=200)
        filled = fill_small_holes(cleaned, area_threshold=200)
        return filled

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        cleaned = remove_small_components(mask, min_size=200)
        filled = fill_small_holes(cleaned, area_threshold=200)
        return filled

    def _spot_radius(self) -> int:
        return max(int(round(self.spot_diameter_var.get() / 2)), 1)

    def _min_center_distance(self) -> int:
        return max(self.spot_diameter_var.get() + self.spot_distance_var.get(), 1)

    def _is_inside_area_mask(self, point: Tuple[int, int]) -> bool:
        if self.area_mask is None:
            return True
        x, y = point
        if y < 0 or x < 0 or y >= self.area_mask.shape[0] or x >= self.area_mask.shape[1]:
            return False
        return self.area_mask[y, x] > 0

    def _filter_manual_centers(self) -> None:
        if not self.plan:
            return
        filtered: List[Tuple[int, int]] = []
        min_distance = self._min_center_distance()
        plan_centers = self._visible_centers(self.plan.circle_centers)
        for center in self.manual_centers:
            if not self._is_inside_area_mask(center):
                continue
            nearest_distance = self._distance_to_nearest_center(center, plan_centers + filtered)
            if nearest_distance is not None and nearest_distance < min_distance:
                continue
            filtered.append(center)
        self.manual_centers = filtered

    def _on_spot_params_change(self) -> None:
        if not self.state.has_plan or self.original_pil is None or self.state.current_mask is None:
            return
        plan = plan_surgery(
            self.original_pil,
            self.state.current_mask,
            self.area_mask,
            spot_diameter=self.spot_diameter_var.get(),
            spot_distance=self.spot_distance_var.get(),
            max_layers=self.spot_layers_var.get(),
        )
        self.plan = plan
        self.hidden_centers = {center for center in self.hidden_centers if center in plan.circle_centers}
        self._filter_manual_centers()
        self._render_current_plan()

    def _visible_centers(self, centers: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not self.hidden_centers:
            return list(centers)
        return [center for center in centers if center not in self.hidden_centers]

    def _remove_circle_at_point(self, point: Tuple[int, int], radius: int) -> bool:
        for idx, center in enumerate(self.manual_centers):
            if self._distance(point, center) <= radius:
                self.manual_centers.pop(idx)
                return True
        if self.plan:
            for center in self._visible_centers(self.plan.circle_centers):
                if self._distance(point, center) <= radius:
                    self.hidden_centers.add(center)
                    return True
        return False

    def _distance(self, point: Tuple[int, int], center: Tuple[int, int]) -> float:
        return float(((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5)

    def _distance_to_nearest_center(
        self, point: Tuple[int, int], centers: List[Tuple[int, int]]
    ) -> Optional[float]:
        if not centers:
            return None
        distances = [self._distance(point, center) for center in centers]
        return min(distances)
