"""Interactive prediction script using model.py only."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from model import build_sam2_video_predictor_npz


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_images(image_dir: Path) -> List[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in supported]
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return paths


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def generate_hex_centers(mask: np.ndarray, spacing: float = 24.0) -> List[Tuple[int, int]]:
    height, width = mask.shape
    centers: List[Tuple[int, int]] = []
    row_spacing = spacing * math.sqrt(3) / 2
    y = 0.0
    row = 0
    while y < height:
        x_offset = 0.0 if row % 2 == 0 else spacing / 2
        x = x_offset
        while x < width:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < width and 0 <= yi < height and mask[yi, xi]:
                centers.append((xi, yi))
            x += spacing
        y += row_spacing
        row += 1
    return centers


def draw_circles(image: Image.Image, mask: np.ndarray) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    radius = 8
    for x, y in generate_hex_centers(mask):
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=(0, 0, 255),
            width=2,
        )
    return output


def run_interactive(model, image_path: Path) -> bool:
    original = Image.open(image_path).convert("RGB")
    width, height = original.size
    image_tensor = preprocess_image(original, model.image_size)
    inference_state = model.init_state(
        images=[image_tensor],
        video_height=height,
        video_width=width,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )

    click_points: List[Tuple[float, float]] = []
    click_labels: List[int] = []
    click_history: List[dict] = []

    fig, ax = plt.subplots()
    ax.axis("off")
    image_artist = ax.imshow(original)

    should_continue = {"next": False, "quit": False}

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button not in (1, 3):
            return
        label = 1 if event.button == 1 else 0
        click_points.append((event.xdata, event.ydata))
        click_labels.append(label)
        points = np.array([[event.xdata, event.ydata]], dtype=np.float32)
        labels = np.array([label], dtype=np.int32)
        _, _, masks = model.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
            clear_old_points=False,
            normalize_coords=True,
        )
        mask = masks[0].detach().cpu().numpy() > 0.0
        click_history.append({
            "points": list(click_points),
            "labels": list(click_labels),
            "mask": mask,
        })
        overlay = draw_circles(original, mask)
        image_artist.set_data(overlay)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in {"enter", "n"}:
            should_continue["next"] = True
            plt.close(fig)
        if event.key in {"escape", "q"}:
            should_continue["quit"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return not should_continue["quit"]


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 interactive predictor")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--device", default=None, help="Device to run inference on")
    args = parser.parse_args()

    model = build_sam2_video_predictor_npz(
        ckpt_path=args.checkpoint,
        device=args.device,
        mode="eval",
        apply_postprocessing=False,
    )

    image_paths = load_images(Path(args.image_dir))
    for image_path in image_paths:
        print(f"Processing {image_path.name} (press 'n' to advance, 'q' to quit)")
        if not run_interactive(model, image_path):
            break


if __name__ == "__main__":
    main()
