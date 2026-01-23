"""Test script for running first.onnx with a single point prompt."""

import argparse
import os
from typing import Tuple

import numpy as np
from PIL import Image

from inference import OnnxModel
from utils import Click, load_image, prepare_image_for_model, resize_mask


def parse_point(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("点坐标格式必须为 x,y")
    try:
        x = int(float(parts[0]))
        y = int(float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("点坐标必须是数字") from exc
    return x, y


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 first.onnx，输入图像与点坐标，输出 mask 图。",
    )
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument(
        "--point",
        required=True,
        type=parse_point,
        help="点击点坐标，格式: x,y (使用原图坐标)",
    )
    parser.add_argument(
        "--output",
        default="first_mask.png",
        help="输出 mask 图片路径",
    )
    parser.add_argument(
        "--onnx",
        default=os.path.join(os.path.dirname(__file__), "onnx", "first.onnx"),
        help="first.onnx 路径（默认使用 standalone_app_v3/onnx/first.onnx）",
    )
    parser.add_argument(
        "--prefer-gpu",
        action="store_true",
        help="优先使用 GPU 运行 onnxruntime",
    )
    return parser


def save_mask(mask: np.ndarray, output_path: str) -> None:
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img).save(output_path)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"未找到 onnx 文件: {args.onnx}")

    image = load_image(args.image)
    model = OnnxModel(args.onnx, prefer_gpu=args.prefer_gpu)

    target_size = model.image_input_size((image.width, image.height))
    model_image = prepare_image_for_model(image, target_size)

    resized_hw = (model_image.resized_np.shape[0], model_image.resized_np.shape[1])
    orig_hw = (model_image.original_np.shape[0], model_image.original_np.shape[1])
    click = Click(x=float(args.point[0]), y=float(args.point[1]), label=1)

    result = model.infer(
        model_image.resized_np,
        resized_hw=resized_hw,
        orig_hw=orig_hw,
        clicks=[click],
    )

    display_mask = resize_mask(result.mask, (image.width, image.height))
    save_mask(display_mask, args.output)
    print(f"Mask 已保存到: {args.output}")


if __name__ == "__main__":
    main()
