import logging
import os
import sys
import tkinter as tk

from pipeline import SurgicalPipeline
from ui import MainWindow


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def ensure_onnx_files(onnx_dir: str) -> None:
    required = [
        "faz.onnx",
        "area.onnx",
        "first.onnx",
        "iteration.onnx",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(onnx_dir, name))]
    if missing:
        logging.error("缺少 ONNX 文件: %s", ", ".join(missing))
        logging.error("请将模型放置到 %s", onnx_dir)
        sys.exit(1)


def main() -> None:
    setup_logging()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_dir = os.path.join(base_dir, "onnx")
    ensure_onnx_files(onnx_dir)

    pipeline = SurgicalPipeline(onnx_dir)
    root = tk.Tk()
    MainWindow(root, pipeline)
    root.mainloop()


if __name__ == "__main__":
    main()
