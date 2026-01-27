# standalone_app_v3_5_go

该目录为 `standalone_app_v3_5` 的 Go 语言重构版本骨架，用于建立 Go 工程结构并保留主要模块边界。

## 现状说明

- 已整理为 Go module，并拆分为 `cmd/` + `internal/` 结构。
- 推理、预分割、UI 等模块仅提供接口与占位实现（返回 `not implemented`），方便后续逐步补全。
- 规划逻辑保留了环形点位与基本绘制逻辑，但图像形态学处理未移植。

## 目录结构

- `cmd/eyesam_app`: CLI 入口，负责校验 ONNX 模型与启动 UI。
- `internal/inference`: 推理接口与 ONNX 模型占位实现。
- `internal/preprocess`: 预分割占位实现。
- `internal/pipeline`: 推理流程骨架。
- `internal/planner`: 规划逻辑（环形点位 + 叠加绘制）。
- `internal/utils`: 数据结构、图像加载与简单 resize。

## 后续建议

1. 引入 Go 版 ONNX Runtime 绑定后补全推理逻辑。
2. 用 GoCV 或自实现替换原 Python 中的形态学处理与连通域分析。
3. 将 Tkinter UI 替换为 Go GUI（如 Fyne/Walk/Wails 等）。

