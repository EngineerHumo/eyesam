# standalone_app_v3_5_go

该目录为 `standalone_app_v3_5` 的 Go 语言重构版本，已补齐推理、预分割、管线与 UI 的完整实现骨架。

## 功能概览

- 使用 `onnxruntime_go`（见 `../onnxruntime_go-master`）加载 ONNX 模型，支持预分割、初始推理与迭代推理。
- 使用 Fyne 构建桌面 GUI，提供 AI 工具 + 传统工具（点位/区域）编辑、区域方案切换与参数调整。
- 形态学处理、连通域分析、距离变换等全部以 Go 自研方式实现（见 `internal/utils`）。

## 运行说明

1. 准备 ONNX 模型：

```
standalone_app_v3_5_go/
  onnx/
    pre.onnx
    first.onnx
    iteration.onnx
```

2. 配置 ONNX Runtime 动态库路径（可选）：

```
export ORT_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so
```

3. 启动应用：

```
go run ./cmd/eyesam_app -base .
```

### CUDA 选项

如需使用 CUDA，请使用带 CUDA 的 onnxruntime 动态库，并设置：

```
export EYESAM_ONNX_USE_CUDA=true
```

## 注意事项

- 由于依赖 `fyne.io/fyne/v2` 与 `golang.org/x/image`，需要在具备网络访问或本地代理的环境中执行 `go mod tidy`。
- `onnxruntime_go` 使用本地 replace 指向 `../onnxruntime_go-master`，请确保该目录存在。
