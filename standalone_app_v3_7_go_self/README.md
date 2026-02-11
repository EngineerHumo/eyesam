# standalone_app_v3_7_go

该目录为 `standalone_app_v3_7` 的 Go 语言重构版本，去掉 UI，保留完整的推理、预分割、形态学处理与连通域分析逻辑，供外部程序直接调用。

## 功能概览

- 使用 `onnxruntime_go`（见 `../onnxruntime_go-master`）加载 `pre.onnx / first.onnx / iteration.onnx`。
- 保留原 Python 版本的预分割、后处理、连通域分析与距离变换逻辑（Go 自研实现）。
- 按 `standalone_app_v3_7` 接口落盘保存 `mask/logits/clicks` 与 `planning_center.json`。

## 目录结构

```
standalone_app_v3_7_go/
  api.go                   # Go API 入口
  internal/
    inference/             # ONNX 推理
    preprocess/            # 预分割
    pipeline/              # 初始规划 + 迭代
    planner/               # FAZ 中心点计算
    npy/                   # .npy 读写
    utils/                 # 形态学 + 连通域等工具
  onnx/
    pre.onnx
    first.onnx
    iteration.onnx
```

## 运行准备

1. 放置 ONNX 模型：

```
standalone_app_v3_7_go/onnx/
  pre.onnx
  first.onnx
  iteration.onnx
```

2. 配置 ONNX Runtime 动态库路径（可选）：

```
export ORT_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so
```

3. （可选）启用 CUDA：

```
export EYESAM_ONNX_USE_CUDA=true
```

## 使用方式

参考 `INTERFACE_SPEC.md`，使用 Go API 调用：

```go
import eyesam "eyesam/standalone_app_v3_7_go"

api, err := eyesam.NewSurgicalInterface("standalone_app_v3_7_go/onnx")
if err != nil {
    panic(err)
}
artifacts, err := api.InitialPlan("demo.png", "output/run1")
if err != nil {
    panic(err)
}
```
