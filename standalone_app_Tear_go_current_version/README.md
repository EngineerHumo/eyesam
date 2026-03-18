# standalone_app_Tear_go_current_version

该目录为 `standalone_app_Tear` 的 Go 后端版本，移除了桌面 UI，但保留了预分割、首轮/迭代 ONNX 推理、掩码处理与手术点位规划逻辑。

## 提供能力

- `RunPreSegmentation`：对应 Python `SurgicalPipeline.run_presegmentation`。
- `RunFirst`：对应首轮 `first.onnx` 推理。
- `RunIteration`：对应 Python `SurgicalPipeline.run_iteration`。
- `PlanSurgery`：对应 Python `planner.plan_surgery`，保留多连通域、逐层外扩、避开 FAZ、去重等逻辑。

## 模型目录

```text
standalone_app_Tear_go_current_version/
  onnx/
    pre.onnx
    first.onnx
    iteration.onnx
```

## 使用示例

```go
svc, err := standalone_app_Tear_go_current_version.NewService("standalone_app_Tear_go_current_version/onnx")
img, err := standalone_app_Tear_go_current_version.LoadImage("demo.png")
pre, err := svc.RunPreSegmentation(img)

modelImg := standalone_app_Tear_go_current_version.PrepareModelImage(img, 1024, 1024)
first, err := svc.RunFirst(modelImg, []standalone_app_Tear_go_current_version.Click{{X: 300, Y: 240, Label: 1}})
iter, err := svc.RunIteration(modelImg, []standalone_app_Tear_go_current_version.Click{{X: 320, Y: 260, Label: 1}}, first.Logits)
plan := svc.PlanSurgery(standalone_app_Tear_go_current_version.PlanRequest{
    Image:        img,
    Mask:         standalone_app_Tear_go_current_version.ResizeMaskToImage(iter.Mask, img),
    AreaMask:     &pre.AreaMask,
    FAZMask:      &pre.FAZMask,
    SpotDiameter: 15,
    SpotDistance: 7,
    MaxLayers:    3,
})
_ = plan
```
