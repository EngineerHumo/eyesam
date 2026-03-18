# standalone_app_Tear_go_current_version 接口说明

## 核心接口

### 1. `NewService(onnxDir string)`
创建后端服务实例，要求目录内存在 `pre.onnx / first.onnx / iteration.onnx`。

### 2. `RunPreSegmentation(img image.Image)`
输出：
- `AreaMask`
- `FAZMask`

### 3. `RunFirst(modelImg utils.ModelImage, clicks []Click)`
执行首轮分割，返回：
- `Mask`
- `Logits`

### 4. `RunIteration(modelImg utils.ModelImage, clicks []Click, prevLogits utils.FloatMask)`
执行迭代分割，返回：
- `Mask`
- `Logits`

### 5. `PlanSurgery(req PlanRequest)`
输入：
- 原图
- 当前 mask
- `area_mask`（可选）
- `faz_mask`（可选）
- 光斑直径/距离/层数

输出：
- `Overlay`
- `CurvePoints`
- `CircleCenters`

## 坐标约定

- 点击点使用原图坐标系。
- `RunFirst/RunIteration` 的 `Mask` 与 `Logits` 保持模型输出分辨率。
- 若需要与原图对齐，请使用 `ResizeMaskToImage`。
