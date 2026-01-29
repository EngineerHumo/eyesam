# 接口调用说明（standalone_app_v3_7）

本文件描述 `interface.py` 中 `SurgicalInterface` 的调用方式、输入要求、输出内容与示例。

## 通用约定

- **坐标系**：所有点击点坐标 `(x, y)` 与原始图像像素坐标一致。
- **文件格式**：mask 与 logits 使用 `.npy` 保存，点击点与规划中心点使用 `.json` 保存。
- **索引规则**：初始规划输出的 mask 编号从 `0` 开始递增。
- **输出目录**：每次调用都会将结果写入外部传入的 `output_dir`。

## 1) 初始规划（initial_plan）

### 输入要求
- **image_path**：原始图像路径（任意可被 PIL 打开的格式）。
- **output_dir**：输出目录路径（若不存在会自动创建）。

### 输出内容
- 若干个 **初始规划 mask**（`.npy`）。
- 每个 mask 对应的 **logits**（`.npy`）。
- 每个 mask 对应的 **点击点坐标**（`.json`）。
- **规划中心点**（`planning_center.json`）。

### 输出文件命名规则
- `mask_{index}.npy`
- `logits_{index}.npy`
- `clicks_{index}.json`
- `planning_center.json`

### 点击点文件格式
```json
{
  "points": [
    {"x": 512, "y": 384, "label": 1}
  ]
}
```

### 规划中心点格式
```json
{
  "x": 521,
  "y": 402
}
```

### 调用示例
```python
from standalone_app_v3_7.interface import SurgicalInterface

api = SurgicalInterface("/workspace/eyesam/standalone_app_v3_7/onnx")
artifacts = api.initial_plan(
    image_path="/data/demo.png",
    output_dir="/data/output"
)
print(artifacts.masks)
print(artifacts.planning_center)
```

---

## 2) 选取初始方案（select_initial_scheme）

### 输入要求
- **mask_index**：需要保留的 mask 编号（与 `initial_plan` 输出的编号一致）。

### 输出内容
- **无显式返回值**。指定编号保留，其余 `mask/logits/clicks` 会被删除。

### 调用示例
```python
api.select_initial_scheme(mask_index=0)
```

---

## 3) 正向/负向点击点（apply_clicks）

### 输入要求
- **clicks_json**：点击点坐标文件路径（`.json`）。
- 点击点格式：
```json
{
  "points": [
    {"x": 500, "y": 420, "label": 1},
    {"x": 530, "y": 410, "label": 0}
  ]
}
```

### 输出内容
- **新的 mask**（`.npy`）
- **新的 logits**（`.npy`）
- **更新后的点击点文件**（`.json`）

### 输出文件说明
- 若当前已有选定的初始方案，则覆盖对应的 `mask_{index}.npy / logits_{index}.npy / clicks_{index}.json`。
- 若当前没有 mask（例如刚执行 `clear_current_plan`），则写入：
  - `current_mask.npy`
  - `current_logits.npy`
  - `current_clicks.json`

### 调用示例
```python
result = api.apply_clicks(clicks_json="/data/new_clicks.json")
print(result["mask"], result["logits"], result["clicks"])
```

---

## 4) 正向/负向点击点预览（preview_clicks）

### 输入要求
- **clicks_json**：点击点坐标文件路径（`.json`），格式同上。

### 输出内容
- **临时 mask**（`.npy`），不保存临时 logits，不更新点击点文件。

### 输出文件
- `preview_mask.npy`

### 调用示例
```python
preview_path = api.preview_clicks(clicks_json="/data/new_clicks.json")
print(preview_path)
```

---

## 5) 清空当前手术方案（clear_current_plan）

### 输入要求
- **无**。

### 输出内容
- **无显式返回值**。
- 会删除当前方案对应的 mask/logits/clicks 文件（当前选择的初始方案或 `current_*` 文件）。

### 调用示例
```python
api.clear_current_plan()
```
