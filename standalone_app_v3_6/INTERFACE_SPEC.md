# standalone_app_v3_6 接口说明

> 所有接口以 `work_dir` 为工作区；状态与中间结果全部落盘。

## 统一约定

### 坐标与分辨率
- **输入点击点/多边形坐标**均使用**原图坐标系**（即读取图像后的宽高）。  
- `mask` 输出为原图分辨率的二值 `.npy`。  
- `logits` 输出为模型原始输出分辨率的 `.npy`，后续迭代直接读取该文件作为 `mask_input`。  

### 点击点 label 约定
- `1`：正向点击（新增/保留区域）。  
- `0`：负向点击（排除区域）。  
- `-1`：占位/忽略点（模型收到但不产生正负效果），与原工程语义保持一致。  

### JSON 格式示例
**单个点击点（用于接口 4/5）：**
```json
{"x": 512, "y": 384}
```

**点击点序列（用于接口 1/3）：**
```json
{
  "clicks": [
    {"x": 420, "y": 300, "label": 1},
    {"x": 380, "y": 260, "label": 0},
    {"x": 512, "y": 384, "label": -1}
  ]
}
```

**多边形坐标序列（用于接口 6/7）：**
```json
{
  "points": [
    {"x": 200, "y": 200},
    {"x": 350, "y": 220},
    {"x": 360, "y": 360},
    {"x": 210, "y": 340}
  ]
}
```

---

## 1) 初始规划

**输入**
- `image_path`：图像文件路径（支持 png/jpg/tif 等）。
- `work_dir`：工作目录。

**输出（落盘）**
- `scheme_{i}_mask.npy`：若干初始 mask。
- `scheme_{i}_logits.npy`：每个 mask 对应的 logits。
- `scheme_{i}_clicks.json`：每个 mask 对应的点击点。
- `planning_center.json`：规划中心点（FAZ 最大连通区域最大内接圆圆心）。
- `area_mask.npy`：手术区域约束 mask（用于后续 area 约束）。
- `faz_mask.npy`：FAZ mask。
- `state.json`：图像路径与分辨率。

**示例**
```python
from standalone_app_v3_6 import SurgicalPlannerAPI

api = SurgicalPlannerAPI("standalone_app_v3_6/onnx")
api.initial_plan("demo.png", "output/run1")
```

---

## 2) 选取初始方案

**输入**
- `scheme_index`：要保留的 mask 编号（如 `0`）。

**输出（落盘）**
- `current_mask.npy`
- `current_logits.npy`
- `current_clicks.json`
- 删除其他 `scheme_*` 文件，仅保留所选方案文件。

**示例**
```python
api.select_initial_scheme("output/run1", scheme_index=0)
```

---

## 3) 正向/负向点击点（迭代）

**输入**
- `click_json_path`：包含正/负点击的 JSON 文件（使用 `label=1/0/-1`）。

**输出（落盘）**
- `current_mask.npy`
- `current_logits.npy`
- `current_clicks.json`（合并历史点击 + 新点击）

**示例**
```python
api.iterate_with_clicks("output/run1", "new_clicks.json")
```

---

## 4) 添加点击点（修改 logits）

**输入**
- `click_json_path`：单个点击点 JSON。

**输出（落盘）**
- 更新 `current_logits.npy`，并将点击点附近 **30×30** 像素置为 `1`。

**示例**
```python
api.add_click_point("output/run1", "add_point.json")
```

---

## 5) 删除点击点（修改 logits）

**输入**
- `click_json_path`：单个点击点 JSON。

**输出（落盘）**
- 更新 `current_logits.npy`，并将点击点附近 **30×30** 像素置为 `0`。

**示例**
```python
api.remove_click_point("output/run1", "remove_point.json")
```

---

## 6) 添加手术区域（修改 logits）

**输入**
- `polygon_json_path`：多边形点序列 JSON（闭合多边形）。

**输出（落盘）**
- 更新 `current_logits.npy`，并将多边形区域置为 `1`。

**示例**
```python
api.add_area("output/run1", "add_area.json")
```

---

## 7) 删除手术区域（修改 logits）

**输入**
- `polygon_json_path`：多边形点序列 JSON（闭合多边形）。

**输出（落盘）**
- 更新 `current_logits.npy`，并将多边形区域置为 `0`。

**示例**
```python
api.remove_area("output/run1", "remove_area.json")
```

---

## 8) 清空当前手术方案

**输入**
- 无。

**输出（落盘）**
- 删除 `current_mask.npy`、`current_logits.npy`、`current_clicks.json`。

**示例**
```python
api.clear_current("output/run1")
```
