# DCW Vision Service

> Orbbec DaBai DCW 相机中央数据服务。
> 
> 只有一个进程可以占用相机硬件，所有消费者从此服务获取数据。
> 包括：RGB 图像、深度数据、障碍物检测、人脸/手势信息。

---

## 快速启动

```bash
# GX10 上 (asus 用户)
export LD_LIBRARY_PATH=/home/asus/orbbec_sdk_libs
cd ~/dimos-advx-zmu
.venv-mediapipe/bin/python \
  -m dimos.hardware.sensors.camera.orbbec_dcw.vision_server \
  --host 0.0.0.0 --port 8890
```

默认端口 **8890**，可通过 `--port` 修改。

---

## 端点一览

| 端点 | 方法 | 内容 | 频率 |
|------|------|------|------|
| `/` | GET | Web 仪表盘（HTML） | — |
| `/healthz` | GET | 健康检查 `{"status":"ok"}` | 按需 |
| `/status` | GET | JSON 完整状态 | 5次/秒 |
| `/s/color` | GET | 标注后的彩色 MJPEG 流 | ~15fps |
| `/s/depth` | GET | 深度伪彩色 MJPEG 流 | ~15fps |
| `/depth_raw` | GET | 原始深度数据（uint16×640×360） | 按需 |
| `/obstacles` | GET | JSON 3D 障碍物列表 | 10次/秒 |
| `/tracking` | POST | 启用/禁用 Face Follow | 按需 |

---

## `/status` — JSON 状态

```json
{
  "camera_connected": true,
  "mediapipe_ready": true,
  "fps": 15.0,
  "frame": 12345,

  "face_count": 1,
  "face_center_x": 0.45,
  "face_center_y": 0.52,
  "expression": "happy",
  "gesture": "palm",
  "flipped": false,
  "face_depth_mm": 845.0,

  "depth_aligned": true,
  "depth_center_mm": 1200.5,

  "obstacle_count": 3,
  "obstacles": [
    {"id": "depth_123_0", "center_m": [0.32, -0.15, 1.20],
     "size_m": [0.25, 0.35, 0.18], "volume_m3": 0.016,
     "first_seen_s": 12.5, "last_seen_s": 0.3},
    {"id": "depth_123_1", "center_m": [-0.18, 0.05, 0.85],
     "size_m": [0.15, 0.20, 0.12], "volume_m3": 0.004,
     "first_seen_s": 8.2, "last_seen_s": 0.1}
  ],

  "tracking_enabled": false,
  "tracking_state": "disabled"
}
```

### 字段说明

| 路径 | 类型 | 说明 |
|------|------|------|
| `camera_connected` | bool | 相机是否在线 |
| `fps` | float | 当前帧率 |
| `face_count` | int | 检测到的人脸数 |
| `face_center_x/y` | float \| null | 人脸在画面中的归一化位置 (0~1) |
| `expression` | string | 表情: `happy`/`sad`/`angry`/`surprised`/`neutral` |
| `gesture` | string | 手势: `fist`/`palm`/`point`/`peace`/`hand_N` |
| `flipped` | bool | 画面是否已自动翻转 |
| `face_depth_mm` | float | 人脸深度距离 (mm) |
| `depth_aligned` | bool | D2C 对齐是否启用 |
| `depth_center_mm` | float | 画面中心深度 (mm) |
| `obstacle_count` | int | 当前检测到的障碍物数 |
| `obstacles[]` | array | 障碍物列表 |
| `obstacles[].center_m` | [x,y,z] | 障碍物在相机坐标系的位置 (米) |
| `obstacles[].size_m` | [w,h,d] | 障碍物包围盒尺寸 (米) |
| `obstacles[].volume_m3` | float | 体积 |
| `obstacles[].first_seen_s` | float | 首次检测到距今秒数 |
| `obstacles[].last_seen_s` | float | 末次检测到距今秒数 |
| `tracking_enabled` | bool | Face Follow 是否启用 |
| `tracking_state` | string | `disabled`/`searching`/`acquiring`/`tracking`/`centered` |

---

## `/s/color` — 彩色 MJPEG 流

- 格式: `multipart/x-mixed-replace; boundary=frame`
- 分辨率: 640×360 (MJPG 解码后 BGR)
- 标注: 人脸框、表情、手势、检测到的障碍物框

### 消费方式 (Python)

```python
import cv2
import requests

stream = requests.get("http://localhost:8890/s/color", stream=True)
bytes_buffer = b""
for chunk in stream.iter_content(1024):
    bytes_buffer += chunk
    a = bytes_buffer.find(b"\xff\xd8")
    b = bytes_buffer.find(b"\xff\xd9")
    if a != -1 and b != -1 and b > a:
        jpg = bytes_buffer[a:b+2]
        bytes_buffer = bytes_buffer[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        # frame.shape → (360, 640, 3)
```

---

## `/s/depth` — 深度伪彩色 MJPEG 流

- 格式: `multipart/x-mixed-replace`
- 分辨率: 640×360
- 编码: 伪彩色 (Turbo colormap)，带中心十字标记
- 仅用于**可视化**，不用于精确分析（有损压缩）

---

## `/depth_raw` — 原始深度数据

- 格式: `application/octet-stream`
- 分辨率: 640×360, 每像素 2 字节 (uint16, little-endian)
- 单位: **毫米 (mm)**
- 尺寸: 640×360×2 = 460,800 字节

### 消费方式

```python
import numpy as np
import requests

resp = requests.get("http://localhost:8890/depth_raw")
data = np.frombuffer(resp.content, dtype=np.uint16).reshape(360, 640)
# data[y, x] = 深度值 (mm), 0 表示无效
```

**注意**: 请求 `/depth_raw` 会暂停 MJPEG 流一帧。
不要频繁轮询（建议 ≤ 5次/秒）。

---

## `/obstacles` — 障碍物检测

- 返回 JSON 数组，结构与 `/status` 中的 `obstacles` 字段相同
- 频率: ~10次/秒
- 坐标系: `dcw_depth_optical_frame`（相机光心坐标系，米制）

### 坐标系约定

```
相机光心 (dcw_depth_optical_frame):
  X → 向右
  Y → 向下（符合图像坐标）
  Z → 向前（深度方向）

因此: 障碍物 center_m = (X右, Y下, Z前)
```

### 消费方式

```python
import requests

obs = requests.get("http://localhost:8890/obstacles").json()
for o in obs:
    cx, cy, cz = o["center_m"]
    w, h, d = o["size_m"]
    print(f"  {o['id']}: ({cx:.2f}, {cy:.2f}, {cz:.2f}) {w:.2f}×{h:.2f}×{d:.2f}m")
```

---

## `/tracking` — Face Follow 控制

启用或禁用 Face Follow 控制器。

### 请求

```json
POST /tracking
Content-Type: application/json

{"enabled": true}
```

### 响应

```json
{
  "enabled": true,
  "state": "searching"
}
```

`state` 取值: `disabled` / `searching` / `acquiring` / `tracking` / `centered` / `timed_out`

---

## `/healthz` — 健康检查

```json
{"status": "ok", "uptime": 1234.5, "frame": 12345}
```

---

## 障碍物检测算法说明

**深度图分块聚类法** (无需 ML 模型):

```
640×360 depth → 降采样为 320×180 → 分 12×20 网格 (每格 16×9 像素)
  → 每格计算中位深度
  → 相邻格深度差 < 15cm 则合并为同一物体
  → 每个物体计算 3D 包围盒 (利用相机内参投影)
  → 过滤体积 < 500cm³ 的噪声
  → 输出 top-N 障碍物
```

**Spatial Memory**: 每个障碍物带 `first_seen_s` / `last_seen_s` 字段。
消费者可根据此字段决定保留多久（例如：超过 5 秒未更新的障碍物仍保留在仿真中）。

**底座红区**: 机械臂底座 XY 平面为固定障碍物，不通过深度检测，需在仿真/规划器中静态配置。

---

## 从其他服务消费 (推荐方式)

### dimos Module 消费者

```python
class ObstacleConsumer(Module):
    """定期轮询 vision service 的障碍物数据"""
    
    def __init__(self):
        self._session = requests.Session()
        self._last_obs = []
        self._world_monitor = WorldObstacleMonitor(...)
    
    def _poll(self):
        obs = self._session.get("http://localhost:8890/obstacles").json()
        detections = []
        for o in obs:
            det = Detection3D()
            det.id = o["id"]
            det.bbox.center.position = Point(*o["center_m"])
            det.bbox.size = Vector3(*o["size_m"])
            detections.append(det)
        self._world_monitor.on_detections(detections)
```

### 外部客户端

```python
# 获取深度数据做自定义分析
depth = get_raw_depth()  # 从 /depth_raw
frame = get_color_frame()  # 从 /s/color
status = get_status()  # 从 /status

# 自己的处理逻辑
```

---

## 部署规范

```bash
# 启动 (独占相机)
python -m dimos.hardware.sensors.camera.orbbec_dcw.vision_server \
  --host 0.0.0.0 --port 8890

# 确保没有其他进程占用相机
# 如有冲突，先:
sudo pkill -f vision_server
sleep 2  # 等待 USB 释放

# 用 systemd 管理 (推荐)
systemctl --user start dcw-vision.service
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1 | 2026-07-24 | 初始规范，RGB + Depth + Face + tracking |
| v1.1 | 2026-07-24 | 增加障碍物检测 `/obstacles`，`/depth_raw`，Spatial Memory 字段 |
