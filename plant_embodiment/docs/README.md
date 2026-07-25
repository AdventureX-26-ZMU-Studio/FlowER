# Plant Embodiment — dimos 集成

> 位置: `dimos-advx-zmu/plant_embodiment/`

```
plant_embodiment/
├── docs/
│   ├── SKILL.md        # DCW Vision Service 调用规范
│   └── README.md       # 本文件
├── scripts/
│   ├── depth_cluster_engine.py   # 深度聚类引擎
│   ├── depth_obstacle_bridge.py  # depth → Detection3D 桥接
│   ├── dimos_dcw_adapter.py      # dimos Module 适配器
│   ├── patch_vision_server.py    # vision_server 补丁指南
│   └── test_depth_clustering.py  # 深度聚类测试
└── configs/
    (运行时配置)
```

## 快速启动

```bash
cd /home/asus/dimos-advx-zmu
sudo -u asus .venv-mediapipe/bin/python \
  -m dimos.hardware.sensors.camera.orbbec_dcw.vision_server \
  --host 0.0.0.0 --port 8890
```

## 关键路径

| 组件 | 路径 |
|------|------|
| Vision Server | `dimos/hardware/sensors/camera/orbbec_dcw/vision_server.py` |
| Depth 聚类引擎 | `plant_embodiment/scripts/depth_cluster_engine.py` |
| Face Follow | `dimos/hardware/sensors/camera/orbbec_dcw/face_follow.py` |
| DCW Camera Module | `dimos/hardware/sensors/camera/orbbec_dcw/camera.py` |

## 端点

`http://<host>:8890/`

| 端点 | 说明 |
|------|------|
| `/` | Web 仪表盘 |
| `/status` | JSON 完整状态（含障碍物） |
| `/s/color` | 彩色 MJPEG |
| `/s/depth` | 深度 MJPEG |
| `/obstacles` | 3D 障碍物列表 |
| `/depth_raw` | 原始深度 uint16 二进制 |
| `/tracking` | POST Face Follow 控制 |
| `/healthz` | 健康检查 |
