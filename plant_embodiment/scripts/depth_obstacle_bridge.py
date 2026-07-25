"""
DCW Depth → Detection3D 障碍物桥接
==================================
把 DaBai DCW depth 流实时转成 Detection3D 消息，
供 WorldObstacleMonitor 消费 → 仿真世界动态更新。

算法: 简单深度聚类（不用 ML）
- 深度图分块，每块计算中位深度
- 相邻块深度相似 → 合并为同一物体
- 每个物体 → bounding box (center + size) → Detection3D

使用方式:
  dimos run dcw-depth-obstacle-bridge
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field as dt_field
from typing import Any

import numpy as np
import cv2

from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.vision_msgs.Detection3D import Detection3D
from dimos.msgs.vision_msgs.BoundingBox3D import BoundingBox3D
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Point import Point
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.geometry_msgs.Quaternion import Quaternion

logger = logging.getLogger(__name__)


# ── Config ──────────────────────────────────────────
@dataclass
class DepthObstacleConfig(ModuleConfig):
    """深度障碍物检测参数"""

    # DCW 内参（640×360）
    fx: float = 570.0   # focal length x (pixels)
    fy: float = 570.0   # focal length y (pixels)
    cx: float = 320.0   # principal point x
    cy: float = 180.0   # principal point y

    # 深度范围（米）
    min_depth: float = 0.1    # 最近 10cm
    max_depth: float = 3.0    # 最远 3m
    floor_threshold: float = 0.02  # 低于此高度的点视为地面

    # 聚类参数
    grid_rows: int = 12    # 深度图分块行数
    grid_cols: int = 20    # 深度图分块列数
    merge_threshold: float = 0.15  # 相邻块深度差 < 此值 → 合并（米）
    min_box_volume: float = 0.0005  # 最小包围盒体积 (m³)，过滤噪声
    max_objects: int = 10    # 最多检测几个物体

    # 发布频率
    publish_hz: float = 10.0


# ── 深度→3D工具 ─────────────────────────────────────

def depth_to_point_cloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """将深度图转为 3D 点云 (N, 4): [x, y, z, intensity]"""
    h, w = depth.shape
    u = np.tile(np.arange(w), (h, 1))
    v = np.tile(np.arange(h)[:, None], (1, w))

    z = depth
    valid = (z > 0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.zeros((h, w, 4), dtype=np.float32)
    points[:, :, 0] = x
    points[:, :, 1] = y
    points[:, :, 2] = z
    points[:, :, 3] = valid.astype(np.float32)
    return points


def depth_grid_cluster(
    depth: np.ndarray,
    grid_rows: int, grid_cols: int,
    fx: float, fy: float, cx: float, cy: float,
    min_depth: float, max_depth: float,
    floor_threshold: float,
    merge_threshold: float,
    min_box_volume: float,
    max_objects: int,
) -> list[dict[str, Any]]:
    """深度图 → 3D 包围盒列表
    
    Returns: list of {center: (x,y,z), size: (w,h,d), volume: float}
    """
    h, w = depth.shape
    cell_h = max(1, h // grid_rows)
    cell_w = max(1, w // grid_cols)

    # Step 1: 每块的中位深度 + 3D 中心
    cells: list[dict[str, Any]] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            r0, r1 = r * cell_h, min((r + 1) * cell_h, h)
            c0, c1 = c * cell_w, min((c + 1) * cell_w, w)
            d = depth[r0:r1, c0:c1]
            d = d[d > 0]
            if d.size < 10:  # 块内有效深度太少，跳过
                continue
            d_med = np.median(d)
            if d_med < min_depth or d_med > max_depth:
                continue

            # 像素中心
            uc = (c0 + c1) / 2
            vc = (r0 + r1) / 2
            xc = (uc - cx) * d_med / fx
            yc = (vc - cy) * d_med / fy
            zc = d_med

            cells.append({
                "x": xc, "y": yc, "z": zc,
                "d": d_med,
                "count": d.size,
            })

    if not cells:
        return []

    # Step 2: 按深度排序，然后区域生长合并
    cells.sort(key=lambda c: c["d"])
    merged: list[dict[str, Any]] = []
    used = [False] * len(cells)

    for i in range(len(cells)):
        if used[i]:
            continue
        # 以 cell[i] 为种子，收集所有深度接近的相邻 cell
        cluster = [cells[i]]
        used[i] = True

        for j in range(i + 1, len(cells)):
            if used[j]:
                continue
            if abs(cells[j]["d"] - cells[i]["d"]) < merge_threshold:
                # 检查空间相邻性
                dx = abs(cells[j]["x"] - cells[i]["x"])
                dy = abs(cells[j]["y"] - cells[i]["y"])
                if dx < 0.3 and dy < 0.3:
                    cluster.append(cells[j])
                    used[j] = True

        if len(cluster) >= 2:  # 至少 2 个 cell 才是一个物体
            xs = [c["x"] for c in cluster]
            ys = [c["y"] for c in cluster]
            zs = [c["z"] for c in cluster]

            center = np.median(xs), np.median(ys), np.median(zs)
            size_w = max(0.05, np.max(xs) - np.min(xs))
            size_h = max(0.05, np.max(ys) - np.min(ys))
            size_d = max(0.03, np.max(zs) - np.min(zs))
            volume = size_w * size_h * size_d

            if volume >= min_box_volume:
                merged.append({
                    "center": center,
                    "size": (size_w, size_h, size_d),
                    "volume": volume,
                    "cells": len(cluster),
                })

    # Step 3: 按体积排序，取最大 N 个
    merged.sort(key=lambda b: b["volume"], reverse=True)
    return merged[:max_objects]


# ── Module ──────────────────────────────────────────

class DepthObstacleBridge(Module):
    """DCW Depth → Detection3D 桥接模块"""

    config: DepthObstacleConfig
    depth_image: In[Image]
    detections: Out[Detection3D]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = False
        self._thread: threading.Thread | None = None
        self._sequence = 0

    def on_depth(self, img: Image) -> None:
        """收到 depth 帧时处理"""
        if not self._running:
            return
        # 累积到队列
        self._latest_depth = img

    def _loop(self) -> None:
        interval = 1.0 / self.config.publish_hz
        last_publish = 0.0
        self._latest_depth: Image | None = None

        while self._running:
            time.sleep(0.01)
            now = time.time()
            if now - last_publish < interval:
                continue
            if self._latest_depth is None:
                continue

            img = self._latest_depth
            self._latest_depth = None

            try:
                depth_data = np.array(img.data, dtype=np.float32)
                if depth_data.ndim == 3:
                    depth_data = depth_data[:, :, 0]
                
                # 降采样加速
                small = cv2.resize(depth_data, (320, 180), interpolation=cv2.INTER_NEAREST)

                objs = depth_grid_cluster(
                    small,
                    grid_rows=self.config.grid_rows,
                    grid_cols=self.config.grid_cols,
                    fx=self.config.fx / 2,  # 降采样后焦距减半
                    fy=self.config.fy / 2,
                    cx=self.config.cx / 2,
                    cy=self.config.cy / 2,
                    min_depth=self.config.min_depth,
                    max_depth=self.config.max_depth,
                    floor_threshold=self.config.floor_threshold,
                    merge_threshold=self.config.merge_threshold,
                    min_box_volume=self.config.min_box_volume,
                    max_objects=self.config.max_objects,
                )

                for i, obj in enumerate(objs):
                    det = Detection3D()
                    det.id = f"depth_{self._sequence}_{i}"
                    det.header.frame_id = "dcw_depth_optical_frame"
                    det.header.seq = self._sequence
                    
                    center = obj["center"]
                    size = obj["size"]
                    
                    det.bbox = BoundingBox3D()
                    det.bbox.center = Pose(
                        position=Point(float(center[0]), float(center[1]), float(center[2])),
                        orientation=Quaternion(0, 0, 0, 1),
                    )
                    det.bbox.size = Vector3(float(size[0]), float(size[1]), float(size[2]))
                    
                    self.detections.publish(det)

                self._sequence += 1
                last_publish = now

            except Exception as e:
                logger.error(f"Depth→Detection failed: {e}")
                import traceback
                traceback.print_exc()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"DepthObstacleBridge started ({self.config.publish_hz}Hz)")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("DepthObstacleBridge stopped")


# ── 独立测试 ────────────────────────────────────────

def main() -> None:
    """测试: 读取 DCW depth 并打印检测到的物体"""
    import sys
    sys.path.insert(0, "/home/asus/dimos-advx-zmu/.venv-mediapipe/lib/python3.12/site-packages")
    
    from pyorbbecsdk import Config, OBSensorType, Pipeline as DCPipeline
    
    pipeline = DCPipeline()
    config = Config()
    profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config.enable_stream(profile)
    pipeline.start(config)
    
    cfg = DepthObstacleConfig()
    print(f"🔍 深度障碍物检测 — 启动")
    print(f"  (站到摄像头前来回走动, 看控制台输出)")
    
    seq = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
            df = frames.get_depth_frame()
            if not df:
                continue

            raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(df.get_height(), df.get_width())
            depth_m = raw.astype(np.float32) * df.get_depth_scale()

            objs = depth_grid_cluster(
                cv2.resize(depth_m, (320, 180)),
                grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols,
                fx=cfg.fx / 2, fy=cfg.fy / 2,
                cx=cfg.cx / 2, cy=cfg.cy / 2,
                min_depth=cfg.min_depth, max_depth=cfg.max_depth,
                floor_threshold=cfg.floor_threshold,
                merge_threshold=cfg.merge_threshold,
                min_box_volume=cfg.min_box_volume,
                max_objects=cfg.max_objects,
            )

            seq += 1
            if seq % 10 == 0:
                if objs:
                    for o in objs:
                        c, s = o["center"], o["size"]
                        print(f"  📦 ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  {s[0]:.2f}×{s[1]:.2f}×{s[2]:.2f}  vol={o['volume']:.4f}")
                else:
                    print(f"  (空，无物体)")
    except KeyboardInterrupt:
        pipeline.stop()
        print("完成")


if __name__ == "__main__":
    main()
