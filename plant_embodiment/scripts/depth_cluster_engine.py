"""
Depth Cluster Engine — DCW 深度聚类引擎
========================================
可作为 vision_server.py 的内置模块使用。
在 capture loop 中调用 process()，结果自动写入 FrameStore。
"""

from __future__ import annotations

import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DepthClusterConfig:
    """深度聚类参数"""
    input_width: int = 320
    input_height: int = 180
    grid_rows: int = 12
    grid_cols: int = 20
    min_depth_m: float = 0.1
    max_depth_m: float = 3.0
    merge_dz: float = 0.15  # 相邻块深度差 < 此值 → 合并 (米)
    min_volume_m3: float = 0.0005  # 最小体积 (m³)
    max_objects: int = 10
    # 持久化参数 (Spatial Memory)
    stale_timeout_s: float = 5.0  # 超过此秒数未更新 → 标记为 stale


@dataclass
class TrackedObject:
    """带时间戳的跟踪物体"""
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    volume: float
    first_seen: float
    last_seen: float


class DepthClusterEngine:
    """深度聚类引擎"""

    def __init__(self, config: DepthClusterConfig | None = None):
        self.config = config or DepthClusterConfig()
        self._seq = 0
        self._objects: dict[str, TrackedObject] = {}

    def process(self, depth_m: np.ndarray, fx: float, fy: float,
                cx: float, cy: float) -> list[dict[str, Any]]:
        """输入深度图 (米)，输出障碍物列表"""
        # 降采样
        h, w = self.config.input_height, self.config.input_width
        small = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

        # 分块
        rh, cw = max(1, h // self.config.grid_rows), max(1, w // self.config.grid_cols)
        cells = []
        for r in range(self.config.grid_rows):
            for c in range(self.config.grid_cols):
                patch = small[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
                patch = patch[patch > 0]
                if patch.size < 10:
                    continue
                d = float(np.median(patch))
                if d < self.config.min_depth_m or d > self.config.max_depth_m:
                    continue
                uc, vc = (2*c+1)*cw//2, (2*r+1)*rh//2
                cells.append((
                    (uc - cx) * d / fx,   # X
                    (vc - cy) * d / fy,   # Y
                    d,                     # Z
                    patch.size,
                ))

        if not cells:
            return []

        # 深度排序 + 区域生长合并
        cells.sort(key=lambda c: c[2])
        used = [False] * len(cells)
        clusters = []

        for i in range(len(cells)):
            if used[i]:
                continue
            group = [cells[i]]
            used[i] = True
            for j in range(i+1, len(cells)):
                if used[j]:
                    continue
                if abs(cells[j][2] - cells[i][2]) < self.config.merge_dz:
                    if (abs(cells[j][0] - cells[i][0]) < 0.3 and
                            abs(cells[j][1] - cells[i][1]) < 0.3):
                        group.append(cells[j])
                        used[j] = True
            if len(group) >= 2:
                xs = [c[0] for c in group]
                ys = [c[1] for c in group]
                zs = [c[2] for c in group]
                cm = (float(np.median(xs)), float(np.median(ys)), float(np.median(zs)))
                sz = (max(0.05, np.max(xs)-np.min(xs)),
                      max(0.05, np.max(ys)-np.min(ys)),
                      max(0.03, np.max(zs)-np.min(zs)))
                vol = sz[0] * sz[1] * sz[2]
                if vol >= self.config.min_volume_m3:
                    clusters.append((cm, sz, vol, len(group)))

        clusters.sort(key=lambda c: c[2], reverse=True)
        clusters = clusters[:self.config.max_objects]

        now = time.monotonic()
        seen_ids = set()
        results = []

        for cm, sz, vol, _ in clusters:
            # 匹配已有物体 (中心距离 < 0.3m 视为同一物体)
            matched_id = None
            for oid, obj in self._objects.items():
                dx = abs(obj.center[0] - cm[0])
                dy = abs(obj.center[1] - cm[1])
                dz = abs(obj.center[2] - cm[2])
                if dx + dy + dz < 0.3:
                    matched_id = oid
                    break

            if matched_id:
                # 更新已有物体
                obj = self._objects[matched_id]
                obj.center = cm
                obj.size = sz
                obj.volume = vol
                obj.last_seen = now
                oid = matched_id
            else:
                # 新建
                oid = f"depth_{self._seq}_{len(results)}"
                self._objects[oid] = TrackedObject(
                    center=cm, size=sz, volume=vol,
                    first_seen=now, last_seen=now,
                )

            seen_ids.add(oid)
            obj = self._objects[oid]
            results.append({
                "id": oid,
                "center_m": [round(cm[0], 3), round(cm[1], 3), round(cm[2], 3)],
                "size_m": [round(sz[0], 3), round(sz[1], 3), round(sz[2], 3)],
                "volume_m3": round(vol, 4),
                "first_seen_s": round(now - obj.first_seen, 1),
                "last_seen_s": round(now - obj.last_seen, 1),
            })

        # 移除过期物体
        stale = [oid for oid, obj in self._objects.items()
                 if now - obj.last_seen > self.config.stale_timeout_s]
        for oid in stale:
            del self._objects[oid]

        self._seq += 1
        return results
