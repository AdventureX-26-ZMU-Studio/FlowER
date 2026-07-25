"""
vision_server.py 补丁指南

需要改 5 处。每一处都用 + 标记新增代码。
"""

# ── 改动 1: import ──────────────────────────────────
# 在文件顶部 imports 区域加入:
"""
from dimos.hardware.sensors.camera.orbbec_dcw.depth_cluster_engine import (
    DepthClusterEngine, DepthClusterConfig,
)
"""

# ── 改动 2: __init__ 初始化引擎 ─────────────────────
# 在 DCWVisionService.__init__() 末尾加入:
"""
        # 深度聚类引擎
        self._cluster_engine = DepthClusterEngine(DepthClusterConfig())
        self._latest_obstacles: list[dict] = []
        self._latest_obstacles_ts = 0.0
"""

# ── 改动 3: capture loop 中调用聚类 ─────────────────
# 在 depth 帧处理完毕后（depth_preview 调用之后）加入:
"""
            # 深度聚类
            if depth_m is not None and depth_m.size > 0:
                h, w = depth_m.shape
                fx = self._calib.get("fx", 570.0) * w / 640
                fy = self._calib.get("fy", 570.0) * h / 360
                cx = self._calib.get("cx", 320.0) * w / 640
                cy = self._calib.get("cy", 180.0) * h / 360
                self._latest_obstacles = self._cluster_engine.process(
                    depth_m, fx, fy, cx, cy,
                )
                self._latest_obstacles_ts = time.monotonic()
"""

# ── 改动 4: /status 加障碍物字段 ────────────────────
# 在 status JSON 构建处加入:
"""
            "obstacle_count": len(self._latest_obstacles),
            "obstacles": self._latest_obstacles,
"""

# ── 改动 5: 新增端点 ────────────────────────────────
# 在 do_GET 的 elif 链中加入:
"""
        elif self.path == "/depth_raw" and self._service._depth_frame is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(640 * 360 * 2))
            self.end_headers()
            raw = self._service._depth_frame
            if raw is not None and raw.size > 0:
                self.wfile.write(raw.astype(np.uint16).tobytes())

        elif self.path == "/obstacles":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(self._service._latest_obstacles).encode())
"""

print("✅ 补丁指南已生成")
print("改动位置: vision_server.py 中的 5 处")
print("引擎文件: depth_cluster_engine.py (同目录)")
