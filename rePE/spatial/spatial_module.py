"""
spatial_module — dimos Module: depth clustering + spatial memory + WM sync.

Consumes VisionHub depth via WebSocket → clusters → maintains world-object
registry → serves HTTP /obstacles (for DepthObstacleSource) and /status.

Architecture:
  VisionHub WS (depth_uint16_b64)
       │
       ▼
  DepthClusterEngine.process()  →  clusters (camera frame)
       │
       ▼
  A1Z wrist pose lookup         →  transform to world coordinates
       │
       ▼
  WorldObjectRegistry           →  match / create / update / (rare delete)
       │
       ▼
  HTTP :8891 /obstacles         →  DepthObstacleSource polls this
  HTTP :8891 /status            →  memory table status
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rePE.spatial.cluster_engine import DepthClusterConfig, DepthClusterEngine

# ── dimos imports (lazy, so module can be tested standalone) ──
try:
    from dimos.core.module import Module, ModuleConfig
    from dimos.core.core import rpc
except ImportError:
    class Module:
        config: Any = None
        def start(self) -> None: pass
        def stop(self) -> None: pass
    class ModuleConfig:
        pass
    def rpc(fn):  # type: ignore
        fn.__rpc__ = True
        return fn


# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════

@dataclass
class SpatialConfig:
    """Configuration for the spatial memory module."""

    # VisionHub
    vision_url: str = "http://127.0.0.1:8890"
    vision_ws_streams: list[str] = field(default_factory=lambda: ["depth"])

    # Clustering
    cluster: DepthClusterConfig = field(default_factory=DepthClusterConfig)

    # DCW intrinsics (640×360 native)
    fx: float = 570.0
    fy: float = 570.0
    cx: float = 320.0
    cy: float = 180.0

    # World matching threshold (meters)
    match_distance_m: float = 0.3

    # HTTP endpoint for DepthObstacleSource
    http_host: str = "0.0.0.0"
    http_port: int = 8891

    # How often to publish /obstacles (Hz)
    publish_hz: float = 5.0

    # Camera mount offset from A1Z wrist (meters, tool frame)
    camera_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ═══════════════════════════════════════════════════
# Memory table entry
# ═══════════════════════════════════════════════════

@dataclass
class WorldObject:
    """An object in the spatial memory table."""

    world_id: str
    label: str | None = None          # from perception (face_0, cup_1)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    viewing_pose: Any = None          # camera pose when first seen
    first_seen: float = 0.0
    last_seen: float = 0.0
    source: str = "depth"             # "depth" | "perception"
    persistent: bool = True           # depth→True, face/hand→False


# ═══════════════════════════════════════════════════
# World object registry
# ═══════════════════════════════════════════════════

class WorldObjectRegistry:
    """Maintains the spatial memory table with CUD operations."""

    def __init__(self, match_distance_m: float = 0.3) -> None:
        self._objects: dict[str, WorldObject] = {}
        self._seq = 0
        self._match_distance = match_distance_m

    def upsert(
        self,
        position: tuple[float, float, float],
        size: tuple[float, float, float],
        source: str = "depth",
    ) -> tuple[str, bool]:
        """Insert or update an object. Returns (world_id, is_new)."""
        now = time.monotonic()

        # Find existing match by world position
        matched_id = self._find_match(position)

        if matched_id:
            obj = self._objects[matched_id]
            obj.position = position
            obj.size = size
            obj.last_seen = now
            obj.source = source
            return matched_id, False

        # New object
        world_id = f"spatial_{self._seq}"
        self._seq += 1
        self._objects[world_id] = WorldObject(
            world_id=world_id,
            position=position,
            size=size,
            first_seen=now,
            last_seen=now,
            source=source,
            persistent=(source == "depth"),
        )
        return world_id, True

    def set_label(self, world_id: str, label: str) -> None:
        """Apply a label from perception to a world object."""
        if world_id in self._objects:
            self._objects[world_id].label = label

    def remove(self, world_id: str) -> bool:
        """Remove an object (only for confirmed false positives)."""
        if world_id in self._objects:
            del self._objects[world_id]
            return True
        return False

    def get_visible_objects(self) -> list[dict[str, Any]]:
        """Return objects for /obstacles endpoint (camera-frame coords)."""
        results = []
        for obj in self._objects.values():
            results.append({
                "id": obj.world_id,
                "center_m": list(obj.position),
                "size_m": list(obj.size),
            })
        return results

    def status(self) -> dict[str, Any]:
        """Return registry status."""
        return {
            "object_count": len(self._objects),
            "objects": [
                {
                    "id": o.world_id,
                    "label": o.label,
                    "position": list(o.position),
                    "size": list(o.size),
                    "persistent": o.persistent,
                    "age_s": round(time.monotonic() - o.first_seen, 1),
                }
                for o in self._objects.values()
            ],
        }

    def _find_match(self, position: tuple[float, float, float]) -> str | None:
        """Find the closest object within match_distance."""
        best_id = None
        best_dist = float("inf")
        for obj_id, obj in self._objects.items():
            dx = abs(obj.position[0] - position[0])
            dy = abs(obj.position[1] - position[1])
            dz = abs(obj.position[2] - position[2])
            dist = dx + dy + dz
            if dist < self._match_distance and dist < best_dist:
                best_dist = dist
                best_id = obj_id
        return best_id


# ═══════════════════════════════════════════════════
# Minimal HTTP server for DepthObstacleSource
# ═══════════════════════════════════════════════════

_OBSTACLES_DATA: list[dict[str, Any]] = []
_REGISTRY: WorldObjectRegistry | None = None


def _run_http_server(host: str, port: int) -> None:
    """Minimal HTTP server (no FastAPI dependency)."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/obstacles":
                self._json_response(_OBSTACLES_DATA)
            elif self.path == "/status":
                self._json_response(
                    _REGISTRY.status() if _REGISTRY else {"error": "not ready"}
                )
            elif self.path == "/health":
                self._json_response({"status": "ok"})
            else:
                self.send_response(404)
                self.end_headers()

        def _json_response(self, data: Any) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass  # suppress logs

    server = HTTPServer((host, port), Handler)
    server.serve_forever()


# ═══════════════════════════════════════════════════
# Spatial Module
# ═══════════════════════════════════════════════════

class SpatialModule(Module):
    """dimos Module: depth clustering + spatial memory + WM sync."""

    config: SpatialConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._engine = DepthClusterEngine(self.config.cluster)
        self._registry = WorldObjectRegistry(self.config.match_distance_m)
        self._running = False
        self._thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None

        # Inject registry into global for HTTP handler
        global _REGISTRY, _OBSTACLES_DATA
        _REGISTRY = self._registry

    @rpc
    def start(self) -> None:
        self._running = True

        # Start HTTP server in background
        self._http_thread = threading.Thread(
            target=_run_http_server,
            args=(self.config.http_host, self.config.http_port),
            daemon=True,
        )
        self._http_thread.start()

        # Start depth processing loop
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        # HTTP server is daemon, exits with process

    def _run_loop(self) -> None:
        """Main loop: consume depth from VisionHub → cluster → update registry."""
        import asyncio as _asyncio

        loop = _asyncio.new_event_loop()

        async def _consume() -> None:
            from rePE.vision.client import VisionHubClient

            client = VisionHubClient(
                self.config.vision_url, service_name="spatial"
            )
            async for frame in client.frames(self.config.vision_ws_streams):
                if not self._running:
                    break
                self._process_frame(frame)

        loop.run_until_complete(_consume())
        loop.close()

    def _process_frame(self, frame: Any) -> None:
        """Decode depth, run clustering, update registry."""
        global _OBSTACLES_DATA

        depth_b64 = getattr(frame, "depth_uint16_b64", None)
        if not depth_b64:
            return

        try:
            raw = base64.b64decode(depth_b64)
            if len(raw) != 360 * 640 * 2:
                return
            depth_mm = np.frombuffer(raw, dtype=np.uint16).reshape(360, 640)
            depth_m = depth_mm.astype(np.float32) / 1000.0
        except Exception:
            return

        # Clustering (camera optical frame)
        clusters = self._engine.process(
            depth_m,
            fx=self.config.fx,
            fy=self.config.fy,
            cx=self.config.cx,
            cy=self.config.cy,
        )

        # Upsert all current clusters into registry.
        # Registry holds camera-frame coords; DepthObstacleSource will
        # transform them to world frame with live A1Z wrist pose.
        for obj in clusters:
            center = tuple(obj["center_m"])
            size = tuple(obj["size_m"])
            self._registry.upsert(center, size, source="depth")

        # Publish FULL memory table (not just current frame).
        # This is the key to spatial memory: DepthObstacleSource sees
        # all objects and never removes them just because the camera
        # moved away.
        _OBSTACLES_DATA = self._registry.get_visible_objects()
