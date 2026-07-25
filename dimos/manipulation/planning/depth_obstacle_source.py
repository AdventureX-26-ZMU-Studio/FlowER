# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Poll DCW depth clusters and maintain collision boxes in a planning world."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any
from urllib import request

import numpy as np
from pydantic import BaseModel, Field

from dimos.manipulation.planning.spec.enums import ObstacleType
from dimos.manipulation.planning.spec.models import Obstacle
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Standard camera optical coordinates are x-right, y-down, z-forward.  The A1Z
# wrist tool frame is x-forward, y-left, z-up.
_EEF_FROM_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


class DepthObstacleSourceConfig(BaseModel):
    """Configuration for the fail-closed DCW obstacle feed."""

    url: str
    robot_name: str = "arm"
    poll_hz: float = Field(default=5.0, gt=0.0, le=20.0)
    request_timeout_s: float = Field(default=0.4, gt=0.0, le=5.0)
    stale_after_s: float = Field(default=1.5, gt=0.0, le=30.0)
    missing_grace_s: float = Field(default=0.75, ge=0.0, le=10.0)
    inflation_m: float = Field(default=0.08, ge=0.0, le=0.5)
    min_extent_m: float = Field(default=0.04, gt=0.0, le=1.0)
    max_extent_m: float = Field(default=3.0, gt=0.0, le=10.0)
    max_obstacles: int = Field(default=10, gt=0, le=100)
    size_update_tolerance_m: float = Field(default=0.04, ge=0.0, le=1.0)
    camera_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class _TrackedObstacle:
    world_id: str
    dimensions: tuple[float, float, float]
    last_seen: float


@dataclass(frozen=True)
class DepthObstacleSourceStatus:
    ready: bool
    fresh: bool
    obstacle_count: int
    age_s: float | None
    last_error: str


class DepthObstacleSource:
    """Background DCW HTTP adapter for ``WorldMonitor``.

    The camera is treated as wrist-mounted and aligned with the A1Z tool:
    optical z is tool x, optical x is -tool y, and optical y is -tool z.
    Boxes are inflated before entering collision checking.  A caller can use
    :meth:`is_fresh` to reject planning and execution if the feed goes stale.
    """

    def __init__(
        self,
        config: DepthObstacleSourceConfig,
        *,
        add_obstacle: Callable[[Obstacle], str],
        update_obstacle_pose: Callable[[str, PoseStamped], bool],
        remove_obstacle: Callable[[str], bool],
        get_mount_pose: Callable[[], PoseStamped | None],
        opener: Callable[..., Any] = request.urlopen,
    ) -> None:
        self.config = config
        self._add_obstacle = add_obstacle
        self._update_obstacle_pose = update_obstacle_pose
        self._remove_obstacle = remove_obstacle
        self._get_mount_pose = get_mount_pose
        self._opener = opener
        self._tracked: dict[str, _TrackedObstacle] = {}
        self._last_success_at = 0.0
        self._last_error = ""
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="DCWDepthObstacles",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, remove: bool = True) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if remove:
            with self._lock:
                for tracked in list(self._tracked.values()):
                    self._remove_obstacle(tracked.world_id)
                self._tracked.clear()

    def is_fresh(self, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            return (
                self._last_success_at > 0.0
                and now - self._last_success_at <= self.config.stale_after_s
            )

    def status(self, *, now: float | None = None) -> DepthObstacleSourceStatus:
        now = time.monotonic() if now is None else now
        with self._lock:
            age = None if self._last_success_at <= 0.0 else max(0.0, now - self._last_success_at)
            return DepthObstacleSourceStatus(
                ready=self._last_success_at > 0.0,
                fresh=self.is_fresh(now=now),
                obstacle_count=len(self._tracked),
                age_s=age,
                last_error=self._last_error,
            )

    def sync_once(self, *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        endpoint = f"{self.config.url.rstrip('/')}/obstacles"
        try:
            with self._opener(endpoint, timeout=self.config.request_timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise ValueError("DCW /obstacles response must be a JSON list")
            mount_pose = self._get_mount_pose()
            if mount_pose is None:
                raise RuntimeError("A1Z mount pose is unavailable")
            obstacles = self._convert_payload(payload, mount_pose)
            self._apply(obstacles, now)
            with self._lock:
                self._last_success_at = now
                self._last_error = ""
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.debug(f"DCW depth obstacle update failed: {exc}")

    def _run(self) -> None:
        period = 1.0 / self.config.poll_hz
        while not self._stop_event.is_set():
            started = time.monotonic()
            self.sync_once(now=started)
            self._stop_event.wait(max(0.0, period - (time.monotonic() - started)))

    def _convert_payload(
        self,
        payload: list[Any],
        mount_pose: PoseStamped,
    ) -> dict[str, Obstacle]:
        rotation_world_mount = mount_pose.orientation.to_rotation_matrix()
        translation_world_mount = mount_pose.position.to_numpy()
        translation_mount_camera = np.asarray(
            self.config.camera_translation_m,
            dtype=np.float64,
        )
        rotation_world_optical = rotation_world_mount @ _EEF_FROM_OPTICAL
        orientation = Quaternion.from_rotation_matrix(rotation_world_optical)
        converted: dict[str, Obstacle] = {}

        for item in payload[: self.config.max_obstacles]:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("id", "")).strip()
            center = np.asarray(item.get("center_m", ()), dtype=np.float64)
            size = np.asarray(item.get("size_m", ()), dtype=np.float64)
            if (
                not source_id
                or center.shape != (3,)
                or size.shape != (3,)
                or not np.all(np.isfinite(center))
                or not np.all(np.isfinite(size))
                or np.any(size <= 0.0)
                or np.any(size > self.config.max_extent_m)
            ):
                continue

            center_mount = translation_mount_camera + _EEF_FROM_OPTICAL @ center
            center_world = translation_world_mount + rotation_world_mount @ center_mount
            dimensions = tuple(
                float(max(self.config.min_extent_m, extent) + 2.0 * self.config.inflation_m)
                for extent in size
            )
            if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
                continue
            converted[source_id] = Obstacle(
                name=f"dcw_{source_id}",
                obstacle_type=ObstacleType.BOX,
                pose=PoseStamped(
                    frame_id="world",
                    position=Vector3(center_world),
                    orientation=orientation,
                ),
                dimensions=dimensions,
                color=(1.0, 0.35, 0.05, 0.35),
            )
        return converted

    def _apply(self, obstacles: dict[str, Obstacle], now: float) -> None:
        with self._lock:
            for source_id, obstacle in obstacles.items():
                tracked = self._tracked.get(source_id)
                if tracked is None:
                    world_id = self._add_obstacle(obstacle)
                    if world_id:
                        self._tracked[source_id] = _TrackedObstacle(
                            world_id=world_id,
                            dimensions=obstacle.dimensions,
                            last_seen=now,
                        )
                    continue

                dimensions_changed = any(
                    abs(old - new) > self.config.size_update_tolerance_m
                    for old, new in zip(tracked.dimensions, obstacle.dimensions, strict=True)
                )
                if dimensions_changed:
                    self._remove_obstacle(tracked.world_id)
                    world_id = self._add_obstacle(obstacle)
                    tracked.world_id = world_id
                    tracked.dimensions = obstacle.dimensions
                else:
                    self._update_obstacle_pose(tracked.world_id, obstacle.pose)
                tracked.last_seen = now

            for source_id, tracked in list(self._tracked.items()):
                if (
                    source_id not in obstacles
                    and now - tracked.last_seen > self.config.missing_grace_s
                ):
                    self._remove_obstacle(tracked.world_id)
                    del self._tracked[source_id]
