# Copyright 2026 Dimensional Inc.
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

"""Bounded face-follow controller for the Orbbec DCW demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Any

from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped


@dataclass(frozen=True)
class FaceFollowConfig:
    """Conservative visual-servo limits for a short operator demo."""

    topic: str = "/coordinator_ee_twist_command"
    task_name: str = "eef_twist_arm"
    deadzone: float = 0.08
    depth_target_m: float = 1.0
    depth_deadzone_m: float = 0.12
    min_depth_m: float = 0.35
    max_depth_m: float = 3.0
    yaw_gain: float = 0.30
    pitch_gain: float = 0.24
    depth_gain: float = 0.10
    max_yaw_speed: float = 0.12
    max_pitch_speed: float = 0.10
    max_linear_speed: float = 0.04
    max_yaw_accel: float = 0.40
    max_pitch_accel: float = 0.35
    max_linear_accel: float = 0.08
    yaw_sign: float = -1.0
    pitch_sign: float = 1.0
    linear_sign: float = 1.0
    face_smoothing: float = 0.35
    depth_smoothing: float = 0.25
    acquire_frames: int = 3
    acquire_depth_frames: int = 3
    max_enabled_seconds: float = 60.0


@dataclass
class FaceFollowStatus:
    enabled: bool = False
    state: str = "disabled"
    face_center_x: float | None = None
    face_center_y: float | None = None
    error_x: float = 0.0
    error_y: float = 0.0
    depth_m: float | None = None
    depth_error_m: float = 0.0
    yaw_speed: float = 0.0
    pitch_speed: float = 0.0
    linear_speed: float = 0.0
    face_streak: int = 0
    depth_streak: int = 0
    enabled_seconds: float = 0.0
    last_error: str = ""


Publisher = Callable[[TwistStamped], Any]


class FaceFollowController:
    """Turn normalized face offsets into timeout-safe EEF angular twists.

    Motion is disabled by default. The controller needs several consecutive
    detections before moving, publishes zero velocity on face loss/disable,
    and automatically disables after ``max_enabled_seconds``.
    """

    def __init__(
        self,
        config: FaceFollowConfig | None = None,
        *,
        publisher: Publisher | None = None,
    ) -> None:
        self.config = config or FaceFollowConfig()
        self.status = FaceFollowStatus()
        self._publisher = publisher
        self._lock = threading.RLock()
        self._enabled_at = 0.0
        self._last_update_at = 0.0
        self._last_command_nonzero = False
        self._filtered_face: tuple[float, float] | None = None
        self._filtered_depth_m: float | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config
        if not 0.0 <= cfg.deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        if cfg.acquire_frames < 1:
            raise ValueError("acquire_frames must be positive")
        if cfg.acquire_depth_frames < 1:
            raise ValueError("acquire_depth_frames must be positive")
        if cfg.max_enabled_seconds <= 0:
            raise ValueError("max_enabled_seconds must be positive")
        if not cfg.min_depth_m < cfg.depth_target_m < cfg.max_depth_m:
            raise ValueError("depth target must be between the minimum and maximum depth")
        if not 0.0 <= cfg.depth_deadzone_m < cfg.depth_target_m - cfg.min_depth_m:
            raise ValueError("depth_deadzone_m is outside the safe target range")
        if not 0.0 < cfg.face_smoothing <= 1.0:
            raise ValueError("face_smoothing must be in (0, 1]")
        if not 0.0 < cfg.depth_smoothing <= 1.0:
            raise ValueError("depth_smoothing must be in (0, 1]")
        non_negative = (
            cfg.yaw_gain,
            cfg.pitch_gain,
            cfg.depth_gain,
            cfg.max_yaw_speed,
            cfg.max_pitch_speed,
            cfg.max_linear_speed,
            cfg.max_yaw_accel,
            cfg.max_pitch_accel,
            cfg.max_linear_accel,
        )
        if min(non_negative) < 0:
            raise ValueError("gains and speed limits must be non-negative")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self.status)

    def set_enabled(self, enabled: bool, *, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        with self._lock:
            if enabled:
                self.status = FaceFollowStatus(enabled=True, state="searching")
                self._enabled_at = now
                self._last_update_at = now
                self._last_command_nonzero = False
                self._filtered_face = None
                self._filtered_depth_m = None
            else:
                self._publish_zero()
                self.status.enabled = False
                self.status.state = "disabled"
                self.status.face_streak = 0
                self.status.depth_streak = 0
                self.status.enabled_seconds = 0.0
                self._filtered_face = None
                self._filtered_depth_m = None
            return self.snapshot()

    def update(
        self,
        face_center: tuple[float, float] | None,
        face_depth_m: float | None = None,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self.status.enabled:
                return self.snapshot()

            dt = max(0.0, min(0.25, now - self._last_update_at))
            self._last_update_at = now
            self.status.enabled_seconds = max(0.0, now - self._enabled_at)
            if self.status.enabled_seconds >= self.config.max_enabled_seconds:
                self._publish_zero()
                self.status.enabled = False
                self.status.state = "timed_out"
                self.status.face_streak = 0
                return self.snapshot()

            if face_center is None:
                self.status.face_center_x = None
                self.status.face_center_y = None
                self.status.error_x = 0.0
                self.status.error_y = 0.0
                self.status.depth_m = None
                self.status.depth_error_m = 0.0
                self.status.face_streak = 0
                self.status.depth_streak = 0
                self.status.state = "searching"
                self._filtered_face = None
                self._filtered_depth_m = None
                self._publish_zero()
                return self.snapshot()

            center_x = min(1.0, max(0.0, float(face_center[0])))
            center_y = min(1.0, max(0.0, float(face_center[1])))
            if self._filtered_face is None:
                filtered_x, filtered_y = center_x, center_y
            else:
                alpha = self.config.face_smoothing
                filtered_x = self._low_pass(self._filtered_face[0], center_x, alpha)
                filtered_y = self._low_pass(self._filtered_face[1], center_y, alpha)
            self._filtered_face = (filtered_x, filtered_y)
            error_x = 2.0 * (filtered_x - 0.5)
            error_y = 2.0 * (filtered_y - 0.5)
            self.status.face_center_x = filtered_x
            self.status.face_center_y = filtered_y
            self.status.error_x = error_x
            self.status.error_y = error_y
            self.status.face_streak += 1

            depth_valid = (
                face_depth_m is not None
                and math.isfinite(face_depth_m)
                and self.config.min_depth_m <= face_depth_m <= self.config.max_depth_m
            )
            if depth_valid:
                measured_depth = float(face_depth_m)
                if self._filtered_depth_m is None:
                    filtered_depth = measured_depth
                else:
                    filtered_depth = self._low_pass(
                        self._filtered_depth_m,
                        measured_depth,
                        self.config.depth_smoothing,
                    )
                self._filtered_depth_m = filtered_depth
                self.status.depth_m = filtered_depth
                self.status.depth_error_m = filtered_depth - self.config.depth_target_m
                self.status.depth_streak += 1
            else:
                self._filtered_depth_m = None
                self.status.depth_m = None
                self.status.depth_error_m = 0.0
                self.status.depth_streak = 0

            if self.status.face_streak < self.config.acquire_frames:
                self.status.state = "acquiring"
                self._publish_zero()
                return self.snapshot()

            yaw = self.config.yaw_sign * self.config.yaw_gain * self._outside_deadzone(error_x)
            pitch = (
                self.config.pitch_sign * self.config.pitch_gain * self._outside_deadzone(error_y)
            )
            yaw_target = self._clip(yaw, self.config.max_yaw_speed)
            pitch_target = self._clip(pitch, self.config.max_pitch_speed)
            yaw = self._slew(
                self.status.yaw_speed,
                yaw_target,
                self.config.max_yaw_accel * dt,
            )
            pitch = self._slew(
                self.status.pitch_speed,
                pitch_target,
                self.config.max_pitch_accel * dt,
            )

            linear = 0.0
            if depth_valid and self.status.depth_streak >= self.config.acquire_depth_frames:
                depth_error = self._outside_symmetric_deadzone(
                    self.status.depth_error_m,
                    self.config.depth_deadzone_m,
                )
                linear_target = self._clip(
                    self.config.linear_sign * self.config.depth_gain * depth_error,
                    self.config.max_linear_speed,
                )
                linear = self._slew(
                    self.status.linear_speed,
                    linear_target,
                    self.config.max_linear_accel * dt,
                )

            self._publish(linear=linear, pitch=pitch, yaw=yaw)
            self.status.yaw_speed = yaw
            self.status.pitch_speed = pitch
            self.status.linear_speed = linear
            if not depth_valid:
                self.status.state = "tracking_no_depth"
            elif yaw == 0.0 and pitch == 0.0 and linear == 0.0:
                self.status.state = "centered"
            else:
                self.status.state = "tracking"
            return self.snapshot()

    def stop(self) -> None:
        self.set_enabled(False)

    def _outside_deadzone(self, error: float) -> float:
        magnitude = abs(error)
        if magnitude <= self.config.deadzone:
            return 0.0
        normalized = (magnitude - self.config.deadzone) / (1.0 - self.config.deadzone)
        return math.copysign(normalized, error)

    @staticmethod
    def _outside_symmetric_deadzone(error: float, deadzone: float) -> float:
        if abs(error) <= deadzone:
            return 0.0
        return math.copysign(abs(error) - deadzone, error)

    @staticmethod
    def _low_pass(previous: float, value: float, alpha: float) -> float:
        return previous + alpha * (value - previous)

    @staticmethod
    def _slew(previous: float, target: float, max_delta: float) -> float:
        if max_delta <= 0.0:
            return previous
        return previous + min(max_delta, max(-max_delta, target - previous))

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return min(limit, max(-limit, value))

    def _ensure_publisher(self) -> Publisher:
        if self._publisher is None:
            from dimos.core.transport import LCMTransport

            transport: LCMTransport[TwistStamped] = LCMTransport(
                self.config.topic,
                TwistStamped,
            )
            self._publisher = transport.publish
        return self._publisher

    def _publish(self, *, linear: float, pitch: float, yaw: float) -> None:
        try:
            self._ensure_publisher()(
                TwistStamped(
                    frame_id=self.config.task_name,
                    linear=[linear, 0.0, 0.0],
                    angular=[0.0, pitch, yaw],
                )
            )
            self._last_command_nonzero = linear != 0.0 or pitch != 0.0 or yaw != 0.0
            self.status.last_error = ""
        except Exception as exc:
            self.status.last_error = str(exc)
            self.status.state = "publish_error"
            self._last_command_nonzero = False

    def _publish_zero(self) -> None:
        self.status.yaw_speed = 0.0
        self.status.pitch_speed = 0.0
        self.status.linear_speed = 0.0
        if self._publisher is None and not self._last_command_nonzero:
            return
        self._publish(linear=0.0, pitch=0.0, yaw=0.0)
        self._last_command_nonzero = False
