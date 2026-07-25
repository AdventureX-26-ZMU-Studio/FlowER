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

"""Orbbec DaBai DCW RGB-D camera module.

The vendor SDK is loaded lazily so importing DimOS still works on machines
without an Orbbec camera. The defaults intentionally use a low-bandwidth
profile that is usable when the camera enumerates through USB 2.0.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from pydantic import Field
import reactivex as rx

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.hardware.sensors.camera.spec import (
    OPTICAL_ROTATION,
    DepthCameraConfig,
    DepthCameraHardware,
)
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.spec import perception
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

if TYPE_CHECKING:
    from pyorbbecsdk import (
        ColorFrame,
        Config,
        DepthFrame,
        Pipeline,
        VideoStreamProfile,
    )

logger = setup_logger()

_MM_TO_M = 0.001


def default_base_transform() -> Transform:
    """Return an identity camera mount transform."""
    return Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )


def _load_sdk() -> dict[str, Any]:
    try:
        from pyorbbecsdk import (
            Config,
            OBAlignMode,
            OBFormat,
            OBSensorType,
            Pipeline,
        )
    except ImportError as exc:
        raise RuntimeError(
            "pyorbbecsdk is not installed. Install the Orbbec aarch64 wheel "
            "before starting OrbbecDCWCamera."
        ) from exc
    return {
        "Config": Config,
        "OBAlignMode": OBAlignMode,
        "OBFormat": OBFormat,
        "OBSensorType": OBSensorType,
        "Pipeline": Pipeline,
    }


def _enum_member(enum: Any, name: str) -> Any:
    try:
        return getattr(enum, name.upper())
    except AttributeError as exc:
        choices = sorted(item for item in dir(enum) if item.isupper())
        raise ValueError(f"Unsupported Orbbec format {name!r}; available: {choices}") from exc


def _select_video_profile(
    pipeline: Pipeline,
    sensor_type: Any,
    width: int,
    height: int,
    pixel_format: Any,
    fps: int,
) -> VideoStreamProfile:
    profiles = pipeline.get_stream_profile_list(sensor_type)
    try:
        return profiles.get_video_stream_profile(width, height, pixel_format, fps)
    except Exception as exc:
        raise RuntimeError(
            "The camera does not expose the requested stream profile "
            f"{width}x{height}@{fps} ({pixel_format})"
        ) from exc


def _camera_info_from_intrinsics(
    intrinsic: Any,
    distortion: Any,
    frame_id: str,
    *,
    fallback_width: int,
    fallback_height: int,
) -> CameraInfo:
    fx = float(intrinsic.fx)
    fy = float(intrinsic.fy)
    cx = float(getattr(intrinsic, "cx", getattr(intrinsic, "ppx", 0.0)))
    cy = float(getattr(intrinsic, "cy", getattr(intrinsic, "ppy", 0.0)))
    width = int(getattr(intrinsic, "width", fallback_width))
    height = int(getattr(intrinsic, "height", fallback_height))
    coefficients = [
        float(getattr(distortion, name, 0.0))
        for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    ]
    while coefficients and coefficients[-1] == 0.0:
        coefficients.pop()
    return CameraInfo(
        width=width,
        height=height,
        distortion_model="plumb_bob",
        D=coefficients,
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        frame_id=frame_id,
    )


def _frame_data_u8(frame: Any) -> np.ndarray:
    """Return SDK frame storage as a contiguous, flat byte array."""
    data = frame.get_data()
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    # pyorbbecsdk 1.3.1 on aarch64/Python 3.12 may expose the contiguous
    # frame buffer as an ndarray with a bogus zero stride. Read the buffer
    # through its pointer before NumPy repeats the first byte.
    if isinstance(data, np.ndarray) and not data.flags.c_contiguous:
        get_data_size = getattr(frame, "get_data_size", None)
        data_size = int(get_data_size()) if get_data_size is not None else 0
        if data_size > 0:
            return np.frombuffer(ctypes.string_at(data.ctypes.data, data_size), dtype=np.uint8)
    return np.ascontiguousarray(data).view(np.uint8).reshape(-1)


def _decode_color_frame(frame: ColorFrame, ob_format: Any) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    pixel_format = frame.get_format()
    raw = _frame_data_u8(frame)

    if pixel_format == ob_format.MJPG:
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Orbbec returned an invalid MJPEG color frame")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if pixel_format == ob_format.RGB:
        return raw.reshape(height, width, 3).copy()
    if pixel_format == ob_format.BGR:
        bgr = raw.reshape(height, width, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if pixel_format in (ob_format.YUYV, ob_format.YUY2):
        yuyv = raw.reshape(height, width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
    raise ValueError(f"Unsupported Orbbec color frame format: {pixel_format}")


def _decode_depth_frame(frame: DepthFrame) -> tuple[np.ndarray, float]:
    width = int(frame.get_width())
    height = int(frame.get_height())
    depth = _frame_data_u8(frame).view(np.uint16).reshape(height, width).copy()
    # Orbbec reports millimetres per raw unit; DimOS expects metres per unit.
    depth_scale_m = float(frame.get_depth_scale()) * _MM_TO_M
    return depth, depth_scale_m


class OrbbecDCWConfig(ModuleConfig, DepthCameraConfig):
    """Configuration for the Orbbec DaBai DCW camera."""

    width: int = 640
    height: int = 360
    fps: int = 15
    color_format: str = "MJPG"
    depth_width: int = 640
    depth_height: int = 360
    depth_fps: int = 15
    # DaBai DCW PID 0x0659 exposes Y11/Y12 depth profiles, not Y16/Z16.
    depth_format: str = "Y12"
    camera_name: str = "dcw"
    base_frame_id: str = "base_link"
    base_transform: Transform | None = Field(default_factory=default_base_transform)
    align_depth_to_color: bool = True
    enable_depth: bool = True
    enable_pointcloud: bool = False
    pointcloud_fps: float = 2.0
    camera_info_fps: float = 1.0
    frame_timeout_ms: int = 1000
    pointcloud_voxel_size: float = 0.005


class OrbbecDCWCamera(DepthCameraHardware, Module, perception.DepthCamera):
    """DimOS RGB-D module for an Orbbec DaBai DCW camera."""

    config: OrbbecDCWConfig

    color_image: Out[Image]
    depth_image: Out[Image]
    pointcloud: Out[PointCloud2]
    camera_info: Out[CameraInfo]
    depth_camera_info: Out[CameraInfo]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pipeline: Pipeline | None = None
        self._sdk: dict[str, Any] | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._color_camera_info: CameraInfo | None = None
        self._depth_camera_info: CameraInfo | None = None
        self._depth_scale = _MM_TO_M
        self._alignment_active = False
        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None
        self._pointcloud_lock = threading.Lock()
        self._last_error = ""
        self._frames_received = 0

    @property
    def _camera_link(self) -> str:
        return f"{self.config.camera_name}_link"

    @property
    def _color_frame(self) -> str:
        return f"{self.config.camera_name}_color_frame"

    @property
    def _color_optical_frame(self) -> str:
        return f"{self.config.camera_name}_color_optical_frame"

    @property
    def _depth_frame(self) -> str:
        return f"{self.config.camera_name}_depth_frame"

    @property
    def _depth_optical_frame(self) -> str:
        return f"{self.config.camera_name}_depth_optical_frame"

    @rpc
    def start(self) -> None:
        if self._running.is_set():
            return
        if self.config.enable_pointcloud and not self.config.enable_depth:
            raise ValueError("enable_pointcloud requires enable_depth")
        if self.config.pointcloud_fps <= 0 or self.config.camera_info_fps <= 0:
            raise ValueError("pointcloud_fps and camera_info_fps must be positive")

        sdk = _load_sdk()
        pipeline = sdk["Pipeline"]()
        config: Config = sdk["Config"]()
        color_format = _enum_member(sdk["OBFormat"], self.config.color_format)
        color_profile = _select_video_profile(
            pipeline,
            sdk["OBSensorType"].COLOR_SENSOR,
            self.config.width,
            self.config.height,
            color_format,
            self.config.fps,
        )
        config.enable_stream(color_profile)

        depth_profile: VideoStreamProfile | None = None
        if self.config.enable_depth:
            depth_format = _enum_member(sdk["OBFormat"], self.config.depth_format)
            depth_profile = _select_video_profile(
                pipeline,
                sdk["OBSensorType"].DEPTH_SENSOR,
                self.config.depth_width,
                self.config.depth_height,
                depth_format,
                self.config.depth_fps,
            )
            config.enable_stream(depth_profile)
            self._alignment_active = self._configure_alignment(config, sdk["OBAlignMode"])

        try:
            pipeline.start(config)
        except Exception as exc:
            if not self._alignment_active:
                raise
            logger.warning(f"DCW D2C start failed; retrying without alignment: {exc}")
            config = sdk["Config"]()
            config.enable_stream(color_profile)
            if depth_profile is not None:
                config.enable_stream(depth_profile)
            pipeline.start(config)
            self._alignment_active = False
        self._sdk = sdk
        self._pipeline = pipeline
        self._build_camera_info(color_profile, depth_profile)
        self._last_error = ""
        self._frames_received = 0
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="orbbec-dcw-capture",
            daemon=True,
        )
        self._thread.start()

        if self.config.enable_pointcloud:
            self.register_disposable(
                backpressure(rx.interval(1.0 / self.config.pointcloud_fps)).subscribe(
                    on_next=lambda _: self._generate_pointcloud(),
                    on_error=lambda error: logger.error(f"DCW pointcloud timer failed: {error}"),
                )
            )
        self.register_disposable(
            rx.interval(1.0 / self.config.camera_info_fps).subscribe(
                on_next=lambda _: self._publish_camera_info(),
                on_error=lambda error: logger.error(f"DCW camera-info timer failed: {error}"),
            )
        )
        logger.info(
            f"Orbbec DCW started: color={self.config.width}x{self.config.height}"
            f"@{self.config.fps}, depth={self.config.enable_depth}, "
            f"aligned={self._alignment_active}"
        )

    def _configure_alignment(self, config: Config, align_mode: Any) -> bool:
        if not self.config.align_depth_to_color:
            return False
        # pyorbbecsdk renamed ALIGN_D2C_HW_MODE to HW_MODE in newer releases.
        mode = getattr(align_mode, "HW_MODE", None)
        if mode is None:
            mode = getattr(align_mode, "ALIGN_D2C_HW_MODE", None)
        if mode is None:
            logger.warning("Orbbec SDK has no hardware D2C alignment mode")
            return False
        try:
            config.set_align_mode(mode)
            if hasattr(config, "set_d2c_target_resolution"):
                config.set_d2c_target_resolution(self.config.width, self.config.height)
            return True
        except Exception as exc:
            logger.warning(f"DCW hardware D2C alignment unavailable: {exc}")
            return False

    def _build_camera_info(
        self,
        color_profile: VideoStreamProfile,
        depth_profile: VideoStreamProfile | None,
    ) -> None:
        self._color_camera_info = _camera_info_from_intrinsics(
            color_profile.get_intrinsic(),
            color_profile.get_distortion(),
            self._color_optical_frame,
            fallback_width=self.config.width,
            fallback_height=self.config.height,
        )
        if depth_profile is None:
            self._depth_camera_info = None
        elif self._alignment_active:
            self._depth_camera_info = self._color_camera_info.with_ts(time.time())
        else:
            self._depth_camera_info = _camera_info_from_intrinsics(
                depth_profile.get_intrinsic(),
                depth_profile.get_distortion(),
                self._depth_optical_frame,
                fallback_width=self.config.depth_width,
                fallback_height=self.config.depth_height,
            )

    def _capture_loop(self) -> None:
        assert self._pipeline is not None
        assert self._sdk is not None
        consecutive_errors = 0
        while self._running.is_set():
            try:
                frames = self._pipeline.wait_for_frames(self.config.frame_timeout_ms)
                if not frames:
                    continue
                ts = time.time()
                color = self._publish_color(frames.get_color_frame(), ts)
                depth = (
                    self._publish_depth(frames.get_depth_frame(), ts)
                    if self.config.enable_depth
                    else None
                )
                if self.config.enable_pointcloud and color is not None and depth is not None:
                    with self._pointcloud_lock:
                        self._latest_color = color
                        self._latest_depth = depth
                self._publish_tf(ts)
                self._frames_received += 1
                consecutive_errors = 0
            except Exception as exc:
                if not self._running.is_set():
                    break
                consecutive_errors += 1
                self._last_error = str(exc)
                logger.warning(f"DCW capture error ({consecutive_errors}): {exc}")
                if consecutive_errors >= 5:
                    logger.error("DCW capture stopped after five consecutive errors")
                    self._running.clear()
                    break
                time.sleep(0.05)

    def _publish_color(self, frame: ColorFrame | None, ts: float) -> Image | None:
        if frame is None or self._sdk is None:
            return None
        rgb = _decode_color_frame(frame, self._sdk["OBFormat"])
        image = Image(
            data=rgb,
            format=ImageFormat.RGB,
            frame_id=self._color_optical_frame,
            ts=ts,
        )
        self.color_image.publish(image)
        return image

    def _publish_depth(self, frame: DepthFrame | None, ts: float) -> Image | None:
        if frame is None:
            return None
        depth, self._depth_scale = _decode_depth_frame(frame)
        frame_id = (
            self._color_optical_frame if self._alignment_active else self._depth_optical_frame
        )
        image = Image(
            data=depth,
            format=ImageFormat.DEPTH16,
            frame_id=frame_id,
            ts=ts,
        )
        self.depth_image.publish(image)
        return image

    def _publish_camera_info(self) -> None:
        ts = time.time()
        if self._color_camera_info is not None:
            self.camera_info.publish(self._color_camera_info.with_ts(ts))
        if self._depth_camera_info is not None:
            self.depth_camera_info.publish(self._depth_camera_info.with_ts(ts))

    def _publish_tf(self, ts: float) -> None:
        transforms: list[Transform] = []
        mount = self.config.base_transform
        if mount is not None:
            parent_frame = mount.frame_id or self.config.base_frame_id
            transforms.append(
                Transform(
                    translation=mount.translation,
                    rotation=mount.rotation,
                    frame_id=parent_frame,
                    child_frame_id=self._camera_link,
                    ts=ts,
                )
            )
        transforms.extend(
            [
                Transform(
                    translation=Vector3(0.0, 0.0, 0.0),
                    rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                    frame_id=self._camera_link,
                    child_frame_id=self._depth_frame,
                    ts=ts,
                ),
                Transform(
                    translation=Vector3(0.0, 0.0, 0.0),
                    rotation=OPTICAL_ROTATION,
                    frame_id=self._depth_frame,
                    child_frame_id=self._depth_optical_frame,
                    ts=ts,
                ),
                Transform(
                    translation=Vector3(0.0, 0.0, 0.0),
                    rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                    frame_id=self._camera_link,
                    child_frame_id=self._color_frame,
                    ts=ts,
                ),
                Transform(
                    translation=Vector3(0.0, 0.0, 0.0),
                    rotation=OPTICAL_ROTATION,
                    frame_id=self._color_frame,
                    child_frame_id=self._color_optical_frame,
                    ts=ts,
                ),
            ]
        )
        self.tf.publish(*transforms)

    def _generate_pointcloud(self) -> None:
        if not self._alignment_active:
            return
        with self._pointcloud_lock:
            color = self._latest_color
            depth = self._latest_depth
        if color is None or depth is None or self._color_camera_info is None:
            return
        try:
            pointcloud = PointCloud2.from_rgbd(
                color_image=color,
                depth_image=depth,
                camera_info=self._color_camera_info,
                depth_scale=self._depth_scale,
            )
            if self.config.pointcloud_voxel_size > 0:
                pointcloud = pointcloud.voxel_downsample(self.config.pointcloud_voxel_size)
            self.pointcloud.publish(pointcloud)
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning(f"DCW pointcloud generation failed: {exc}")

    @rpc
    def stop(self) -> None:
        self._running.clear()
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._thread = None
        self._sdk = None
        self._latest_color = None
        self._latest_depth = None
        super().stop()

    @rpc
    def get_color_camera_info(self) -> CameraInfo | None:
        return self._color_camera_info

    @rpc
    def get_depth_camera_info(self) -> CameraInfo | None:
        return self._depth_camera_info

    @rpc
    def get_depth_scale(self) -> float:
        return self._depth_scale

    @rpc
    def get_status(self) -> dict[str, Any]:
        """Return a compact health snapshot for diagnostics."""
        return {
            "running": self._running.is_set(),
            "frames_received": self._frames_received,
            "alignment_active": self._alignment_active,
            "depth_scale_m": self._depth_scale,
            "last_error": self._last_error,
        }
