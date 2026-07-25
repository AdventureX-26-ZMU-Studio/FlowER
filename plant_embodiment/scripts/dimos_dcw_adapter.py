"""
Orbbec DaBai DCW — dimos Camera Adapter

模仿 RealSense adapter (dimos/hardware/sensors/camera/realsense/camera.py) 编写。

用法:
  from dimos_dcw_adapter import OrbbecDCWCamera

  camera = dimos.deploy(OrbbecDCWCamera, serial_number=None)
  camera.color_image.transport = make_transport("/dcw/color", Image)
  camera.depth_image.transport = make_transport("/dcw/depth", Image)
"""

from __future__ import annotations

import atexit
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pydantic import Field
import reactivex as rx
from scipy.spatial.transform import Rotation

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.core.transport_factory import make_transport
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
from dimos.utils.reactive import backpressure

try:
    from pyorbbecsdk import (
        Config as OBConfig,
        OBFormat,
        OBSensorType,
        Pipeline,
        OBAlignMode,
    )
except ImportError:
    # Fallback for when running outside GX10
    OBConfig = None
    OBSensorType = None
    Pipeline = None
    OBAlignMode = None


def default_base_transform() -> Transform:
    return Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )


class OrbbecDCWConfig(ModuleConfig, DepthCameraConfig):
    """DaBai DCW 相机配置"""

    width: int = 640
    height: int = 360  # DCW D2C 仅支持 640x360
    fps: int = 30
    camera_name: str = "dcw"
    base_frame_id: str = "base_link"
    base_transform: Transform | None = Field(default_factory=default_base_transform)
    align_depth_to_color: bool = True  # D2C 硬件对齐
    enable_depth: bool = True
    enable_ir: bool = False  # IR 流默认关闭（省带宽）
    enable_pointcloud: bool = False
    pointcloud_fps: float = 5.0
    camera_info_fps: float = 1.0
    serial_number: str | None = None  # DCW 序列号（可选）


class OrbbecDCWCamera(DepthCameraHardware, Module, perception.DepthCamera):
    """
    Orbbec DaBai DCW 深度相机 dimos Module

    输出:
      - color_image:   RGB 图像 (640x360, 30fps)
      - depth_image:   Depth 图像 (640x360, 30fps, 对齐到 color)
      - camera_info:   彩色相机内参
      - depth_camera_info: 深度相机内参
      - pointcloud:    点云 (可选)
    """

    config: OrbbecDCWConfig

    color_image: Out[Image]
    depth_image: Out[Image]
    pointcloud: Out[PointCloud2]
    camera_info: Out[CameraInfo]
    depth_camera_info: Out[CameraInfo]

    # ── frame_id 属性 ──
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pipeline: Pipeline | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._color_camera_info: CameraInfo | None = None
        self._depth_camera_info: CameraInfo | None = None
        self._depth_scale: float = 0.001  # 默认 1mm
        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None
        self._pc_lock = threading.Lock()

    # ═══════════════════════════════════════════════
    # 启动 / 停止
    # ═══════════════════════════════════════════════

    @rpc
    def start(self) -> None:
        if Pipeline is None:
            raise RuntimeError("pyorbbecsdk not installed on this machine")

        self._pipeline = Pipeline()
        config = OBConfig()

        # 彩色流
        color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            self.config.width, self.config.height, OBFormat.MJPG, self.config.fps
        )
        config.enable_stream(color_profile)

        # 深度流
        if self.config.enable_depth:
            depth_profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            config.enable_stream(depth_profile)

        # IR 流（可选）
        if self.config.enable_ir:
            ir_profiles = self._pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
            ir_profile = ir_profiles.get_default_video_stream_profile()
            config.enable_stream(ir_profile)

        # D2C 硬件对齐
        if self.config.align_depth_to_color and self.config.enable_depth:
            config.set_align_mode(OBAlignMode.ALIGN_D2C_HW_MODE)

        self._pipeline.start(config)
        self._build_camera_info()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # 点云定时器
        if self.config.enable_pointcloud and self.config.enable_depth:
            interval = 1.0 / self.config.pointcloud_fps
            self.register_disposable(
                backpressure(rx.interval(interval)).subscribe(
                    on_next=lambda _: self._generate_pointcloud(),
                    on_error=lambda e: print(f"Pointcloud error: {e}"),
                )
            )

        # CameraInfo 定时发布
        interval = 1.0 / self.config.camera_info_fps
        self.register_disposable(
            rx.interval(interval).subscribe(
                on_next=lambda _: self._publish_camera_info(),
                on_error=lambda e: print(f"CameraInfo error: {e}"),
            )
        )

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            self._thread = None
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

    # ── 采集循环 ──

    def _capture_loop(self) -> None:
        while self._running and self._pipeline is not None:
            try:
                frames = self._pipeline.wait_for_frames(1000)
            except Exception:
                break
            if not frames:
                continue

            ts = time.time()

            # Color: MJPG → decode → RGB
            color_frame = frames.get_color_frame()
            color_img = None
            if color_frame:
                raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if bgr is not None and bgr.size > 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    color_img = Image(
                        data=rgb,
                        format=ImageFormat.RGB,
                        frame_id=self._color_optical_frame,
                        ts=ts,
                    )
                    self.color_image.publish(color_img)

            # Depth
            depth_frame = frames.get_depth_frame()
            depth_img = None
            if depth_frame:
                raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    depth_frame.get_height(), depth_frame.get_width()
                )
                depth_mm = raw.astype(np.float32) * depth_frame.get_depth_scale()
                self._depth_scale = depth_frame.get_depth_scale()
                depth_frame_id = (
                    self._color_optical_frame
                    if self.config.align_depth_to_color
                    else self._depth_optical_frame
                )
                depth_img = Image(
                    data=depth_mm.astype(np.float32),
                    format=ImageFormat.DEPTH32,
                    frame_id=depth_frame_id,
                    ts=ts,
                )
                self.depth_image.publish(depth_img)

            # 缓存最新帧用于点云
            if self.config.enable_pointcloud and color_img and depth_img:
                with self._pc_lock:
                    self._latest_color = color_img
                    self._latest_depth = depth_img

            # TF
            self._publish_tf(ts)

    # ── CameraInfo ──

    def _build_camera_info(self) -> None:
        """从 SDK 获取相机内参"""
        if self._pipeline is None:
            return
        try:
            param = self._pipeline.get_camera_param()
            # 彩色内参
            self._color_camera_info = CameraInfo(
                height=self.config.height,
                width=self.config.width,
                distortion_model="plumb_bob",
                D=list(param.color_distortion) if hasattr(param, "color_distortion") else [],
                K=list(param.color_intrinsic) if hasattr(param, "color_intrinsic") else [],
                P=list(param.color_intrinsic) if hasattr(param, "color_intrinsic") else [],
                frame_id=self._color_optical_frame,
            )
            # 深度内参
            if self.config.align_depth_to_color:
                self._depth_camera_info = self._color_camera_info
            else:
                self._depth_camera_info = CameraInfo(
                    height=self.config.height,
                    width=self.config.width,
                    distortion_model="plumb_bob",
                    D=list(param.depth_distortion) if hasattr(param, "depth_distortion") else [],
                    K=list(param.depth_intrinsic) if hasattr(param, "depth_intrinsic") else [],
                    P=list(param.depth_intrinsic) if hasattr(param, "depth_intrinsic") else [],
                    frame_id=self._depth_optical_frame,
                )
        except Exception as e:
            print(f"Failed to get camera params: {e}")

    def _publish_camera_info(self) -> None:
        ts = time.time()
        if self._color_camera_info:
            self._color_camera_info.ts = ts
            self.camera_info.publish(self._color_camera_info)
        if self._depth_camera_info and not self.config.align_depth_to_color:
            self._depth_camera_info.ts = ts
            self.depth_camera_info.publish(self._depth_camera_info)

    # ── TF ──

    def _publish_tf(self, ts: float) -> None:
        transforms = []
        # base_link → camera_link
        if self.config.base_transform is not None:
            transforms.append(Transform(
                translation=self.config.base_transform.translation,
                rotation=self.config.base_transform.rotation,
                frame_id=self.config.base_frame_id,
                child_frame_id=self._camera_link,
                ts=ts,
            ))
        # camera_link → depth_frame
        transforms.append(Transform(
            translation=Vector3(0, 0, 0),
            rotation=Quaternion(0, 0, 0, 1),
            frame_id=self._camera_link,
            child_frame_id=self._depth_frame,
            ts=ts,
        ))
        # depth_frame → depth_optical
        transforms.append(Transform(
            translation=Vector3(0, 0, 0),
            rotation=OPTICAL_ROTATION,
            frame_id=self._depth_frame,
            child_frame_id=self._depth_optical_frame,
            ts=ts,
        ))
        # camera_link → color_frame
        transforms.append(Transform(
            translation=Vector3(0, 0, 0),
            rotation=Quaternion(0, 0, 0, 1),
            frame_id=self._camera_link,
            child_frame_id=self._color_frame,
            ts=ts,
        ))
        # color_frame → color_optical
        transforms.append(Transform(
            translation=Vector3(0, 0, 0),
            rotation=OPTICAL_ROTATION,
            frame_id=self._color_frame,
            child_frame_id=self._color_optical_frame,
            ts=ts,
        ))
        self.tf.publish(*transforms)

    # ── 点云 ──

    def _generate_pointcloud(self) -> None:
        with self._pc_lock:
            color = self._latest_color
            depth = self._latest_depth
        if color is None or depth is None or self._color_camera_info is None:
            return
        try:
            pcd = PointCloud2.from_rgbd(
                color_image=color,
                depth_image=depth,
                camera_info=self._color_camera_info,
                depth_scale=self._depth_scale,
            )
            self.pointcloud.publish(pcd.voxel_downsample(0.005))
        except Exception as e:
            print(f"Pointcloud error: {e}")


# ── 独立测试 ──
def main() -> None:
    dimos = ModuleCoordinator()
    dimos.start()

    camera = dimos.deploy(
        OrbbecDCWCamera,
        enable_depth=True,
        align_depth_to_color=True,
        enable_pointcloud=False,
    )
    camera.color_image.transport = make_transport("/dcw/color", Image)
    camera.depth_image.transport = make_transport("/dcw/depth", Image)
    camera.camera_info.transport = make_transport("/dcw/color_info", CameraInfo)
    camera.depth_camera_info.transport = make_transport("/dcw/depth_info", CameraInfo)

    def cleanup():
        try:
            dimos.stop()
        except Exception:
            pass
    atexit.register(cleanup)
    dimos.start_all_modules()

    try:
        while True:
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        atexit.unregister(cleanup)
        cleanup()


if __name__ == "__main__":
    main()
