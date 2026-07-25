from __future__ import annotations

import asyncio
import time
from typing import Any

from rePE.vision.models import CameraState, VisionFrame
from rePE.vision.camera import CameraBackend, b64_bytes


class OrbbecDCWBackend(CameraBackend):
    """DaBaiDCW backend using pyorbbecsdk v1.

    This class mirrors the known-good preview pattern from the local
    DCW-Camera-Driver project and keeps imports lazy for non-GX10 machines.
    """

    def __init__(self, *, wait_ms: int = 1000, jpeg_quality: int = 75,
                 width: int = 640, height: int = 360,
                 color_fps: int = 30, depth_fps: int = 15) -> None:
        self.wait_ms = wait_ms
        self.jpeg_quality = jpeg_quality
        self.width = width
        self.height = height
        self.color_fps = color_fps
        self.depth_fps = depth_fps
        self._pipeline: Any | None = None
        self._config: Any | None = None
        self._cv2: Any | None = None
        self._np: Any | None = None
        self._ob_format: Any | None = None

    async def start(self) -> None:
        await asyncio.to_thread(self._start_sync)

    async def stop(self) -> None:
        await asyncio.to_thread(self._stop_sync)

    async def next_frame(self) -> VisionFrame:
        return await asyncio.to_thread(self._next_frame_sync)

    def _start_sync(self) -> None:
        from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline

        pipeline, config = Pipeline(), Config()

        # Color stream: MJPG at configured resolution + fps
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            self.width, self.height, OBFormat.MJPG, self.color_fps
        )
        config.enable_stream(color_profile)

        # Depth stream: default format at configured fps
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        pipeline.start(config)
        self._pipeline = pipeline
        self._config = config

    def _stop_sync(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._config = None

    def _load_image_deps(self) -> tuple[Any, Any, Any]:
        if self._cv2 is None or self._np is None or self._ob_format is None:
            import cv2
            import numpy as np
            from pyorbbecsdk import OBFormat

            self._cv2 = cv2
            self._np = np
            self._ob_format = OBFormat
        return self._cv2, self._np, self._ob_format

    def _next_frame_sync(self) -> VisionFrame:
        if self._pipeline is None:
            return VisionFrame(
                camera_state=CameraState.STOPPED,
                unavailable_reason="DaBaiDCW pipeline is not started",
            )

        frames = self._pipeline.wait_for_frames(self.wait_ms)
        if not frames:
            return VisionFrame(
                camera_state=CameraState.RUNNING,
                unavailable_reason="DaBaiDCW frame timeout",
            )

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not (color and depth):
            return VisionFrame(
                camera_state=CameraState.RUNNING,
                unavailable_reason="DaBaiDCW color/depth frame missing",
            )

        bgr = self._color_image(color)
        depth_mm = self._depth_mm(depth)
        depth_shape = tuple(int(v) for v in depth_mm.shape)
        depth_uint16 = (depth_mm / depth.get_depth_scale()).astype(self._np.uint16)
        depth_b64 = b64_bytes(depth_uint16.tobytes())
        _, jpg = self._cv2.imencode(
            ".jpg",
            bgr,
            [int(self._cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        return VisionFrame(
            ts=time.time(),
            camera_state=CameraState.RUNNING,
            rgb_jpeg_b64=b64_bytes(jpg.tobytes()),
            depth_shape=depth_shape,
            depth_uint16_b64=depth_b64,
        )

    def _color_image(self, frame: Any) -> Any:
        cv2, np, ob_format = self._load_image_deps()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        h, w, fmt = frame.get_height(), frame.get_width(), frame.get_format()
        if fmt == ob_format.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == ob_format.RGB:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == ob_format.BGR:
            return data.reshape(h, w, 3)
        if fmt == ob_format.YUYV:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
        raise RuntimeError(f"Unsupported color format: {fmt}")

    def _depth_mm(self, frame: Any) -> Any:
        _, np, _ = self._load_image_deps()
        raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(
            frame.get_height(), frame.get_width()
        )
        return raw.astype(np.float32) * frame.get_depth_scale()



