"""Orbbec DCW backend using pyorbbecsdk. Minimal working version."""
import asyncio, time, base64
from typing import Any
from rePE.vision.models import VisionFrame, CameraState
from rePE.vision.camera import CameraBackend

def b64_bytes(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")

class OrbbecDCWBackend(CameraBackend):
    def __init__(self, *, wait_ms: int = 2000, jpeg_quality: int = 75,
                 width: int = 640, height: int = 360,
                 color_fps: int = 30, depth_fps: int = 15) -> None:
        self.wait_ms = wait_ms
        self.jpeg_quality = jpeg_quality
        self.width = width; self.height = height
        self.color_fps = color_fps; self.depth_fps = depth_fps
        self._pipeline = None; self._config = None

    async def start(self) -> None:
        await asyncio.to_thread(self._start_sync)

    async def stop(self) -> None:
        await asyncio.to_thread(self._stop_sync)

    async def next_frame(self) -> VisionFrame:
        return await asyncio.to_thread(self._next_frame_sync)

    def _start_sync(self) -> None:
        from pyorbbecsdk import Config, OBSensorType, Pipeline
        pipeline, config = Pipeline(), Config()
        config.enable_stream(pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile())
        config.enable_stream(pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile())
        pipeline.start(config)
        self._pipeline = pipeline; self._config = config

    def _stop_sync(self) -> None:
        if self._pipeline: self._pipeline.stop()
        self._pipeline = None; self._config = None

    def _next_frame_sync(self) -> VisionFrame:
        import cv2, numpy as np
        if self._pipeline is None:
            return VisionFrame(camera_state=CameraState.STOPPED, unavailable_reason="not started")
        frames = self._pipeline.wait_for_frames(self.wait_ms)
        if not frames:
            return VisionFrame(camera_state=CameraState.RUNNING, unavailable_reason="timeout")
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not (color and depth):
            return VisionFrame(camera_state=CameraState.RUNNING, unavailable_reason="missing")
        data = np.frombuffer(color.get_data(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return VisionFrame(camera_state=CameraState.RUNNING, unavailable_reason="decode fail")
        _, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(depth.get_height(), depth.get_width())
        return VisionFrame(ts=time.time(), camera_state=CameraState.RUNNING,
                           rgb_jpeg_b64=b64_bytes(jpg.tobytes()), depth_shape=(depth.get_height(), depth.get_width()))
