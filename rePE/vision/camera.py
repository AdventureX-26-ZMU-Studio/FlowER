from __future__ import annotations

import asyncio
import base64
import time
from abc import ABC, abstractmethod

from rePE.vision.models import CameraState, FaceObservation, HandObservation, VisionFrame


class CameraBackend(ABC):
    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def next_frame(self) -> VisionFrame:
        raise NotImplementedError


class StubCameraBackend(CameraBackend):
    """Safe backend for development when DaBaiDCW is not attached."""

    def __init__(self) -> None:
        self.running = False
        self._tick = 0

    async def start(self) -> None:
        self.running = True

    async def stop(self) -> None:
        self.running = False

    async def next_frame(self) -> VisionFrame:
        await asyncio.sleep(0.1)
        self._tick += 1
        if not self.running:
            return VisionFrame(
                camera_state=CameraState.STOPPED,
                unavailable_reason="camera is released by Vision Hub",
            )

        # A transparent 1x1 PNG keeps WebSocket payloads small in stub mode.
        png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        face_depth = max(0.9, 2.6 - self._tick * 0.03)
        return VisionFrame(
            ts=time.time(),
            camera_state=CameraState.RUNNING,
            rgb_jpeg_b64=png,
            depth_shape=(480, 640),
            faces=[
                FaceObservation(
                    present=True,
                    depth_m=face_depth,
                    bbox_xyxy=(0.38, 0.2, 0.62, 0.55),
                    confidence=0.8,
                )
            ],
            hands=[HandObservation(handedness="Unknown", gesture="none")],
            obstacles=[
                {
                    "frame_id": "universe",
                    "center_xyz": [0.55, 0.0, 0.45],
                    "size_xyz": [0.28, 0.24, 0.65],
                    "confidence": 0.5,
                }
            ],
        )


def make_camera_backend(kind: str = "stub") -> CameraBackend:
    if kind == "stub":
        return StubCameraBackend()
    if kind == "orbbec":
        from rePE.vision.orbbec_dcw import OrbbecDCWBackend

        return OrbbecDCWBackend()
    raise RuntimeError(f"Unknown camera backend: {kind}")


def b64_bytes(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")
