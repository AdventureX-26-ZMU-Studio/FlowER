from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def now_s() -> float:
    return time.time()


class CameraState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class SensorPacket(BaseModel):
    node_id: str = "opi-3b"
    ts: float = Field(default_factory=now_s)
    temperature_c: float | None = None
    humidity_percent: float | None = None
    soil_raw: int | None = None
    soil_percent: float | None = None
    soil_calibrated: bool = False
    quality: Literal["ok", "stale", "error"] = "ok"
    errors: list[str] = Field(default_factory=list)


class FaceObservation(BaseModel):
    ts: float = Field(default_factory=now_s)
    present: bool = False
    depth_m: float | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    embedding: list[float] | None = None


class HandObservation(BaseModel):
    ts: float = Field(default_factory=now_s)
    handedness: Literal["Left", "Right", "Unknown"] = "Unknown"
    gesture: Literal["none", "victory", "wave", "open_palm"] = "none"
    confidence: float = 0.0
    x: float | None = None
    y: float | None = None


class VisionFrame(BaseModel):
    ts: float = Field(default_factory=now_s)
    frame_id: str = "dcw_link"
    camera_state: CameraState = CameraState.STOPPED
    rgb_jpeg_b64: str | None = None
    depth_shape: tuple[int, int] | None = None
    depth_uint16_b64: str | None = None
    faces: list[FaceObservation] = Field(default_factory=list)
    hands: list[HandObservation] = Field(default_factory=list)
    obstacles: list[dict[str, Any]] = Field(default_factory=list)
    unavailable_reason: str | None = None


class ManualLease(BaseModel):
    holder: str
    expires_at: float
    mode: Literal["observe", "manual"] = "manual"


class BehaviorIntent(str, Enum):
    NONE = "none"
    GREET = "greet"
    REGISTER_FACE = "register_face"
    SPRING = "spring"
    HOLD = "hold"


class PlantAffect(str, Enum):
    CALM = "calm"
    CURIOUS = "curious"
    HAPPY = "happy"
    SHY = "shy"
    DRY = "dry"
    WATERLOGGED = "waterlogged"
    WILTED = "wilted"

