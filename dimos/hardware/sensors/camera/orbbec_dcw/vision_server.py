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

"""Standalone Orbbec DCW visual-perception web service.

This service owns the camera directly and therefore must not run at the same
time as :class:`OrbbecDCWCamera`. It serves annotated color and depth MJPEG
streams plus JSON status on port 8890 by default.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import signal
import threading
import time
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from dimos.hardware.sensors.camera.orbbec_dcw.camera import (
    _decode_color_frame,
    _decode_depth_frame,
    _enum_member,
    _load_sdk,
    _select_video_profile,
)
from dimos.hardware.sensors.camera.orbbec_dcw.depth_cluster_engine import (
    DepthClusterConfig,
    DepthClusterEngine,
)
from dimos.hardware.sensors.camera.orbbec_dcw.face_follow import FaceFollowController
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_HTTP_PORT = 8890
_FINGER_TIPS = (8, 12, 16, 20)


@dataclass
class VisionStatus:
    camera_connected: bool = False
    mediapipe_ready: bool = False
    face_count: int = 0
    face_center_x: float | None = None
    face_center_y: float | None = None
    expression: str = ""
    gesture: str = ""
    flipped: bool = False
    depth_aligned: bool = False
    depth_center_mm: float = 0.0
    face_depth_mm: float = 0.0
    fps: float = 0.0
    frame: int = 0
    obstacle_count: int = 0
    obstacles: list[dict[str, Any]] = field(default_factory=list)
    tracking_enabled: bool = False
    tracking_state: str = "disabled"
    tracking_error_x: float = 0.0
    tracking_error_y: float = 0.0
    tracking_yaw_speed: float = 0.0
    tracking_pitch_speed: float = 0.0
    tracking_linear_speed: float = 0.0
    tracking_depth_error_m: float = 0.0
    tracking_enabled_seconds: float = 0.0
    last_error: str = ""


def _landmark_y(landmarks: Any, index: int) -> float:
    return float(landmarks[index].y)


def classify_expression(landmarks: Any) -> str:
    """Classify a coarse facial expression from MediaPipe face landmarks."""
    ys = [_landmark_y(landmarks, index) for index in range(len(landmarks))]
    face_height = max(ys) - min(ys)
    if face_height <= 1e-6:
        return "neutral"

    mouth_open = (_landmark_y(landmarks, 14) - _landmark_y(landmarks, 13)) / face_height
    mouth_center = (_landmark_y(landmarks, 13) + _landmark_y(landmarks, 14)) / 2
    mouth_corners = (_landmark_y(landmarks, 61) + _landmark_y(landmarks, 291)) / 2
    smile = (mouth_center - mouth_corners) / face_height
    left_eye = np.mean([_landmark_y(landmarks, i) for i in (33, 160, 159, 158, 157, 173)])
    right_eye = np.mean([_landmark_y(landmarks, i) for i in (263, 387, 386, 385, 384, 398)])
    left_brow = np.mean([_landmark_y(landmarks, i) for i in (46, 53, 52, 65)])
    right_brow = np.mean([_landmark_y(landmarks, i) for i in (276, 283, 282, 293)])
    brow_gap = ((left_eye - left_brow) + (right_eye - right_brow)) / 2 / face_height

    if mouth_open > 0.08:
        return "surprised"
    if smile > 0.02:
        return "happy"
    if brow_gap < -0.02:
        return "angry"
    if mouth_open > 0.03 and smile < -0.01:
        return "sad"
    return "neutral"


def classify_gesture(hand_landmarks: Any) -> str:
    """Classify a small set of static hand gestures."""
    landmarks = hand_landmarks.landmark
    extended = sum(
        _landmark_y(landmarks, tip) < _landmark_y(landmarks, tip - 2) for tip in _FINGER_TIPS
    )
    thumb_extended = abs(float(landmarks[4].x) - float(landmarks[3].x)) > 0.05
    count = extended + int(thumb_extended)
    if count == 0:
        return "fist"
    if count >= 4:
        return "palm"
    if count == 1 and _landmark_y(landmarks, 8) < _landmark_y(landmarks, 6):
        return "point"
    if count == 2:
        return "peace"
    return f"hand_{count}"


def depth_preview(depth: np.ndarray, depth_scale_m: float) -> tuple[np.ndarray, float]:
    """Create a colored depth preview and return its center distance in mm."""
    depth_mm = depth.astype(np.float32) * depth_scale_m * 1000.0
    valid = np.isfinite(depth_mm) & (depth_mm >= 20.0) & (depth_mm <= 5000.0)
    gray = np.clip(depth_mm * (255.0 / 5000.0), 0.0, 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    center_y, center_x = depth.shape[0] // 2, depth.shape[1] // 2
    sample = depth_mm[
        max(0, center_y - 2) : center_y + 3,
        max(0, center_x - 2) : center_x + 3,
    ]
    sample = sample[np.isfinite(sample) & (sample > 0)]
    center_mm = float(np.median(sample)) if sample.size else 0.0
    cv2.drawMarker(
        preview,
        (center_x, center_y),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        20,
        2,
    )
    cv2.putText(
        preview,
        f"{center_mm:.0f} mm",
        (center_x + 15, center_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return preview, center_mm


def depth_at_normalized_point(
    depth: np.ndarray,
    depth_scale_m: float,
    point: tuple[float, float] | None,
    *,
    radius_fraction: float = 0.04,
) -> float | None:
    """Return robust aligned depth at a normalized color-image point in metres."""
    if point is None or depth.ndim != 2 or depth.size == 0 or depth_scale_m <= 0.0:
        return None
    height, width = depth.shape
    x = min(width - 1, max(0, round(float(point[0]) * (width - 1))))
    y = min(height - 1, max(0, round(float(point[1]) * (height - 1))))
    radius_x = max(2, round(width * radius_fraction))
    radius_y = max(2, round(height * radius_fraction))
    sample = depth[
        max(0, y - radius_y) : min(height, y + radius_y + 1),
        max(0, x - radius_x) : min(width, x + radius_x + 1),
    ].astype(np.float32)
    sample_m = sample * depth_scale_m
    valid = sample_m[
        np.isfinite(sample_m) & (sample_m >= 0.20) & (sample_m <= 4.0)
    ]
    if valid.size < 9:
        return None
    return float(np.median(valid))


def _configure_depth_alignment(config: Any, sdk: dict[str, Any]) -> bool:
    """Enable hardware depth-to-color alignment when the Orbbec SDK supports it."""
    align_mode = sdk.get("OBAlignMode")
    if align_mode is None:
        return False
    mode = getattr(align_mode, "HW_MODE", None)
    if mode is None:
        mode = getattr(align_mode, "ALIGN_D2C_HW_MODE", None)
    if mode is None:
        return False
    try:
        config.set_align_mode(mode)
        if hasattr(config, "set_d2c_target_resolution"):
            config.set_d2c_target_resolution(640, 360)
        return True
    except Exception as exc:
        logger.warning(f"DCW hardware D2C alignment unavailable: {exc}")
        return False


class MediaPipeRecognizer:
    """MediaPipe face/hand recognizer with stable automatic 180-degree rotation."""

    def __init__(self, *, auto_flip: bool, sticky_frames: int = 60) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for visual recognition. Install a Python 3.12 "
                "aarch64 MediaPipe wheel before starting the service."
            ) from exc
        if not hasattr(mp, "solutions"):
            raise RuntimeError("This service requires the MediaPipe Solutions API")

        self._mp = mp
        self._face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.35,
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._auto_flip = auto_flip
        self._sticky_frames = sticky_frames
        self._sticky_remaining = 0
        self.flipped = False
        self.face_center: tuple[float, float] | None = None

    def close(self) -> None:
        self._face_detector.close()
        self._face_mesh.close()
        self._hands.close()

    def _orientation_score(self, rgb: np.ndarray, flipped: bool) -> float:
        candidate = cv2.rotate(rgb, cv2.ROTATE_180) if flipped else rgb
        result = self._face_detector.process(candidate)
        if not result or not result.detections:
            return 0.0
        return float(np.mean([detection.score[0] for detection in result.detections]))

    def _update_orientation(self, rgb: np.ndarray) -> None:
        if not self._auto_flip:
            return
        if self._sticky_remaining > 0:
            self._sticky_remaining -= 1
            return
        upright_score = self._orientation_score(rgb, False)
        flipped_score = self._orientation_score(rgb, True)
        requested = flipped_score > upright_score + 0.08
        has_face = max(upright_score, flipped_score) > 0
        if has_face and requested != self.flipped and self._sticky_remaining == 0:
            self.flipped = requested
            self._sticky_remaining = self._sticky_frames

    def process(self, bgr: np.ndarray) -> tuple[np.ndarray, int, str, str]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._update_orientation(rgb)
        if self.flipped:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        face_count = 0
        expression = ""
        self.face_center = None
        mesh_result = self._face_mesh.process(rgb)
        if mesh_result and mesh_result.multi_face_landmarks:
            height, width = bgr.shape[:2]
            largest_area = -1.0
            for face in mesh_result.multi_face_landmarks:
                face_count += 1
                landmarks = face.landmark
                xs = [float(point.x) for point in landmarks]
                ys = [float(point.y) for point in landmarks]
                x1 = max(0, int(min(xs) * width))
                y1 = max(0, int(min(ys) * height))
                x2 = min(width - 1, int(max(xs) * width))
                y2 = min(height - 1, int(max(ys) * height))
                area = max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
                if area > largest_area:
                    largest_area = area
                    self.face_center = (
                        min(1.0, max(0.0, (min(xs) + max(xs)) / 2)),
                        min(1.0, max(0.0, (min(ys) + max(ys)) / 2)),
                    )
                    expression = classify_expression(landmarks)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    bgr,
                    expression,
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                self._mp.solutions.drawing_utils.draw_landmarks(
                    bgr,
                    face,
                    self._mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                        self._mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                    ),
                )
            if self.face_center is not None:
                center = (
                    int(self.face_center[0] * width),
                    int(self.face_center[1] * height),
                )
                frame_center = (width // 2, height // 2)
                cv2.line(bgr, frame_center, center, (0, 255, 255), 2)
                cv2.circle(bgr, center, 7, (0, 255, 255), 2)
                cv2.drawMarker(
                    bgr,
                    frame_center,
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    24,
                    2,
                )

        gesture = ""
        hand_result = self._hands.process(rgb)
        if hand_result and hand_result.multi_hand_landmarks:
            gestures = []
            for hand in hand_result.multi_hand_landmarks:
                gestures.append(classify_gesture(hand))
                self._mp.solutions.drawing_utils.draw_landmarks(
                    bgr,
                    hand,
                    self._mp.solutions.hands.HAND_CONNECTIONS,
                )
            gesture = ",".join(gestures)

        marker = "ROTATED" if self.flipped else "UP"
        cv2.putText(
            bgr,
            marker,
            (max(5, bgr.shape[1] - 90), 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        return bgr, face_count, expression, gesture


class FrameStore:
    """Thread-safe JPEG and status storage used by capture and HTTP threads."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frames: dict[str, bytes] = {"color": b"", "depth": b""}
        self.sequence: dict[str, int] = {"color": 0, "depth": 0}
        self.status = VisionStatus()
        self.depth_raw = b""
        self.depth_width = 0
        self.depth_height = 0
        self.depth_scale_m = 0.0

    def update_frame(self, name: str, data: bytes) -> None:
        with self.condition:
            self.frames[name] = data
            self.sequence[name] += 1
            self.condition.notify_all()

    def update_status(self, **values: Any) -> None:
        unknown = set(values) - set(VisionStatus.__dataclass_fields__)
        if unknown:
            raise AttributeError(f"Unknown vision status fields: {sorted(unknown)}")
        with self.condition:
            for name, value in values.items():
                setattr(self.status, name, value)

    def update_depth(self, depth: np.ndarray, depth_scale_m: float) -> None:
        if depth.ndim != 2:
            raise ValueError("raw depth must be a 2D image")
        raw = np.ascontiguousarray(depth, dtype="<u2")
        with self.condition:
            self.depth_raw = raw.tobytes()
            self.depth_height, self.depth_width = raw.shape
            self.depth_scale_m = float(depth_scale_m)

    def status_snapshot(self) -> dict[str, Any]:
        with self.condition:
            return asdict(self.status)

    def status_json(self) -> bytes:
        return json.dumps(self.status_snapshot(), ensure_ascii=False).encode()

    def obstacles_json(self) -> bytes:
        with self.condition:
            return json.dumps(self.status.obstacles, ensure_ascii=False).encode()

    def depth_snapshot(self) -> tuple[bytes, int, int, float]:
        with self.condition:
            return (
                self.depth_raw,
                self.depth_width,
                self.depth_height,
                self.depth_scale_m,
            )

    def wait_for_frame(
        self,
        name: str,
        after_sequence: int,
        timeout: float = 1.0,
    ) -> tuple[int, bytes]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence[name] > after_sequence,
                timeout=timeout,
            )
            return self.sequence[name], self.frames[name]


class DCWVisionService:
    """Own the DCW pipeline, run inference, and serve HTTP."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        fps: int,
        inference_every: int,
        auto_flip: bool,
        enable_depth: bool,
    ) -> None:
        if fps <= 0 or inference_every <= 0:
            raise ValueError("fps and inference_every must be positive")
        self.host = host
        self.port = port
        self.fps = fps
        self.inference_every = inference_every
        self.auto_flip = auto_flip
        self.enable_depth = enable_depth
        self.store = FrameStore()
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._recognizer: MediaPipeRecognizer | None = None
        self._recognizer_error = ""
        self._depth_aligned = False
        self._face_follow = FaceFollowController()
        self._cluster = DepthClusterEngine(DepthClusterConfig())
        self._latest_obstacles: list[dict] = []

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler_class())
        try:
            self._recognizer = MediaPipeRecognizer(auto_flip=self.auto_flip)
            self.store.update_status(mediapipe_ready=True)
        except RuntimeError as exc:
            self._recognizer_error = str(exc)
            self.store.update_status(
                mediapipe_ready=False,
                last_error=self._recognizer_error,
            )
            logger.warning(
                f"DCW vision is starting without MediaPipe recognition: {self._recognizer_error}"
            )
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="dcw-vision-capture",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info(f"DCW vision web service listening on http://{self.host}:{self.port}")
        try:
            self._httpd.serve_forever(poll_interval=0.2)
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._face_follow.stop()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None

    def _capture_loop(self) -> None:
        sdk = _load_sdk()
        pipeline = sdk["Pipeline"]()
        config = sdk["Config"]()
        color_profile = _select_video_profile(
            pipeline,
            sdk["OBSensorType"].COLOR_SENSOR,
            640,
            360,
            sdk["OBFormat"].MJPG,
            self.fps,
        )
        config.enable_stream(color_profile)
        depth_profile: Any | None = None
        depth_intrinsic: Any | None = None
        if self.enable_depth:
            depth_profile = _select_video_profile(
                pipeline,
                sdk["OBSensorType"].DEPTH_SENSOR,
                640,
                360,
                _enum_member(sdk["OBFormat"], "Y12"),
                self.fps,
            )
            config.enable_stream(depth_profile)
            self._depth_aligned = _configure_depth_alignment(config, sdk)
            self.store.update_status(depth_aligned=self._depth_aligned)
            intrinsic_profile = color_profile if self._depth_aligned else depth_profile
            depth_intrinsic = intrinsic_profile.get_intrinsic()
        try:
            try:
                pipeline.start(config)
            except Exception as exc:
                if not self._depth_aligned or depth_profile is None:
                    raise
                logger.warning(f"DCW D2C start failed; retrying without alignment: {exc}")
                config = sdk["Config"]()
                config.enable_stream(color_profile)
                config.enable_stream(depth_profile)
                pipeline.start(config)
                self._depth_aligned = False
                depth_intrinsic = depth_profile.get_intrinsic()
                self.store.update_status(depth_aligned=False)
            self.store.update_status(camera_connected=True)
            self._capture_frames(
                pipeline,
                sdk["OBFormat"],
                depth_intrinsic=depth_intrinsic,
            )
        except Exception as exc:
            self.store.update_status(camera_connected=False, last_error=str(exc))
            logger.error(f"DCW vision capture failed: {exc}")
        finally:
            tracking = self._face_follow.set_enabled(False)
            try:
                pipeline.stop()
            except Exception:
                pass
            self.store.update_status(
                camera_connected=False,
                tracking_enabled=tracking["enabled"],
                tracking_state=tracking["state"],
                tracking_linear_speed=0.0,
                tracking_pitch_speed=0.0,
                tracking_yaw_speed=0.0,
            )

    def _capture_frames(
        self,
        pipeline: Any,
        ob_format: Any,
        *,
        depth_intrinsic: Any | None,
    ) -> None:
        frame_index = 0
        fps_count = 0
        fps_value = 0.0
        fps_started = time.monotonic()
        last_face_count = 0
        last_face_center: tuple[float, float] | None = None
        last_expression = ""
        last_gesture = ""
        while not self._stop.is_set():
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            frame_index += 1
            fps_count += 1
            bgr = cv2.cvtColor(_decode_color_frame(color_frame, ob_format), cv2.COLOR_RGB2BGR)
            if frame_index % self.inference_every == 0 and self._recognizer is not None:
                bgr, last_face_count, last_expression, last_gesture = self._recognizer.process(bgr)
                last_face_center = self._recognizer.face_center
            elif self._recognizer is not None and self._recognizer.flipped:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)

            ok, color_jpeg = cv2.imencode(
                ".jpg",
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 70],
            )
            if ok:
                self.store.update_frame("color", color_jpeg.tobytes())

            center_mm = 0.0
            face_depth_m: float | None = None
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    depth, scale_m = _decode_depth_frame(depth_frame)
                    self.store.update_depth(depth, scale_m)
                    if self._recognizer is not None and self._recognizer.flipped:
                        depth = cv2.rotate(depth, cv2.ROTATE_180)
                    if self._depth_aligned:
                        face_depth_m = depth_at_normalized_point(
                            depth,
                            scale_m,
                            last_face_center,
                        )
                    preview, center_mm = depth_preview(depth, scale_m)
                    if last_face_center is not None:
                        marker_x = round(last_face_center[0] * (preview.shape[1] - 1))
                        marker_y = round(last_face_center[1] * (preview.shape[0] - 1))
                        cv2.drawMarker(
                            preview,
                            (marker_x, marker_y),
                            (255, 255, 255),
                            cv2.MARKER_DIAMOND,
                            24,
                            2,
                        )
                        depth_label = (
                            "invalid"
                            if face_depth_m is None
                            else f"face {face_depth_m * 1000.0:.0f} mm"
                        )
                        cv2.putText(
                            preview,
                            depth_label,
                            (max(5, marker_x - 45), max(18, marker_y - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                    ok, depth_jpeg = cv2.imencode(
                        ".jpg",
                        preview,
                        [cv2.IMWRITE_JPEG_QUALITY, 60],
                    )
                    if ok:
                        self.store.update_frame("depth", depth_jpeg.tobytes())
                    if depth_intrinsic is not None:
                        intrinsic_width = max(1, int(depth_intrinsic.width))
                        intrinsic_height = max(1, int(depth_intrinsic.height))
                        source_height, source_width = depth.shape
                        scale_x = source_width / intrinsic_width
                        scale_y = source_height / intrinsic_height
                        depth_m = depth.astype(np.float32) * scale_m
                        self._latest_obstacles = self._cluster.process(
                            depth_m,
                            float(depth_intrinsic.fx) * scale_x,
                            float(depth_intrinsic.fy) * scale_y,
                            float(depth_intrinsic.cx) * scale_x,
                            float(depth_intrinsic.cy) * scale_y,
                        )

            tracking = self._face_follow.update(last_face_center, face_depth_m)
            now = time.monotonic()
            if now - fps_started >= 1.0:
                fps_value = fps_count / (now - fps_started)
                fps_started = now
                fps_count = 0
            self.store.update_status(
                camera_connected=True,
                obstacle_count=len(self._latest_obstacles),
                obstacles=list(self._latest_obstacles),
                face_count=last_face_count,
                face_center_x=None if last_face_center is None else round(last_face_center[0], 4),
                face_center_y=None if last_face_center is None else round(last_face_center[1], 4),
                expression=last_expression,
                gesture=last_gesture,
                flipped=bool(self._recognizer and self._recognizer.flipped),
                depth_aligned=self._depth_aligned,
                depth_center_mm=round(center_mm, 1),
                face_depth_mm=(
                    0.0 if face_depth_m is None else round(face_depth_m * 1000.0, 1)
                ),
                fps=round(fps_value, 1),
                frame=frame_index,
                tracking_enabled=tracking["enabled"],
                tracking_state=tracking["state"],
                tracking_error_x=round(float(tracking["error_x"]), 4),
                tracking_error_y=round(float(tracking["error_y"]), 4),
                tracking_yaw_speed=round(float(tracking["yaw_speed"]), 4),
                tracking_pitch_speed=round(float(tracking["pitch_speed"]), 4),
                tracking_linear_speed=round(float(tracking["linear_speed"]), 4),
                tracking_depth_error_m=round(float(tracking["depth_error_m"]), 4),
                tracking_enabled_seconds=round(float(tracking["enabled_seconds"]), 1),
                last_error=str(tracking["last_error"] or self._recognizer_error),
            )

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        store = self.store
        face_follow = self._face_follow

        def recognizer_ready() -> bool:
            return self._recognizer is not None

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    self._send(HTTPStatus.OK, "text/html; charset=utf-8", _HTML.encode())
                    return
                if path in ("/s", "/status"):
                    self._send(HTTPStatus.OK, "application/json", store.status_json())
                    return
                if path == "/healthz":
                    status = store.status_snapshot()
                    code = (
                        HTTPStatus.OK
                        if status["camera_connected"]
                        else HTTPStatus.SERVICE_UNAVAILABLE
                    )
                    self._send(
                        code, "text/plain; charset=utf-8", b"ok\n" if code == 200 else b"down\n"
                    )
                    return
                stream_name = {
                    "/s/color": "color",
                    "/stream/color": "color",
                    "/s/depth": "depth",
                    "/stream/depth": "depth",
                }.get(path)
                if stream_name is not None:
                    self._stream(stream_name)
                    return
                if path == "/obstacles":
                    self._send(
                        HTTPStatus.OK,
                        "application/json",
                        store.obstacles_json(),
                    )
                    return
                if path == "/depth_raw":
                    raw, width, height, scale_m = store.depth_snapshot()
                    if not raw:
                        self._send(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "application/json",
                            b'{"error":"depth frame unavailable"}',
                        )
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("X-Depth-Width", str(width))
                    self.send_header("X-Depth-Height", str(height))
                    self.send_header("X-Depth-Scale-M", f"{scale_m:.9g}")
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/tracking":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                    self.send_error(HTTPStatus.FORBIDDEN, "Tracking control is localhost-only")
                    return
                try:
                    length = min(1024, int(self.headers.get("Content-Length", "0")))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    enabled = payload["enabled"]
                    if not isinstance(enabled, bool):
                        raise ValueError("enabled must be a boolean")
                    if enabled and not recognizer_ready():
                        raise RuntimeError("MediaPipe is not ready")
                    tracking = face_follow.set_enabled(enabled)
                    # Publish the accepted state before responding. The Viser
                    # client immediately reads /s after POST; waiting for the
                    # next camera frame creates a stale-state toggle race.
                    store.update_status(
                        tracking_enabled=tracking["enabled"],
                        tracking_state=tracking["state"],
                        tracking_error_x=round(float(tracking["error_x"]), 4),
                        tracking_error_y=round(float(tracking["error_y"]), 4),
                        tracking_yaw_speed=round(float(tracking["yaw_speed"]), 4),
                        tracking_pitch_speed=round(float(tracking["pitch_speed"]), 4),
                        tracking_linear_speed=round(float(tracking["linear_speed"]), 4),
                        tracking_depth_error_m=round(
                            float(tracking["depth_error_m"]),
                            4,
                        ),
                        tracking_enabled_seconds=round(
                            float(tracking["enabled_seconds"]),
                            1,
                        ),
                    )
                    body = json.dumps(tracking).encode()
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    self._send(
                        HTTPStatus.BAD_REQUEST,
                        "application/json",
                        json.dumps({"error": str(exc)}).encode(),
                    )
                    return
                self._send(HTTPStatus.OK, "application/json", body)

            def _send(self, code: int, content_type: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _stream(self, name: str) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                sequence = -1
                try:
                    while True:
                        sequence, jpeg = store.wait_for_frame(name, sequence)
                        if not jpeg:
                            continue
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler


_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DimOS · Orbbec DCW Vision</title>
<style>
body{margin:0;background:#101318;color:#e9eef6;font:14px system-ui,sans-serif}
header{padding:14px 20px;background:#171b22;border-bottom:1px solid #29313d}
main{display:grid;grid-template-columns:2fr 1fr;gap:14px;padding:14px}
.card{background:#171b22;border:1px solid #29313d;border-radius:10px;padding:10px}
img{display:block;width:100%;border-radius:6px;background:#080a0d}
#status{font-family:ui-monospace,monospace;white-space:pre-wrap;line-height:1.7}
@media(max-width:850px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><strong>Orbbec DCW · MediaPipe Vision</strong></header>
<main>
<section class="card"><img src="/s/color" alt="color stream"></section>
<section>
  <div class="card"><img src="/s/depth" alt="depth stream"></div>
  <div class="card" id="status">connecting…</div>
</section>
</main>
<script>
async function refresh(){
  try{
    const s=await fetch('/s',{cache:'no-store'}).then(r=>r.json());
    document.getElementById('status').textContent=
      `camera: ${s.camera_connected?'online':'offline'}\n`+
      `mediapipe: ${s.mediapipe_ready?'ready':'not ready'}\n`+
      `faces: ${s.face_count}\nexpression: ${s.expression||'-'}\n`+
      `face center: ${s.face_center_x??'-'}, ${s.face_center_y??'-'}\n`+
      `gesture: ${s.gesture||'-'}\nflipped: ${s.flipped}\n`+
      `tracking: ${s.tracking_enabled?'ON':'off'} (${s.tracking_state})\n`+
      `tracking error: ${s.tracking_error_x}, ${s.tracking_error_y}\n`+
      `tracking cmd: forward=${s.tracking_linear_speed}, pitch=${s.tracking_pitch_speed}, yaw=${s.tracking_yaw_speed}\n`+
      `depth aligned: ${s.depth_aligned}\nface depth: ${s.face_depth_mm} mm\n`+
      `depth error: ${s.tracking_depth_error_m} m\n`+
      `depth center: ${s.depth_center_mm} mm\nfps: ${s.fps}\nframe: ${s.frame}\n`+
      `error: ${s.last_error||'-'}`;
  }catch(e){document.getElementById('status').textContent='offline: '+e}
}
setInterval(refresh,400); refresh();
</script>
</body>
</html>
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orbbec DCW MediaPipe web service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--inference-every", type=int, default=2)
    parser.add_argument("--no-auto-flip", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    service = DCWVisionService(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_every=args.inference_every,
        auto_flip=not args.no_auto_flip,
        enable_depth=not args.no_depth,
    )

    def stop_service(_signum: int, _frame: Any) -> None:
        # Let serve_forever unwind before DCWVisionService.stop() calls
        # HTTPServer.shutdown(); calling shutdown directly from the signal
        # handler would run it on the serving thread and deadlock.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_service)
    signal.signal(signal.SIGINT, stop_service)
    try:
        service.start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
