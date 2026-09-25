"""Direct Orbbec DaBai DCW RGB camera capture module."""
import base64
import ctypes
import threading
import time
from typing import Optional


# pyorbbecsdk 的 get_data() 在此设备上返回 stride=0 的 ndarray；
# 从其 PyCapsule 数据指针复制，才能取得实际 MJPEG 缓冲区。
_capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_get_pointer.restype = ctypes.c_void_p
_CAPSULE_NAME = b"frame_data_pointer"


def b64_bytes(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


class OrbbecCamera:
    """DaBai DCW RGB camera; one background reader, many safe consumers."""

    def __init__(self, width: int = 640, height: int = 360,
                 fps: int = 30, wait_ms: int = 2000) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.wait_ms = wait_ms
        self._pipeline = None
        self._running = False
        self._last_frame: Optional[bytes] = None
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Open RGB stream and start the single frame-capture thread."""
        from pyorbbecsdk import Config, OBSensorType, Pipeline

        with self._lock:
            if self._running:
                return
            pipeline = Pipeline()
            config = Config()
            profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            config.enable_stream(profiles.get_default_video_stream_profile())
            pipeline.start(config)
            self._pipeline = pipeline
            self._running = True
            self._error = None

        self._thread = threading.Thread(
            target=self._capture_loop, name="orbbec-rgb-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop capture then release the USB camera."""
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            if self._pipeline:
                self._pipeline.stop()
                self._pipeline = None

    def _capture_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
                pipeline = self._pipeline
            if pipeline is None:
                return

            try:
                frames = pipeline.wait_for_frames(self.wait_ms)
                if not frames:
                    continue
                color = frames.get_color_frame()
                if not color:
                    continue

                # Copy the native MJPEG payload while the SDK frame is alive.
                pointer_capsule = color.get_data_pointer()
                address = _capsule_get_pointer(pointer_capsule, _CAPSULE_NAME)
                jpg = ctypes.string_at(address, color.get_data_size())
                if len(jpg) < 100 or not jpg.startswith(b"\xff\xd8"):
                    raise RuntimeError("camera returned an invalid MJPEG frame")

                with self._lock:
                    self._last_frame = jpg
                    self._last_frame_time = time.time()
                    self._frame_count += 1
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                time.sleep(0.05)

    @property
    def is_running(self) -> bool:
        return self._running

    def capture_jpeg(self) -> Optional[bytes]:
        """Return the newest valid JPEG, without blocking capture."""
        with self._lock:
            return self._last_frame

    def capture_b64(self) -> Optional[str]:
        jpg = self.capture_jpeg()
        return b64_bytes(jpg) if jpg else None


_camera: Optional[OrbbecCamera] = None
_camera_lock = threading.Lock()


def get_camera() -> OrbbecCamera:
    global _camera
    with _camera_lock:
        if _camera is None:
            _camera = OrbbecCamera()
            _camera.start()
        return _camera


def shutdown_camera() -> None:
    global _camera
    with _camera_lock:
        if _camera is not None:
            _camera.stop()
            _camera = None
