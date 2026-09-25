from flask import render_template, make_response, Response
from app.blueprints.vision import bp
from app.services.api_client import get_video_stream_url
import threading
import time
import ctypes
import numpy as np

_dcw_lock = threading.Lock()
_dcw_pipeline = None


def _frame_data_u8(frame):
    """Return SDK frame data as contiguous uint8 array.
    Handles pyorbbecsdk aarch64 zero-stride bug.
    """
    data = frame.get_data()
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    if isinstance(data, np.ndarray) and not data.flags.c_contiguous:
        data_size = int(frame.get_data_size()) if hasattr(frame, "get_data_size") else 0
        if data_size > 0:
            return np.frombuffer(ctypes.string_at(data.ctypes.data, data_size), dtype=np.uint8)
    return np.ascontiguousarray(data).view(np.uint8).reshape(-1)


def _get_pipeline():
    global _dcw_pipeline
    if _dcw_pipeline is None:
        from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline
        p = Pipeline()
        c = Config()
        profiles = p.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile = profiles.get_video_stream_profile(640, 360, OBFormat.MJPG, 30)
        c.enable_stream(profile)
        p.start(c)
        _dcw_pipeline = p
    return _dcw_pipeline


@bp.route("")
def stream():
    stream_url = get_video_stream_url()
    resp = make_response(render_template("vision/stream.html", stream_url=stream_url))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@bp.route("/api/stream")
def video_feed():
    with _dcw_lock:
        pipeline = _get_pipeline()

    def generate():
        while True:
            frames = pipeline.wait_for_frames(2000)
            if not frames:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            jpg = _frame_data_u8(color)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# WebSocket endpoint
from flask_sock import Sock
sock = Sock()


@sock.route("/ws/dcw")
def dcw_ws(ws):
    with _dcw_lock:
        pipeline = _get_pipeline()

    while True:
        frames = pipeline.wait_for_frames(2000)
        if not frames:
            continue
        color = frames.get_color_frame()
        if not color:
            continue
        try:
            ws.send(_frame_data_u8(color).tobytes())
        except Exception:
            break
        time.sleep(0.033)
