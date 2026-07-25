from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rePE.vision.models import CameraState, VisionFrame
from rePE.vision.camera import CameraBackend, make_camera_backend


class ConsumerRegistration(BaseModel):
    service_name: str
    streams: list[str] = Field(default_factory=list)


class ConsumerRecord(ConsumerRegistration):
    consumer_id: str
    connected_at: float
    last_seen_at: float


class HubRuntime:
    def __init__(self, backend: CameraBackend | None = None) -> None:
        if backend is not None:
            self.backend = backend
        else:
            backend_kind = os.getenv("FLOWER_VISION_BACKEND", "stub")
            if backend_kind == "orbbec":
                from rePE.vision.orbbec_dcw import OrbbecDCWBackend
                self.backend = OrbbecDCWBackend(
                    width=int(os.getenv("FLOWER_VISION_WIDTH", "640")),
                    height=int(os.getenv("FLOWER_VISION_HEIGHT", "360")),
                    color_fps=int(os.getenv("FLOWER_VISION_COLOR_FPS", "30")),
                    depth_fps=int(os.getenv("FLOWER_VISION_DEPTH_FPS", "15")),
                )
            else:
                self.backend = make_camera_backend(backend_kind)
        self.camera_state = CameraState.STOPPED
        self.consumers: dict[str, ConsumerRecord] = {}
        self.last_frame: VisionFrame | None = None
        self.last_error: str | None = None
        self._lock = asyncio.Lock()

    async def start_camera(self) -> None:
        async with self._lock:
            if self.camera_state == CameraState.RUNNING:
                return
            self.camera_state = CameraState.STARTING
            try:
                await self.backend.start()
                self.camera_state = CameraState.RUNNING
                self.last_error = None
            except Exception as exc:
                self.camera_state = CameraState.ERROR
                self.last_error = str(exc)
                raise

    async def release_camera(self) -> None:
        async with self._lock:
            await self.backend.stop()
            self.camera_state = CameraState.STOPPED

    def register_consumer(self, registration: ConsumerRegistration) -> ConsumerRecord:
        consumer_id = uuid.uuid4().hex
        record = ConsumerRecord(
            consumer_id=consumer_id,
            connected_at=time.time(),
            last_seen_at=time.time(),
            **registration.model_dump(),
        )
        self.consumers[consumer_id] = record
        return record

    def mark_seen(self, consumer_id: str) -> None:
        if consumer_id in self.consumers:
            self.consumers[consumer_id].last_seen_at = time.time()

    def unregister(self, consumer_id: str) -> None:
        self.consumers.pop(consumer_id, None)

    async def next_frame(self) -> VisionFrame:
        frame = await self.backend.next_frame()
        frame.camera_state = self.camera_state
        self.last_frame = frame
        return frame

    def status(self) -> dict[str, Any]:
        return {
            "camera_state": self.camera_state,
            "consumer_count": len(self.consumers),
            "consumers": [record.model_dump() for record in self.consumers.values()],
            "last_frame_ts": self.last_frame.ts if self.last_frame else None,
            "last_error": self.last_error,
        }


runtime = HubRuntime()
app = FastAPI(title="FlowER Vision Hub", version="0.1.0")


def _require_admin(request: Request) -> None:
    configured = os.getenv("FLOWER_VISION_ADMIN_TOKEN")
    if not configured:
        return
    provided = request.headers.get("x-flower-admin-token") or request.query_params.get("token")
    if provided != configured:
        raise HTTPException(status_code=403, detail="admin token required")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <title>FlowER Vision Hub</title>
        <style>
          body { font-family: system-ui, sans-serif; margin: 24px; background: #101412; color: #edf7ef; }
          button { margin-right: 8px; padding: 8px 12px; }
          pre { background: #1d261f; padding: 12px; border-radius: 6px; overflow: auto; }
        </style>
      </head>
      <body>
        <h1>FlowER Vision Hub</h1>
        <p>DaBaiDCW USB ownership is controlled here. Consumer modules subscribe only.</p>
        <button onclick="post('/v1/camera/start')">Start DCW</button>
        <button onclick="post('/v1/camera/release')">Release DCW</button>
        <button onclick="refresh()">Refresh</button>
        <pre id="status">loading...</pre>
        <script>
          async function post(path) { await fetch(path, {method: 'POST'}); await refresh(); }
          async function refresh() {
            const res = await fetch('/v1/status');
            document.getElementById('status').textContent = JSON.stringify(await res.json(), null, 2);
          }
          setInterval(refresh, 1000); refresh();
        </script>
      </body>
    </html>
    """


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/status")
async def status() -> dict[str, Any]:
    return runtime.status()


@app.post("/v1/camera/start")
async def start_camera(request: Request) -> dict[str, str]:
    _require_admin(request)
    await runtime.start_camera()
    return {"camera_state": runtime.camera_state}


@app.post("/v1/camera/release")
async def release_camera(request: Request) -> dict[str, str]:
    _require_admin(request)
    await runtime.release_camera()
    return {"camera_state": runtime.camera_state}


@app.post("/v1/consumers/register")
async def register_consumer(registration: ConsumerRegistration) -> dict[str, Any]:
    record = runtime.register_consumer(registration)
    return record.model_dump()


@app.delete("/v1/consumers/{consumer_id}")
async def unregister_consumer(consumer_id: str) -> dict[str, bool]:
    runtime.unregister(consumer_id)
    return {"ok": True}


@app.get("/v1/streams/export/latest")
async def export_latest() -> dict[str, Any]:
    if runtime.last_frame is None:
        return VisionFrame(camera_state=runtime.camera_state, unavailable_reason="no frame yet").model_dump()
    return runtime.last_frame.model_dump()


def _filter_frame(frame: VisionFrame, streams: list[str]) -> VisionFrame:
    """Filter VisionFrame fields based on requested streams."""
    data = frame.model_dump()
    if "rgb" not in streams:
        data["rgb_jpeg_b64"] = None
    if "depth" not in streams:
        data["depth_shape"] = None
        data["depth_uint16_b64"] = None
    if "detections" not in streams:
        data["obstacles"] = []
        data["faces"] = []
        data["hands"] = []
    return VisionFrame(**data)


@app.websocket("/v1/streams/ws")
async def stream_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    consumer_id = websocket.query_params.get("consumer_id")
    service_name = websocket.query_params.get("service_name", "anonymous-consumer")
    streams = websocket.query_params.get("streams", "rgb").split(",")
    if not consumer_id:
        record = runtime.register_consumer(
            ConsumerRegistration(service_name=service_name, streams=streams)
        )
        consumer_id = record.consumer_id
    try:
        while True:
            runtime.mark_seen(consumer_id)
            frame = await runtime.next_frame()
            filtered = _filter_frame(frame, streams)
            await websocket.send_text(filtered.model_dump_json())
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        runtime.unregister(consumer_id)


def run(host: str = "0.0.0.0", port: int = 8890) -> None:
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
