from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from rePE.vision.models import VisionFrame

# WebSocket 重连参数
_MAX_RECONNECT_DELAY = 60.0   # 秒
_INITIAL_RECONNECT_DELAY = 1.0
_BACKOFF_FACTOR = 2.0


class VisionHubClient:
    """Consumer-side client. It deliberately has no start/release camera methods."""

    def __init__(self, base_url: str, *, service_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_name = service_name

    async def status(self) -> dict:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{self.base_url}/v1/status")
            response.raise_for_status()
            return response.json()

    async def register(self, streams: list[str]) -> str:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/consumers/register",
                json={"service_name": self.service_name, "streams": streams},
            )
            response.raise_for_status()
            return str(response.json()["consumer_id"])

    async def frames(self, streams: list[str]) -> AsyncIterator[VisionFrame]:
        """WebSocket 帧流，自动重连（指数退避，上限60s）。"""
        consumer_id = await self.register(streams)
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        stream_csv = ",".join(streams)
        reconnect_delay = _INITIAL_RECONNECT_DELAY

        while True:
            try:
                async with websockets.connect(
                    f"{ws_url}/v1/streams/ws?consumer_id={consumer_id}&streams={stream_csv}"
                ) as websocket:
                    reconnect_delay = _INITIAL_RECONNECT_DELAY  # 连接成功，重置延迟
                    async for message in websocket:
                        yield VisionFrame.model_validate_json(message)
            except (ConnectionClosed, OSError, asyncio.TimeoutError):
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * _BACKOFF_FACTOR, _MAX_RECONNECT_DELAY)


async def read_one_frame(base_url: str, service_name: str = "diagnostic") -> VisionFrame:
    client = VisionHubClient(base_url, service_name=service_name)
    async for frame in client.frames(["rgb", "depth", "detections"]):
        return frame
    raise RuntimeError("vision stream ended before yielding a frame")


def read_one_frame_sync(base_url: str, service_name: str = "diagnostic") -> VisionFrame:
    return asyncio.run(read_one_frame(base_url, service_name))

