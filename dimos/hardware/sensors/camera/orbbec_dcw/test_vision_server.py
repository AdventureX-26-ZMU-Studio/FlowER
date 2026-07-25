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

from __future__ import annotations

from dataclasses import dataclass
from http.server import ThreadingHTTPServer
import json
import threading
import urllib.request

import numpy as np
import pytest

from dimos.hardware.sensors.camera.orbbec_dcw.vision_server import (
    DCWVisionService,
    FrameStore,
    classify_gesture,
    depth_at_normalized_point,
    depth_preview,
)


@dataclass
class _Landmark:
    x: float = 0.5
    y: float = 0.5


@dataclass
class _Hand:
    landmark: list[_Landmark]


def _hand_with_extended_fingers(count: int) -> _Hand:
    points = [_Landmark() for _ in range(21)]
    for index, tip in enumerate((8, 12, 16, 20)):
        points[tip].y = 0.2 if index < count else 0.8
        points[tip - 2].y = 0.5
    points[4].x = points[3].x
    return _Hand(points)


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, "fist"), (2, "peace"), (4, "palm")],
)
def test_classify_gesture(count: int, expected: str) -> None:
    assert classify_gesture(_hand_with_extended_fingers(count)) == expected


def test_depth_preview_reports_center_in_millimetres() -> None:
    depth = np.full((9, 9), 1000, dtype=np.uint16)
    preview, center_mm = depth_preview(depth, 0.001)
    assert preview.shape == (9, 9, 3)
    assert center_mm == pytest.approx(1000.0)


def test_depth_preview_handles_missing_depth() -> None:
    depth = np.zeros((5, 5), dtype=np.uint16)
    _, center_mm = depth_preview(depth, 0.001)
    assert center_mm == 0.0


def test_depth_at_normalized_point_uses_local_robust_median() -> None:
    depth = np.full((100, 200), 2000, dtype=np.uint16)
    depth[40:61, 90:111] = 900
    depth[50, 100] = 0

    depth_m = depth_at_normalized_point(
        depth,
        0.001,
        (0.5, 0.5),
        radius_fraction=0.04,
    )

    assert depth_m == pytest.approx(0.9)


def test_depth_at_normalized_point_rejects_invalid_samples() -> None:
    depth = np.zeros((20, 20), dtype=np.uint16)
    assert depth_at_normalized_point(depth, 0.001, (0.5, 0.5)) is None
    assert depth_at_normalized_point(depth, 0.001, None) is None



def test_frame_store_serializes_obstacles_and_real_depth() -> None:
    store = FrameStore()
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)

    store.update_status(
        obstacle_count=1,
        obstacles=[{"id": "depth_0", "center_m": [0.0, 0.0, 1.0]}],
    )
    store.update_depth(depth, 0.001)

    status = json.loads(store.status_json())
    raw, width, height, scale_m = store.depth_snapshot()
    assert status["obstacle_count"] == 1
    assert status["obstacles"][0]["id"] == "depth_0"
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype="<u2").reshape(height, width),
        depth,
    )
    assert (width, height, scale_m) == (4, 3, 0.001)


def test_http_obstacle_and_raw_depth_endpoints() -> None:
    service = DCWVisionService(
        host="127.0.0.1",
        port=0,
        fps=15,
        inference_every=2,
        auto_flip=False,
        enable_depth=True,
    )
    depth = np.array([[0, 1000], [1200, 1400]], dtype=np.uint16)
    service.store.update_status(
        obstacle_count=1,
        obstacles=[{"id": "depth_7"}],
    )
    service.store.update_depth(depth, 0.001)
    server = ThreadingHTTPServer(("127.0.0.1", 0), service._handler_class())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with opener.open(f"{base_url}/obstacles", timeout=2.0) as response:
            assert json.loads(response.read()) == [{"id": "depth_7"}]
        with opener.open(f"{base_url}/depth_raw", timeout=2.0) as response:
            raw = response.read()
            assert response.headers["X-Depth-Width"] == "2"
            assert response.headers["X-Depth-Height"] == "2"
            assert float(response.headers["X-Depth-Scale-M"]) == pytest.approx(0.001)
            np.testing.assert_array_equal(
                np.frombuffer(raw, dtype="<u2").reshape(2, 2),
                depth,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
