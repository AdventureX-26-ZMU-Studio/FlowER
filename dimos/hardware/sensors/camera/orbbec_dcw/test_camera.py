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

import cv2
import numpy as np
import pytest

from dimos.hardware.sensors.camera.orbbec_dcw.camera import (
    _camera_info_from_intrinsics,
    _decode_color_frame,
    _decode_depth_frame,
    _frame_data_u8,
)


@dataclass
class _Intrinsic:
    fx: float = 500.0
    fy: float = 510.0
    cx: float = 320.0
    cy: float = 180.0
    width: int = 640
    height: int = 360


@dataclass
class _Distortion:
    k1: float = 0.1
    k2: float = -0.2
    p1: float = 0.01
    p2: float = -0.01
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0


class _Format:
    MJPG = "mjpg"
    RGB = "rgb"
    BGR = "bgr"
    YUYV = "yuyv"
    YUY2 = "yuy2"


class _Frame:
    def __init__(
        self,
        data: bytes,
        *,
        width: int,
        height: int,
        pixel_format: str,
        depth_scale: float = 1.0,
    ) -> None:
        self._data = data
        self._width = width
        self._height = height
        self._format = pixel_format
        self._depth_scale = depth_scale

    def get_data(self) -> bytes:
        return self._data

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def get_format(self) -> str:
        return self._format

    def get_depth_scale(self) -> float:
        return self._depth_scale


def test_camera_info_uses_matrix_layout() -> None:
    info = _camera_info_from_intrinsics(
        _Intrinsic(),
        _Distortion(),
        "dcw_color_optical_frame",
        fallback_width=1,
        fallback_height=1,
    )
    assert info.width == 640
    assert info.height == 360
    assert info.K == [500.0, 0.0, 320.0, 0.0, 510.0, 180.0, 0.0, 0.0, 1.0]
    assert len(info.P) == 12
    assert info.D == [0.1, -0.2, 0.01, -0.01]


def test_decode_mjpeg_to_rgb() -> None:
    bgr = np.zeros((3, 4, 3), dtype=np.uint8)
    bgr[:, :] = (10, 20, 240)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    frame = _Frame(
        encoded.tobytes(),
        width=4,
        height=3,
        pixel_format=_Format.MJPG,
    )
    rgb = _decode_color_frame(frame, _Format)
    assert rgb.shape == (3, 4, 3)
    assert float(rgb[..., 0].mean()) > float(rgb[..., 2].mean())


def test_decode_rgb_copies_sdk_buffer() -> None:
    expected = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    frame = _Frame(
        expected.tobytes(),
        width=3,
        height=2,
        pixel_format=_Format.RGB,
    )
    actual = _decode_color_frame(frame, _Format)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.owndata


def test_decode_depth_scale_is_meters_per_unit() -> None:
    expected = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    frame = _Frame(
        expected.tobytes(),
        width=2,
        height=2,
        pixel_format="y16",
        depth_scale=0.5,
    )
    depth, scale_m = _decode_depth_frame(frame)
    np.testing.assert_array_equal(depth, expected)
    assert scale_m == pytest.approx(0.0005)


def test_non_contiguous_sdk_array_is_flattened() -> None:
    source = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
    assert not source.flags.c_contiguous
    frame = _Frame(
        b"",
        width=3,
        height=4,
        pixel_format=_Format.RGB,
    )
    frame._data = source  # type: ignore[assignment]
    actual = _frame_data_u8(frame)
    np.testing.assert_array_equal(actual, np.ascontiguousarray(source).reshape(-1))


def test_zero_stride_sdk_array_uses_underlying_frame_buffer() -> None:
    storage = np.arange(8, dtype=np.uint8)
    zero_stride = np.lib.stride_tricks.as_strided(storage, shape=(8,), strides=(0,))

    class Frame:
        def get_data(self) -> np.ndarray:
            return zero_stride

        def get_data_size(self) -> int:
            return storage.nbytes

    actual = _frame_data_u8(Frame())

    np.testing.assert_array_equal(actual, storage)
