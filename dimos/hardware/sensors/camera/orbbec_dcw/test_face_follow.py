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

import pytest

from dimos.hardware.sensors.camera.orbbec_dcw.face_follow import (
    FaceFollowConfig,
    FaceFollowController,
)


def test_controller_is_disabled_by_default() -> None:
    published = []
    controller = FaceFollowController(publisher=published.append)

    status = controller.update((0.9, 0.1), now=1.0)

    assert not status["enabled"]
    assert status["state"] == "disabled"
    assert published == []


def test_controller_acquires_then_publishes_bounded_twist() -> None:
    published = []
    controller = FaceFollowController(
        FaceFollowConfig(acquire_frames=2),
        publisher=published.append,
    )
    controller.set_enabled(True, now=1.0)

    first = controller.update((1.0, 1.0), now=1.1)
    second = controller.update((1.0, 1.0), now=1.2)

    assert first["state"] == "acquiring"
    assert second["state"] == "tracking_no_depth"
    assert second["yaw_speed"] == pytest.approx(-0.04)
    assert second["pitch_speed"] == pytest.approx(0.035)
    assert list(published[-1].linear) == [0.0, 0.0, 0.0]
    assert published[-1].angular.y == pytest.approx(0.035)
    assert published[-1].angular.z == pytest.approx(-0.04)
    assert published[-1].frame_id == "eef_twist_arm"


def test_face_loss_and_disable_publish_zero() -> None:
    published = []
    controller = FaceFollowController(
        FaceFollowConfig(acquire_frames=1),
        publisher=published.append,
    )
    controller.set_enabled(True, now=1.0)
    controller.update((0.9, 0.5), now=1.1)

    lost = controller.update(None, now=1.2)
    disabled = controller.set_enabled(False, now=1.3)

    assert lost["state"] == "searching"
    assert disabled["state"] == "disabled"
    assert published[-1].is_zero()


def test_controller_auto_disables_after_bounded_demo_window() -> None:
    published = []
    controller = FaceFollowController(
        FaceFollowConfig(max_enabled_seconds=2.0, acquire_frames=1),
        publisher=published.append,
    )
    controller.set_enabled(True, now=10.0)
    controller.update((0.7, 0.5), now=10.1)

    status = controller.update((0.7, 0.5), now=12.0)

    assert not status["enabled"]
    assert status["state"] == "timed_out"
    assert published[-1].is_zero()


def test_depth_follow_is_bounded_and_stops_immediately_on_invalid_depth() -> None:
    published = []
    controller = FaceFollowController(
        FaceFollowConfig(
            acquire_frames=1,
            acquire_depth_frames=1,
            depth_smoothing=1.0,
            max_linear_accel=1.0,
        ),
        publisher=published.append,
    )
    controller.set_enabled(True, now=1.0)

    far = controller.update((0.5, 0.5), 2.0, now=1.1)
    near = controller.update((0.5, 0.5), 0.5, now=1.2)
    invalid = controller.update((0.5, 0.5), None, now=1.3)

    assert far["linear_speed"] == pytest.approx(0.04)
    assert near["linear_speed"] == pytest.approx(-0.038)
    assert invalid["linear_speed"] == 0.0
    assert invalid["state"] == "tracking_no_depth"
    assert published[-1].linear.x == 0.0


def test_face_filter_and_slew_limit_command_changes() -> None:
    published = []
    controller = FaceFollowController(
        FaceFollowConfig(acquire_frames=1, face_smoothing=0.5),
        publisher=published.append,
    )
    controller.set_enabled(True, now=1.0)

    first = controller.update((1.0, 0.5), now=1.1)
    second = controller.update((0.0, 0.5), now=1.2)

    assert first["face_center_x"] == 1.0
    assert second["face_center_x"] == 0.5
    assert first["yaw_speed"] == pytest.approx(-0.04)
    assert second["yaw_speed"] == pytest.approx(0.0)
