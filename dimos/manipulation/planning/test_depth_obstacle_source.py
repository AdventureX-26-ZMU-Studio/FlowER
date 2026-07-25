from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from dimos.manipulation.planning.depth_obstacle_source import (
    DepthObstacleSource,
    DepthObstacleSourceConfig,
)
from dimos.manipulation.planning.spec.models import Obstacle
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped


@dataclass
class _Response:
    payload: object

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class _FakeWorld:
    def __init__(self) -> None:
        self.obstacles: dict[str, Obstacle] = {}
        self.updated: list[str] = []
        self.removed: list[str] = []

    def add(self, obstacle: Obstacle) -> str:
        self.obstacles[obstacle.name] = obstacle
        return obstacle.name

    def update(self, obstacle_id: str, pose: PoseStamped) -> bool:
        self.obstacles[obstacle_id].pose = pose
        self.updated.append(obstacle_id)
        return True

    def remove(self, obstacle_id: str) -> bool:
        self.obstacles.pop(obstacle_id, None)
        self.removed.append(obstacle_id)
        return True


def _source(
    payloads: list[object],
    *,
    world: _FakeWorld | None = None,
    **config: object,
) -> tuple[DepthObstacleSource, _FakeWorld]:
    target = world or _FakeWorld()

    def open_payload(_url: str, *, timeout: float) -> _Response:
        assert timeout > 0.0
        return _Response(payloads.pop(0))

    source = DepthObstacleSource(
        DepthObstacleSourceConfig(
            url="http://127.0.0.1:8890",
            inflation_m=0.1,
            **config,
        ),
        add_obstacle=target.add,
        update_obstacle_pose=target.update,
        remove_obstacle=target.remove,
        get_mount_pose=lambda: PoseStamped(
            frame_id="world",
            position=[1.0, 2.0, 3.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
        ),
        opener=open_payload,
    )
    return source, target


def test_optical_boxes_are_transformed_to_a1z_tool_frame() -> None:
    source, world = _source(
        [
            [
                {
                    "id": "depth_7",
                    "center_m": [0.2, 0.3, 0.8],
                    "size_m": [0.1, 0.2, 0.3],
                }
            ]
        ]
    )

    source.sync_once(now=10.0)

    obstacle = world.obstacles["dcw_depth_7"]
    # optical [right, down, forward] -> tool [forward, left, up]
    assert np.allclose(obstacle.pose.position.to_numpy(), [1.8, 1.8, 2.7])
    assert np.allclose(obstacle.dimensions, [0.3, 0.4, 0.5])
    assert source.status(now=10.1).fresh is True


def test_existing_box_pose_updates_without_recreating_geometry() -> None:
    first = [{"id": "depth_1", "center_m": [0.0, 0.0, 1.0], "size_m": [0.2] * 3}]
    second = [{"id": "depth_1", "center_m": [0.0, 0.0, 1.1], "size_m": [0.2] * 3}]
    source, world = _source([first, second])

    source.sync_once(now=1.0)
    source.sync_once(now=1.2)

    assert world.updated == ["dcw_depth_1"]
    assert world.removed == []
    assert np.isclose(world.obstacles["dcw_depth_1"].pose.position.x, 2.1)


def test_dimension_change_recreates_box_and_missing_grace_removes_it() -> None:
    first = [{"id": "depth_2", "center_m": [0.0, 0.0, 1.0], "size_m": [0.2] * 3}]
    resized = [{"id": "depth_2", "center_m": [0.0, 0.0, 1.0], "size_m": [0.4] * 3}]
    source, world = _source([first, resized, [], []], missing_grace_s=0.5)

    source.sync_once(now=1.0)
    source.sync_once(now=1.2)
    assert world.removed == ["dcw_depth_2"]
    assert "dcw_depth_2" in world.obstacles

    source.sync_once(now=1.4)
    assert "dcw_depth_2" in world.obstacles
    source.sync_once(now=1.8)
    assert "dcw_depth_2" not in world.obstacles


def test_feed_is_fail_closed_when_response_is_invalid() -> None:
    source, _world = _source([{"not": "a-list"}], stale_after_s=0.5)

    source.sync_once(now=4.0)

    status = source.status(now=4.1)
    assert status.ready is False
    assert status.fresh is False
    assert "JSON list" in status.last_error


def test_stale_feed_retains_last_obstacles_but_reports_not_fresh() -> None:
    payload = [{"id": "depth_3", "center_m": [0.0, 0.0, 1.0], "size_m": [0.2] * 3}]
    source, world = _source([payload], stale_after_s=0.5)

    source.sync_once(now=5.0)

    assert source.is_fresh(now=5.4) is True
    assert source.is_fresh(now=5.6) is False
    assert "dcw_depth_3" in world.obstacles
