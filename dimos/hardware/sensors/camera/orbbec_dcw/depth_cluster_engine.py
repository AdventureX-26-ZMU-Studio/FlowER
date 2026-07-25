"""Lightweight depth-surface clustering for the Orbbec DCW service.

The engine accepts a depth image in metres plus intrinsics calibrated for the
input image. It downsamples both the image and the intrinsics, groups adjacent
grid cells with similar depth, and assigns stable one-to-one IDs to the
resulting surfaces.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthClusterConfig:
    input_width: int = 320
    input_height: int = 180
    grid_rows: int = 12
    grid_cols: int = 20
    min_depth_m: float = 0.1
    max_depth_m: float = 3.0
    merge_dz_m: float = 0.15
    min_cells: int = 2
    min_volume_m3: float = 0.0005
    max_objects: int = 10
    association_distance_m: float = 0.3
    stale_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        positive_ints = (
            self.input_width,
            self.input_height,
            self.grid_rows,
            self.grid_cols,
            self.min_cells,
            self.max_objects,
        )
        if min(positive_ints) <= 0:
            raise ValueError("image, grid, cell, and object counts must be positive")
        if self.grid_rows > self.input_height or self.grid_cols > self.input_width:
            raise ValueError("grid dimensions cannot exceed the downsampled image")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("depth range must be positive and increasing")
        if self.merge_dz_m < 0.0:
            raise ValueError("merge_dz_m must be non-negative")
        if self.min_volume_m3 < 0.0:
            raise ValueError("min_volume_m3 must be non-negative")
        if self.association_distance_m <= 0.0 or self.stale_timeout_s <= 0.0:
            raise ValueError("tracking distances and timeouts must be positive")


@dataclass
class _Cell:
    row: int
    col: int
    x: float
    y: float
    z: float
    half_width: float
    half_height: float


@dataclass
class _TrackedObject:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    volume: float
    first_seen: float
    last_seen: float


class DepthClusterEngine:
    """Cluster adjacent depth cells and maintain stable object IDs."""

    def __init__(self, config: DepthClusterConfig | None = None) -> None:
        self.config = config or DepthClusterConfig()
        self._next_id = 0
        self._objects: dict[str, _TrackedObject] = {}

    def process(
        self,
        depth_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return clustered depth surfaces.

        ``depth_m`` must contain metres. The supplied intrinsics must correspond
        to its original width and height; they are scaled internally when the
        image is resized.
        """
        source = np.asarray(depth_m)
        if source.ndim != 2 or source.size == 0:
            raise ValueError("depth_m must be a non-empty 2D image")
        intrinsics = (float(fx), float(fy), float(cx), float(cy))
        if not all(math.isfinite(value) for value in intrinsics):
            raise ValueError("camera intrinsics must be finite")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("focal lengths must be positive")

        source_height, source_width = source.shape
        target_width = self.config.input_width
        target_height = self.config.input_height
        small = cv2.resize(
            source.astype(np.float32, copy=False),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        scaled_fx = fx * scale_x
        scaled_fy = fy * scale_y
        scaled_cx = cx * scale_x
        scaled_cy = cy * scale_y

        cells = self._build_cells(
            small,
            scaled_fx,
            scaled_fy,
            scaled_cx,
            scaled_cy,
        )
        clusters = self._cluster_cells(cells)
        timestamp = time.monotonic() if now is None else float(now)
        return self._track(clusters, timestamp)

    def _build_cells(
        self,
        depth_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> dict[tuple[int, int], _Cell]:
        height, width = depth_m.shape
        row_edges = np.linspace(0, height, self.config.grid_rows + 1, dtype=int)
        col_edges = np.linspace(0, width, self.config.grid_cols + 1, dtype=int)
        cells: dict[tuple[int, int], _Cell] = {}

        for row in range(self.config.grid_rows):
            y0, y1 = int(row_edges[row]), int(row_edges[row + 1])
            for col in range(self.config.grid_cols):
                x0, x1 = int(col_edges[col]), int(col_edges[col + 1])
                patch = depth_m[y0:y1, x0:x1]
                valid = patch[
                    np.isfinite(patch)
                    & (patch >= self.config.min_depth_m)
                    & (patch <= self.config.max_depth_m)
                ]
                if valid.size < 10:
                    continue
                depth = float(np.median(valid))
                pixel_x = (x0 + x1 - 1) / 2.0
                pixel_y = (y0 + y1 - 1) / 2.0
                cells[(row, col)] = _Cell(
                    row=row,
                    col=col,
                    x=(pixel_x - cx) * depth / fx,
                    y=(pixel_y - cy) * depth / fy,
                    z=depth,
                    half_width=max(0.0, (x1 - x0) * depth / (2.0 * fx)),
                    half_height=max(0.0, (y1 - y0) * depth / (2.0 * fy)),
                )
        return cells

    def _cluster_cells(
        self,
        cells: dict[tuple[int, int], _Cell],
    ) -> list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            int,
        ]
    ]:
        remaining = set(cells)
        clusters: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                float,
                int,
            ]
        ] = []

        while remaining:
            seed_key = min(remaining)
            remaining.remove(seed_key)
            queue: deque[tuple[int, int]] = deque([seed_key])
            group: list[_Cell] = []

            while queue:
                key = queue.popleft()
                cell = cells[key]
                group.append(cell)
                for neighbour in (
                    (cell.row - 1, cell.col),
                    (cell.row + 1, cell.col),
                    (cell.row, cell.col - 1),
                    (cell.row, cell.col + 1),
                ):
                    other = cells.get(neighbour)
                    if (
                        neighbour in remaining
                        and other is not None
                        and abs(other.z - cell.z) <= self.config.merge_dz_m
                    ):
                        remaining.remove(neighbour)
                        queue.append(neighbour)

            if len(group) < self.config.min_cells:
                continue
            xs_min = [cell.x - cell.half_width for cell in group]
            xs_max = [cell.x + cell.half_width for cell in group]
            ys_min = [cell.y - cell.half_height for cell in group]
            ys_max = [cell.y + cell.half_height for cell in group]
            zs = [cell.z for cell in group]
            min_x, max_x = min(xs_min), max(xs_max)
            min_y, max_y = min(ys_min), max(ys_max)
            min_z, max_z = min(zs), max(zs)
            center = (
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                float(np.median(zs)),
            )
            size = (
                max(0.05, max_x - min_x),
                max(0.05, max_y - min_y),
                max(0.03, max_z - min_z),
            )
            volume = size[0] * size[1] * size[2]
            if volume >= self.config.min_volume_m3:
                clusters.append((center, size, volume, len(group)))

        clusters.sort(key=lambda item: (-item[2], item[0]))
        return clusters[: self.config.max_objects]

    def _track(
        self,
        clusters: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                float,
                int,
            ]
        ],
        now: float,
    ) -> list[dict[str, Any]]:
        self._objects = {
            object_id: tracked
            for object_id, tracked in self._objects.items()
            if now - tracked.last_seen <= self.config.stale_timeout_s
        }
        available_ids = set(self._objects)
        results: list[dict[str, Any]] = []

        for center, size, volume, cell_count in clusters:
            object_id: str | None = None
            best_distance = self.config.association_distance_m
            for candidate_id in sorted(available_ids):
                candidate = self._objects[candidate_id]
                distance = math.dist(candidate.center, center)
                if distance < best_distance:
                    object_id = candidate_id
                    best_distance = distance

            if object_id is None:
                object_id = f"depth_{self._next_id}"
                self._next_id += 1
                tracked = _TrackedObject(
                    center=center,
                    size=size,
                    volume=volume,
                    first_seen=now,
                    last_seen=now,
                )
                self._objects[object_id] = tracked
            else:
                available_ids.remove(object_id)
                tracked = self._objects[object_id]
                tracked.center = center
                tracked.size = size
                tracked.volume = volume
                tracked.last_seen = now

            results.append(
                {
                    "id": object_id,
                    "center_m": [round(value, 3) for value in center],
                    "size_m": [round(value, 3) for value in size],
                    "volume_m3": round(volume, 6),
                    "cell_count": cell_count,
                    "first_seen_s": round(now - tracked.first_seen, 1),
                    "last_seen_s": round(now - tracked.last_seen, 1),
                }
            )
        return results
