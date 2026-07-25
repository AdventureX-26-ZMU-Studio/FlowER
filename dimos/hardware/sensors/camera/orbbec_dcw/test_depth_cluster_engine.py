from __future__ import annotations

import numpy as np
import pytest

from dimos.hardware.sensors.camera.orbbec_dcw.depth_cluster_engine import (
    DepthClusterConfig,
    DepthClusterEngine,
)


def test_raw_millimetre_units_are_not_misreported_as_metres() -> None:
    raw_depth = np.full((360, 640), 1000, dtype=np.uint16)

    result = DepthClusterEngine().process(
        raw_depth,
        fx=570.0,
        fy=570.0,
        cx=320.0,
        cy=180.0,
    )

    assert result == []


def test_intrinsics_are_scaled_with_the_downsampled_image() -> None:
    depth_m = np.full((360, 640), 1.0, dtype=np.float32)

    result = DepthClusterEngine().process(
        depth_m,
        fx=570.0,
        fy=570.0,
        cx=320.0,
        cy=180.0,
    )

    assert len(result) == 1
    assert result[0]["center_m"][0] == pytest.approx(0.0, abs=0.01)
    assert result[0]["center_m"][1] == pytest.approx(0.0, abs=0.01)
    assert result[0]["center_m"][2] == pytest.approx(1.0)


def test_tracking_assigns_unique_stable_ids() -> None:
    depth_m = np.zeros((360, 640), dtype=np.float32)
    depth_m[80:280, 40:250] = 0.8
    depth_m[80:280, 390:600] = 1.4
    engine = DepthClusterEngine(
        DepthClusterConfig(
            min_cells=2,
            min_volume_m3=0.0,
        )
    )

    first = engine.process(
        depth_m,
        fx=570.0,
        fy=570.0,
        cx=320.0,
        cy=180.0,
        now=1.0,
    )
    second = engine.process(
        depth_m,
        fx=570.0,
        fy=570.0,
        cx=320.0,
        cy=180.0,
        now=1.1,
    )

    first_ids = [item["id"] for item in first]
    second_ids = [item["id"] for item in second]
    assert len(first_ids) == 2
    assert len(set(first_ids)) == 2
    assert second_ids == first_ids


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="grid dimensions"):
        DepthClusterConfig(input_width=10, grid_cols=20)
