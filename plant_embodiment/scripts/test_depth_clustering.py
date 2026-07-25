#!/usr/bin/env python3
"""DCW Depth → 3D 聚类测试（独立，不需要 dimos）"""

import sys
import numpy as np
import cv2
import time

def depth_grid_cluster(
    depth, grid_rows=12, grid_cols=20,
    fx=285.0, fy=285.0, cx=160.0, cy=90.0,
    min_depth=0.1, max_depth=3.0, floor_y=0.02,
    merge_dz=0.15, min_vol=0.0005, max_objs=10,
):
    """深度图分块聚类 → 3D 包围盒"""
    h, w = depth.shape
    ch, cw = max(1, h // grid_rows), max(1, w // grid_cols)

    cells = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            patch = depth[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            patch = patch[patch > 0]
            if patch.size < 10:
                continue
            d = float(np.median(patch))
            if d < min_depth or d > max_depth:
                continue
            uc, vc = (2*c+1)*cw//2, (2*r+1)*ch//2
            xc = (uc - cx) * d / fx
            yc = (vc - cy) * d / fy
            cells.append((xc, yc, d, patch.size))

    if not cells:
        return []

    cells.sort(key=lambda c: c[2])
    used = [False] * len(cells)
    objects = []

    for i in range(len(cells)):
        if used[i]:
            continue
        group = [cells[i]]
        used[i] = True
        for j in range(i+1, len(cells)):
            if used[j]:
                continue
            if abs(cells[j][2] - cells[i][2]) < merge_dz:
                if abs(cells[j][0] - cells[i][0]) < 0.3 and abs(cells[j][1] - cells[i][1]) < 0.3:
                    group.append(cells[j])
                    used[j] = True

        if len(group) >= 2:
            xs = [c[0] for c in group]
            ys = [c[1] for c in group]
            zs = [c[2] for c in group]
            cx_m = float(np.median(xs))
            cy_m = float(np.median(ys))
            cz_m = float(np.median(zs))
            sw = max(0.05, np.max(xs) - np.min(xs))
            sh = max(0.05, np.max(ys) - np.min(ys))
            sd = max(0.03, np.max(zs) - np.min(zs))
            vol = sw * sh * sd
            if vol >= min_vol:
                objects.append((cx_m, cy_m, cz_m, sw, sh, sd, vol, len(group)))

    objects.sort(key=lambda o: o[6], reverse=True)
    return objects[:max_objs]


def main():
    sys.path.insert(0, "/home/asus/orbbec_sdk_libs")
    from pyorbbecsdk import Config, OBSensorType, Pipeline

    pipeline = Pipeline()
    cfg = Config()
    profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    cfg.enable_stream(profile)
    pipeline.start(cfg)

    print("🔍 DCW Depth → 3D 障碍物检测")
    print("  站到摄像头前来回走动...")
    print("-" * 60)
    print(f"{'#':>4s}  {'X(m)':>7s} {'Y(m)':>7s} {'Z(m)':>7s} {'W(m)':>7s} {'H(m)':>7s} {'D(m)':>7s} {'Vol(m³)':>7s}")
    print("-" * 60)

    frame = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue
            df = frames.get_depth_frame()
            if not df:
                continue

            raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(df.get_height(), df.get_width())
            depth_m = raw.astype(np.float32) * df.get_depth_scale()

            # 降采样（加速）
            small = cv2.resize(depth_m, (320, 180), interpolation=cv2.INTER_NEAREST)

            objs = depth_grid_cluster(small)

            frame += 1
            if frame % 10 == 0:
                if objs:
                    for o in objs:
                        print(f"{frame:>4d}  {o[0]:7.2f} {o[1]:7.2f} {o[2]:7.2f} {o[3]:7.2f} {o[4]:7.2f} {o[5]:7.2f} {o[6]:7.4f}")
                else:
                    print(f"{frame:>4d}  (空)")

    except KeyboardInterrupt:
        pipeline.stop()
        print("\n✅ 完成")


if __name__ == "__main__":
    main()
