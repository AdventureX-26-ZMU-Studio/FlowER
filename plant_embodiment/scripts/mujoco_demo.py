#!/usr/bin/env python3
"""
MuJoCo 仿真独立演示 — 无需 dimos 完整管线即可运行。

键盘控制:
  W/S:  前进/后退
  A/D:  左转/右转
  ESC:  退出

用法:
  .venv/bin/mjpython plant_embodiment/scripts/mujoco_demo.py
"""

import time
import mujoco
import mujoco.viewer
import numpy as np

# ── 简单场景：平面 + 可移动的球体（模拟机器人） ──
SCENE_XML = """
<mujoco>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4"
             width="512" height="512"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" texuniform="true" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="2 4 4" dir="-1 -2 -2" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="groundplane"/>

    <!-- 机器人主体 (球体) -->
    <body name="robot" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="body" type="sphere" size="0.3" rgba="0.3 0.6 0.9 1"/>
      <geom name="arrow" type="capsule" fromto="0 0 0 0.5 0 0" size="0.05"
            rgba="0.9 0.3 0.3 1"/>
    </body>

    <!-- 一些障碍物 -->
    <body pos="2 1 0.3">
      <geom type="box" size="0.3 0.3 0.3" rgba="0.8 0.5 0.2 1"/>
    </body>
    <body pos="-1 2 0.5">
      <geom type="cylinder" size="0.2 0.5" rgba="0.2 0.8 0.4 1"/>
    </body>
    <body pos="3 -1 0.4">
      <geom type="box" size="0.4 0.2 0.4" rgba="0.7 0.3 0.7 1"/>
    </body>
    <body pos="-2 -1.5 0.3">
      <geom type="sphere" size="0.3" rgba="0.9 0.9 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)

    # 速度参数
    linear_speed = 2.0
    angular_speed = 3.0
    
    # 按键状态 (用 glfw 或 viewer key callback 跟踪)
    keys_pressed = set()

    def key_callback(keycode: int):
        """按下按键时触发."""
        keys_pressed.add(keycode)

    print("=" * 50)
    print("  MuJoCo 仿真演示")
    print("  W/S: 前进/后退  A/D: 左/右转  ESC: 退出")
    print("  鼠标: 旋转视角  右键拖拽: 平移  滚轮: 缩放")
    print("=" * 50)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
    ) as viewer_handle:
        # 设置初始视角
        viewer_handle.cam.lookat = [0, 0, 0.5]
        viewer_handle.cam.distance = 8
        viewer_handle.cam.azimuth = -30
        viewer_handle.cam.elevation = -25

        while viewer_handle.is_running():
            step_start = time.time()

            # ── 键盘控制 ──
            lin_vel = 0.0
            ang_vel = 0.0

            # W/S → 前进/后退 (glfw key codes)
            GLFW_KEY_W = 87
            GLFW_KEY_S = 83
            GLFW_KEY_A = 65
            GLFW_KEY_D = 68
            GLFW_KEY_ESC = 256

            if GLFW_KEY_W in keys_pressed:
                lin_vel = linear_speed
            elif GLFW_KEY_S in keys_pressed:
                lin_vel = -linear_speed

            if GLFW_KEY_A in keys_pressed:
                ang_vel = angular_speed
            elif GLFW_KEY_D in keys_pressed:
                ang_vel = -angular_speed

            if GLFW_KEY_ESC in keys_pressed:
                break

            if lin_vel != 0 or ang_vel != 0:
                # 用 quaternion 算朝向
                w, x, y, z_q = data.qpos[3:7]
                siny_cosp = 2 * (w * z_q + x * y)
                cosy_cosp = 1 - 2 * (y * y + z_q * z_q)
                yaw = np.arctan2(siny_cosp, cosy_cosp)

                # 世界坐标下的速度
                data.qvel[0] = lin_vel * np.cos(yaw)
                data.qvel[1] = lin_vel * np.sin(yaw)
                data.qvel[5] = ang_vel
            else:
                data.qvel[0] = 0
                data.qvel[1] = 0
                data.qvel[5] = 0

            # 仿真步进
            mujoco.mj_step(model, data)

            # 同步显示
            viewer_handle.sync()

            # 控制帧率 (~60fps)
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

        print("\n退出。")


if __name__ == "__main__":
    main()
