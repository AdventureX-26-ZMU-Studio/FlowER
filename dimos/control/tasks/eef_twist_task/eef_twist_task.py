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
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
import pinocchio

from dimos.control.task import (
    BaseControlTask,
    ControlMode,
    CoordinatorState,
    JointCommandOutput,
    ResourceClaim,
)
from dimos.manipulation.planning.kinematics.pinocchio_ik import (
    PinocchioIK,
    check_joint_delta,
    get_worst_joint_delta,
)
from dimos.protocol.service.spec import BaseConfig
from dimos.utils.logging_config import setup_logger
from dimos.utils.transform_utils import twist_to_numpy

if TYPE_CHECKING:
    from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
    from dimos.msgs.std_msgs.Bool import Bool

logger = setup_logger()

_MAX_DT = 0.05
_JOINT_LIMIT_TOLERANCE_RAD = 0.05
_JOINT_LIMIT_LOG_INTERVAL_S = 1.0
_PREEMPT_LOG_INTERVAL_S = 1.0


@dataclass
class EEFTwistTaskConfig:
    joint_names: list[str]
    model_path: str | Path
    ee_joint_id: int
    timeout: float
    max_joint_delta_deg: float
    twist_frame: Literal["world", "eef"] = "world"
    priority: int = 10
    gripper_joint: str | None = None
    gripper_open_pos: float = 0.0
    gripper_closed_pos: float = 0.0


class EEFTwistTask(BaseControlTask):
    def __init__(self, name: str, config: EEFTwistTaskConfig) -> None:
        if not config.joint_names:
            raise ValueError(f"EEFTwistTask '{name}' requires at least one joint")
        if not config.model_path:
            raise ValueError(f"EEFTwistTask '{name}' requires model_path for IK solver")
        if config.twist_frame not in ("world", "eef"):
            raise ValueError(f"EEFTwistTask '{name}' has unsupported twist_frame")
        self._name = name
        self._config = config
        self._joint_names = frozenset(config.joint_names)
        self._joint_names_list = list(config.joint_names)
        self._ik = PinocchioIK.from_model_path(config.model_path, config.ee_joint_id)
        if self._ik.nq != len(config.joint_names):
            raise ValueError(
                f"EEFTwistTask {name}: model DOF ({self._ik.nq}) != "
                f"joint_names count ({len(config.joint_names)})"
            )
        self._joint_lower = np.asarray(
            self._ik.model.lowerPositionLimit,
            dtype=np.float64,
        ).reshape(-1)
        self._joint_upper = np.asarray(
            self._ik.model.upperPositionLimit,
            dtype=np.float64,
        ).reshape(-1)
        if (
            self._joint_lower.size != len(config.joint_names)
            or self._joint_upper.size != len(config.joint_names)
        ):
            raise ValueError(
                f"EEFTwistTask {name}: model limit count does not match joint_names"
            )
        self._lock = threading.Lock()
        self._latest_twist: TwistStamped | None = None
        self._last_update_time = 0.0
        self._last_joint_limit_log_time = float("-inf")
        self._last_preempt_log_time = float("-inf")
        self._estopped = False

        self._hold_target: NDArray[np.floating[Any]] | None = None
        self._gripper_target: float = config.gripper_open_pos

    def claim(self) -> ResourceClaim:
        joints = self._joint_names
        if self._config.gripper_joint:
            joints = joints | frozenset([self._config.gripper_joint])
        return ResourceClaim(joints, self._config.priority, ControlMode.SERVO_POSITION)

    def is_active(self) -> bool:
        with self._lock:
            return not self._estopped

    def set_estop(self, estopped: bool) -> None:
        """Latch/clear E-STOP. On latch, drop the pending jog and hold anchor so
        clearing resumes from the current pose, not a stale target. The gripper
        target is kept so a held payload isn't released."""
        with self._lock:
            self._estopped = estopped
            if estopped:
                self._latest_twist = None
                self._hold_target = None

    def on_ee_twist_command(self, twist: TwistStamped, t_now: float) -> bool:
        values = twist_to_numpy(twist)
        if not np.all(np.isfinite(values)):
            logger.warning("EEFTwistTask rejecting non-finite twist", task=self._name)
            return False
        with self._lock:
            if self._estopped:
                # A twist in transit when E-STOP latched must not be stored, or
                # it would replay on the next tick after the latch clears.
                return False
            self._last_update_time = t_now
            # Zero twist → hold (None); non-zero → jog. The anchor persists either way.
            self._latest_twist = None if np.allclose(values, 0.0) else twist
        return True

    def on_gripper_command(self, msg: Bool, t_now: float) -> bool:
        if not self._config.gripper_joint:
            return False
        with self._lock:
            if self._estopped:
                # Reject new grip changes during a stop; the held target (from
                # before E-STOP) is kept so a payload isn't dropped.
                return False
            self._gripper_target = (
                self._config.gripper_closed_pos if msg.data else self._config.gripper_open_pos
            )
        return True

    def _with_gripper(self, joint_names: list[str], positions: list[float]) -> JointCommandOutput:
        if self._config.gripper_joint:
            with self._lock:
                gripper_pos = self._gripper_target
            joint_names = [*joint_names, self._config.gripper_joint]
            positions = [*positions, gripper_pos]
        return JointCommandOutput(
            joint_names=joint_names,
            positions=positions,
            mode=ControlMode.SERVO_POSITION,
        )

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        with self._lock:
            if (
                self._latest_twist is not None
                and self._config.timeout > 0
                and state.t_now - self._last_update_time > self._config.timeout
            ):
                self._latest_twist = None
            twist = self._latest_twist
            anchor = self._hold_target  # last commanded joint target (not live pos)

        q_current = self._get_current_joints(state)
        if q_current is None or not np.all(np.isfinite(q_current)):
            return None

        if anchor is None:
            anchor = q_current
            with self._lock:
                self._hold_target = anchor

        safe_anchor = self._safe_joint_target(anchor, state.t_now)
        if safe_anchor is None:
            with self._lock:
                self._hold_target = None
            return None
        anchor = safe_anchor
        with self._lock:
            self._hold_target = anchor

        if twist is None:
            return self._with_gripper(self._joint_names_list, anchor.flatten().tolist())

        target_pose = self._ik.forward_kinematics(anchor)
        dt = min(max(state.dt, 0.0), _MAX_DT)
        candidate = self._integrate_twist(target_pose, twist, dt)

        q_solution, converged, final_error = self._ik.solve(candidate, anchor)
        if not np.all(np.isfinite(q_solution)):
            return None
        q_solution = self._safe_joint_target(q_solution, state.t_now)
        if q_solution is None:
            return None
        if not converged:
            logger.debug(
                "EEFTwistTask IK did not converge, using partial solution",
                task=self._name,
                error=final_error,
            )
        if not check_joint_delta(q_solution, anchor, self._config.max_joint_delta_deg):
            worst_idx, worst_deg = get_worst_joint_delta(q_solution, anchor)
            logger.warning(
                "EEFTwistTask rejecting solution: joint delta exceeds limit",
                task=self._name,
                joint=self._joint_names_list[worst_idx],
                delta_deg=worst_deg,
                max_delta_deg=self._config.max_joint_delta_deg,
            )
            return None

        # Advance the anchor so the next tick (and any hold) continues from here.
        q_solution = q_solution.flatten()
        with self._lock:
            self._hold_target = q_solution
        return self._with_gripper(self._joint_names_list, q_solution.tolist())

    def _safe_joint_target(
        self,
        target: NDArray[np.floating[Any]],
        t_now: float,
    ) -> NDArray[np.floating[Any]] | None:
        """Clip tiny numerical overshoots and reject unsafe IK/hold targets."""
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if target.size != self._joint_lower.size:
            logger.warning(
                "EEFTwistTask rejecting target with unexpected shape",
                task=self._name,
                expected=self._joint_lower.size,
                actual=target.size,
            )
            return None
        below = target < self._joint_lower - _JOINT_LIMIT_TOLERANCE_RAD
        above = target > self._joint_upper + _JOINT_LIMIT_TOLERANCE_RAD
        invalid = below | above
        if np.any(invalid):
            if t_now - self._last_joint_limit_log_time >= _JOINT_LIMIT_LOG_INTERVAL_S:
                self._last_joint_limit_log_time = t_now
                offenders = [
                    (
                        f"{self._joint_names_list[index]}={target[index]:.3f} "
                        f"outside [{self._joint_lower[index]:.3f}, "
                        f"{self._joint_upper[index]:.3f}]"
                    )
                    for index in np.flatnonzero(invalid)
                ]
                logger.warning(
                    "EEFTwistTask rejecting target outside model joint limits",
                    task=self._name,
                    joints="; ".join(offenders),
                )
            return None
        return np.clip(target, self._joint_lower, self._joint_upper)

    def on_preempted(self, by_task: str, joints: frozenset[str]) -> None:
        if joints & self._joint_names:
            now = time.monotonic()
            if now - self._last_preempt_log_time >= _PREEMPT_LOG_INTERVAL_S:
                self._last_preempt_log_time = now
                logger.warning(
                    "EEFTwistTask preempted",
                    task=self._name,
                    by_task=by_task,
                    joints=joints,
                )
            with self._lock:
                self._hold_target = None
                self._latest_twist = None

    def _get_current_joints(self, state: CoordinatorState) -> NDArray[np.floating[Any]] | None:
        positions = []
        for joint_name in self._joint_names_list:
            pos = state.joints.get_position(joint_name)
            if pos is None:
                return None
            positions.append(pos)
        return np.array(positions, dtype=np.float64)

    def _integrate_twist(
        self, pose: pinocchio.SE3, twist: TwistStamped, dt: float
    ) -> pinocchio.SE3:
        candidate = pose.copy()
        values = twist_to_numpy(twist)
        if self._config.twist_frame == "eef":
            linear_velocity = pose.rotation @ values[:3]
            angular_velocity = pose.rotation @ values[3:]
        else:
            linear_velocity = values[:3]
            angular_velocity = values[3:]
        candidate.translation = candidate.translation + linear_velocity * dt
        angular_step = angular_velocity * dt
        if np.linalg.norm(angular_step) > 0.0:
            candidate.rotation = pinocchio.exp3(angular_step) @ candidate.rotation
        return candidate


class EEFTwistTaskParams(BaseConfig):
    model_path: str | Path
    ee_joint_id: int = 6
    timeout: float = 0.3
    max_joint_delta_deg: float = 15.0
    twist_frame: Literal["world", "eef"] = "world"
    gripper_joint: str | None = None
    gripper_open_pos: float = 0.0
    gripper_closed_pos: float = 0.0


def create_task(cfg: Any, hardware: Any) -> EEFTwistTask:
    params = EEFTwistTaskParams.model_validate(cfg.params)
    return EEFTwistTask(
        cfg.name,
        EEFTwistTaskConfig(
            joint_names=cfg.joint_names,
            model_path=params.model_path,
            ee_joint_id=params.ee_joint_id,
            priority=cfg.priority,
            timeout=params.timeout,
            max_joint_delta_deg=params.max_joint_delta_deg,
            twist_frame=params.twist_frame,
            gripper_joint=params.gripper_joint,
            gripper_open_pos=params.gripper_open_pos,
            gripper_closed_pos=params.gripper_closed_pos,
        ),
    )
