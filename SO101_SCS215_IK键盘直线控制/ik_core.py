"""SO-101 URDF kinematics and straight-line Cartesian controller."""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "forward": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    # With the measured ID2-ID5 mapping, physical vertical matches the URDF
    # base Z axis.  The earlier inversion was observed with placeholder joint
    # zero positions and must not be retained after calibration.
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}

# A larger value discourages motion of that joint; a smaller value lets it
# contribute earlier.  Downward motion must not consume the limited ID3 range
# before the wrist joint (ID4) starts helping.
DIRECTION_JOINT_MOTION_WEIGHTS: dict[str, tuple[float, ...]] = {
    "down": (1.0, 1.0, 4.0, 0.5, 1.0),
}


def _vector(text: str | None, default: Sequence[float]) -> np.ndarray:
    values = default if text is None else tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"URDF 三维向量格式错误：{text!r}")
    return np.asarray(values, dtype=float)


def _origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        return np.eye(4)
    x, y, z = axis / length
    c, s = math.cos(angle), math.sin(angle)
    one = 1.0 - c
    result = np.eye(4)
    result[:3, :3] = np.asarray(
        (
            (c + x * x * one, x * y * one - z * s, x * z * one + y * s),
            (y * x * one + z * s, c + y * y * one, y * z * one - x * s),
            (z * x * one - y * s, z * y * one + x * s, c + z * z * one),
        )
    )
    return result


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class IKResult:
    success: bool
    joints_deg: tuple[float, ...]
    achieved_xyz: tuple[float, float, float]
    residual_m: float
    iterations: int
    reason: str


class URDFKinematics:
    """Minimal URDF parser with position-only damped least-squares IK."""

    def __init__(
        self,
        urdf_path: str | Path,
        base_link: str = "base_link",
        target_link: str = "gripper_frame_link",
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"找不到 URDF：{self.urdf_path}")

        by_child: dict[str, JointSpec] = {}
        root = ET.parse(self.urdf_path).getroot()
        for element in root.findall("joint"):
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                continue
            origin = element.find("origin")
            axis = element.find("axis")
            limit = element.find("limit")
            kind = element.attrib.get("type", "fixed")
            lower, upper = -math.inf, math.inf
            if kind == "revolute" and limit is not None:
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
            spec = JointSpec(
                name=element.attrib["name"],
                kind=kind,
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=_vector(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0)),
                rpy=_vector(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0)),
                axis=_vector(axis.attrib.get("xyz") if axis is not None else None, (1, 0, 0)),
                lower=lower,
                upper=upper,
            )
            by_child[spec.child] = spec

        reverse_chain: list[JointSpec] = []
        link = target_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"URDF 中不存在 {base_link} 到 {target_link} 的关节链")
            joint = by_child[link]
            reverse_chain.append(joint)
            link = joint.parent
        self.chain = tuple(reversed(reverse_chain))
        self.active_joints = tuple(
            joint for joint in self.chain if joint.kind in {"revolute", "continuous"}
        )
        names = tuple(joint.name for joint in self.active_joints)
        if names != ARM_JOINT_NAMES:
            raise ValueError(f"URDF 关节顺序 {names} 与 SO-101 不一致")

    @property
    def joint_limits_deg(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (math.degrees(joint.lower), math.degrees(joint.upper))
            for joint in self.active_joints
        )

    def _forward_rad(
        self, joints_rad: Sequence[float], *, frames: bool = False
    ) -> tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        if len(joints_rad) != len(ARM_JOINT_NAMES):
            raise ValueError("正运动学需要 5 个关节角")
        values = dict(zip(ARM_JOINT_NAMES, joints_rad, strict=True))
        transform = np.eye(4)
        joint_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for joint in self.chain:
            transform = transform @ _origin_transform(joint.xyz, joint.rpy)
            if joint.kind in {"revolute", "continuous"}:
                if frames:
                    joint_frames.append(
                        (transform[:3, 3].copy(), transform[:3, :3] @ joint.axis)
                    )
                transform = transform @ _axis_angle(joint.axis, values[joint.name])
        return transform, tuple(joint_frames)

    def forward(self, joints_deg: Sequence[float]) -> np.ndarray:
        joints_rad = np.radians(np.asarray(joints_deg[:5], dtype=float))
        return self._forward_rad(joints_rad)[0]

    def position(self, joints_deg: Sequence[float]) -> tuple[float, float, float]:
        xyz = self.forward(joints_deg)[:3, 3]
        return tuple(float(value) for value in xyz)

    def inverse(
        self,
        target_xyz: Sequence[float],
        initial_joints_deg: Sequence[float],
        *,
        joint_limits_deg: Sequence[tuple[float, float]] | None = None,
        tolerance_m: float = 0.0008,
        damping: float = 0.015,
        max_iterations: int = 180,
        max_iteration_step_rad: float = 0.18,
        joint_motion_weights: Sequence[float] | None = None,
    ) -> IKResult:
        """Solve weighted damped least-squares IK with limit-aware line search."""
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or len(initial_joints_deg) < 5:
            raise ValueError("IK 目标必须是 xyz，初值必须包含 5 个关节")
        limits_deg = tuple(joint_limits_deg or self.joint_limits_deg)
        if len(limits_deg) != 5:
            raise ValueError("IK 限位必须包含 5 个关节")
        limits = np.radians(np.asarray(limits_deg, dtype=float))
        weight_values = (
            (1.0,) * len(ARM_JOINT_NAMES)
            if joint_motion_weights is None
            else tuple(joint_motion_weights)
        )
        base_weights = np.asarray(weight_values, dtype=float)
        if base_weights.shape != (len(ARM_JOINT_NAMES),) or not np.all(
            np.isfinite(base_weights) & (base_weights > 0.0)
        ):
            raise ValueError("IK 关节运动权重必须包含 5 个正有限数值")
        q = np.clip(np.radians(np.asarray(initial_joints_deg[:5], dtype=float)), limits[:, 0], limits[:, 1])
        best_q = q.copy()
        best_xyz = self._forward_rad(q)[0][:3, 3]
        best_error = float(np.linalg.norm(target - best_xyz))
        adaptive_damping = damping

        for iteration in range(1, max_iterations + 1):
            transform, frames = self._forward_rad(q, frames=True)
            xyz = transform[:3, 3]
            error = target - xyz
            error_norm = float(np.linalg.norm(error))
            if error_norm < best_error:
                best_error, best_q, best_xyz = error_norm, q.copy(), xyz.copy()
            if error_norm <= tolerance_m:
                return IKResult(
                    True,
                    tuple(float(value) for value in np.degrees(q)),
                    tuple(float(value) for value in xyz),
                    error_norm,
                    iteration,
                    "converged",
                )

            jacobian = np.column_stack(
                [np.cross(axis, xyz - origin) for origin, axis in frames]
            )

            def weighted_delta(weights: np.ndarray) -> np.ndarray:
                inverse_weights = 1.0 / weights
                weighted_transpose = inverse_weights[:, None] * jacobian.T
                regularized = (
                    jacobian @ weighted_transpose
                    + adaptive_damping**2 * np.eye(3)
                )
                return weighted_transpose @ np.linalg.solve(regularized, error)

            try:
                # First estimate the requested direction.  Only penalize a
                # near-limit joint when that estimate would move farther into
                # its closest limit; motion away from the limit stays free.
                preliminary = weighted_delta(base_weights)
                effective_weights = base_weights.copy()
                spans = limits[:, 1] - limits[:, 0]
                lower_room = (q - limits[:, 0]) / spans
                upper_room = (limits[:, 1] - q) / spans
                for index, step in enumerate(preliminary):
                    room = lower_room[index] if step < 0.0 else upper_room[index]
                    if room < 0.15:
                        proximity = (0.15 - max(0.0, room)) / 0.15
                        effective_weights[index] *= 1.0 + 40.0 * proximity**2
                delta = weighted_delta(effective_weights)
            except np.linalg.LinAlgError:
                break
            largest = float(np.max(np.abs(delta)))
            if largest > max_iteration_step_rad:
                delta *= max_iteration_step_rad / largest

            improved = False
            scale = 1.0
            for _ in range(12):
                candidate = np.clip(q + scale * delta, limits[:, 0], limits[:, 1])
                candidate_xyz = self._forward_rad(candidate)[0][:3, 3]
                if float(np.linalg.norm(target - candidate_xyz)) < error_norm - 1e-10:
                    q = candidate
                    adaptive_damping = max(damping, adaptive_damping * 0.75)
                    improved = True
                    break
                scale *= 0.5
            if not improved:
                adaptive_damping *= 2.0
                if adaptive_damping > 0.8:
                    break

        return IKResult(
            False,
            tuple(float(value) for value in np.degrees(best_q)),
            tuple(float(value) for value in best_xyz),
            best_error,
            max_iterations,
            "位置残差超过容差；目标可能超出工作空间、受关节限位约束或位于奇异点",
        )


class JointDriver(Protocol):
    def get_joint_state(self) -> tuple[tuple[float, ...], float]: ...

    def send_joint_target(self, joints_deg: Sequence[float], gripper_raw: float) -> None: ...


class DryRunDriver:
    def __init__(self, joints_deg: Sequence[float], gripper_raw: float = 200.0) -> None:
        self.joints_deg = tuple(float(value) for value in joints_deg)
        self.gripper_raw = float(gripper_raw)
        self.history: list[tuple[float, ...]] = []

    def get_joint_state(self) -> tuple[tuple[float, ...], float]:
        return self.joints_deg, self.gripper_raw

    def send_joint_target(self, joints_deg: Sequence[float], gripper_raw: float) -> None:
        self.joints_deg = tuple(float(value) for value in joints_deg)
        self.gripper_raw = float(gripper_raw)
        self.history.append(self.joints_deg)


@dataclass(frozen=True)
class MotionResult:
    direction: str
    requested_m: float
    executed_m: float
    full_success: bool
    waypoint_count: int
    start_xyz: tuple[float, float, float]
    final_xyz: tuple[float, float, float]
    message: str


class CartesianController:
    def __init__(
        self,
        kinematics: URDFKinematics,
        driver: JointDriver,
        *,
        joint_limits_deg: Sequence[tuple[float, float]] | None = None,
        cartesian_resolution_m: float = 0.002,
        tolerance_m: float = 0.0008,
        max_joint_step_deg: float = 12.0,
        command_interval_s: float = 0.05,
    ) -> None:
        self.kinematics = kinematics
        self.driver = driver
        self.joint_limits_deg = tuple(joint_limits_deg or kinematics.joint_limits_deg)
        self.cartesian_resolution_m = float(cartesian_resolution_m)
        self.tolerance_m = float(tolerance_m)
        self.max_joint_step_deg = float(max_joint_step_deg)
        self.command_interval_s = float(command_interval_s)
        self.joints_deg: tuple[float, ...] = ()
        self.gripper_raw = 0.0
        self.cartesian_reference_xyz: tuple[float, float, float] | None = None

    def initialize(self) -> None:
        self.joints_deg, self.gripper_raw = self.driver.get_joint_state()
        self._check_limits(self.joints_deg)

    def _check_limits(self, joints_deg: Sequence[float]) -> None:
        for name, value, (lower, upper) in zip(
            ARM_JOINT_NAMES, joints_deg, self.joint_limits_deg, strict=True
        ):
            if not lower - 1e-6 <= value <= upper + 1e-6:
                raise ValueError(
                    f"{name} 当前 {value:.2f}° 超出软件安全范围 {lower:.2f}°~{upper:.2f}°"
                )

    def _acceptable(self, result: IKResult, reference: Sequence[float]) -> tuple[bool, str]:
        if not result.success:
            return False, result.reason
        jump = max(abs(a - b) for a, b in zip(result.joints_deg, reference, strict=True))
        if jump > self.max_joint_step_deg:
            return False, f"单路点关节变化 {jump:.2f}° 超过 {self.max_joint_step_deg:.2f}°"
        return True, ""

    def _solve(
        self,
        target: np.ndarray,
        reference: Sequence[float],
        joint_motion_weights: Sequence[float] | None = None,
    ) -> IKResult:
        return self.kinematics.inverse(
            target,
            reference,
            joint_limits_deg=self.joint_limits_deg,
            tolerance_m=self.tolerance_m,
            joint_motion_weights=joint_motion_weights,
        )

    def _send(self, result: IKResult) -> None:
        self._check_limits(result.joints_deg)
        self.driver.send_joint_target(result.joints_deg, self.gripper_raw)
        self.joints_deg = result.joints_deg
        if self.command_interval_s > 0:
            time.sleep(self.command_interval_s)

    def _last_reachable(
        self,
        reachable_xyz: np.ndarray,
        failed_xyz: np.ndarray,
        reference: Sequence[float] | None = None,
        joint_motion_weights: Sequence[float] | None = None,
    ) -> tuple[float, IKResult | None]:
        low, high = 0.0, 1.0
        best: IKResult | None = None
        reference = tuple(reference or self.joints_deg)
        for _ in range(15):
            fraction = (low + high) / 2.0
            target = reachable_xyz + fraction * (failed_xyz - reachable_xyz)
            candidate = self._solve(target, reference, joint_motion_weights)
            acceptable, _ = self._acceptable(candidate, reference)
            if acceptable:
                low, best = fraction, candidate
            else:
                high = fraction
        return low, best

    def move(
        self,
        direction: str,
        distance_m: float,
        *,
        final_target_only: bool = False,
        stream_targets: bool = False,
    ) -> MotionResult:
        if direction not in DIRECTIONS:
            raise ValueError(f"未知方向：{direction}")
        if distance_m <= 0:
            raise ValueError("移动距离必须大于 0")
        if not self.joints_deg:
            raise RuntimeError("控制器尚未初始化")
        if final_target_only and stream_targets:
            raise ValueError("单次终点和流式轨迹不能同时启用")

        unit = np.asarray(DIRECTIONS[direction], dtype=float)
        start = np.asarray(self.kinematics.position(self.joints_deg))
        waypoint_count = max(1, math.ceil(distance_m / self.cartesian_resolution_m))
        last_xyz = start.copy()
        reference = self.joints_deg
        final_result: IKResult | None = None
        planned_results: list[IKResult] = []
        sent = 0
        failure_reason = ""
        joint_motion_weights = DIRECTION_JOINT_MOTION_WEIGHTS.get(direction)

        for index in range(1, waypoint_count + 1):
            target = start + unit * distance_m * index / waypoint_count
            result = self._solve(target, reference, joint_motion_weights)
            acceptable, reason = self._acceptable(result, reference)
            if not acceptable:
                fraction, partial = self._last_reachable(
                    last_xyz, target, reference, joint_motion_weights
                )
                if partial is not None and fraction * float(np.linalg.norm(target - last_xyz)) > 0.0002:
                    if not final_target_only and not stream_targets:
                        self._send(partial)
                    reference = partial.joints_deg
                    final_result = partial
                    planned_results.append(partial)
                    last_xyz = np.asarray(partial.achieved_xyz)
                    sent += 1
                failure_reason = reason
                break
            if not final_target_only and not stream_targets:
                self._send(result)
            reference = result.joints_deg
            final_result = result
            planned_results.append(result)
            last_xyz = np.asarray(result.achieved_xyz)
            sent += 1

        # For real hardware a single synchronized final goal is much smoother
        # than stop-starting at every Cartesian planning waypoint. The
        # intermediate waypoints above still validate reachability and keep the
        # IK solution on the requested branch.
        if final_target_only and final_result is not None:
            self._send(final_result)
        elif stream_targets and planned_results:
            set_streaming = getattr(self.driver, "set_streaming", None)
            if set_streaming is not None:
                set_streaming(True)
            try:
                for planned in planned_results:
                    self._send(planned)
            finally:
                if set_streaming is not None:
                    set_streaming(False)

        executed = max(0.0, float(np.dot(last_xyz - start, unit)))
        full = sent >= waypoint_count and not failure_reason
        if full and stream_targets:
            message = "直线 IK 已预计算，并以固定节拍连续发送全部路点"
        elif full and final_target_only:
            message = "直线 IK 路径已验证，已发送单次同步终点"
        elif full:
            message = "完整直线已执行"
        else:
            message = f"IK 无解：{failure_reason}；已停在同一直线的最后可达点"
        return MotionResult(
            direction=direction,
            requested_m=float(distance_m),
            executed_m=min(float(distance_m), executed),
            full_success=full,
            waypoint_count=sent,
            start_xyz=tuple(float(value) for value in start),
            final_xyz=tuple(float(value) for value in last_xyz),
            message=message,
        )

    def move_to_xyz(
        self,
        target_xyz: Sequence[float],
        *,
        stream_targets: bool = False,
        joint_motion_weights: Sequence[float] | None = None,
    ) -> MotionResult:
        """Move along one straight Cartesian segment to an absolute XYZ target."""
        if not self.joints_deg:
            raise RuntimeError("控制器尚未初始化")
        target = np.asarray(tuple(float(value) for value in target_xyz), dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("末端目标必须是 3 个有限坐标")

        start = np.asarray(self.kinematics.position(self.joints_deg))
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance <= self.tolerance_m:
            return MotionResult(
                direction="target",
                requested_m=distance,
                executed_m=distance,
                full_success=True,
                waypoint_count=0,
                start_xyz=tuple(float(value) for value in start),
                final_xyz=tuple(float(value) for value in start),
                message="现场末端已在目标容差内",
            )

        unit = delta / distance
        waypoint_count = max(1, math.ceil(distance / self.cartesian_resolution_m))
        last_xyz = start.copy()
        reference = self.joints_deg
        final_result: IKResult | None = None
        planned_results: list[IKResult] = []
        sent = 0
        failure_reason = ""

        for index in range(1, waypoint_count + 1):
            waypoint = start + delta * index / waypoint_count
            result = self._solve(waypoint, reference, joint_motion_weights)
            acceptable, reason = self._acceptable(result, reference)
            if not acceptable:
                fraction, partial = self._last_reachable(
                    last_xyz, waypoint, reference, joint_motion_weights
                )
                if (
                    partial is not None
                    and fraction * float(np.linalg.norm(waypoint - last_xyz)) > 0.0002
                ):
                    if not stream_targets:
                        self._send(partial)
                    reference = partial.joints_deg
                    final_result = partial
                    planned_results.append(partial)
                    last_xyz = np.asarray(partial.achieved_xyz)
                    sent += 1
                failure_reason = reason
                break
            if not stream_targets:
                self._send(result)
            reference = result.joints_deg
            final_result = result
            planned_results.append(result)
            last_xyz = np.asarray(result.achieved_xyz)
            sent += 1

        if stream_targets and planned_results:
            set_streaming = getattr(self.driver, "set_streaming", None)
            if set_streaming is not None:
                set_streaming(True)
            try:
                for planned in planned_results:
                    self._send(planned)
            finally:
                if set_streaming is not None:
                    set_streaming(False)

        executed = max(0.0, float(np.dot(last_xyz - start, unit)))
        full = sent >= waypoint_count and not failure_reason
        message = (
            "笛卡尔目标已沿直线平滑执行"
            if full
            else f"IK 无解：{failure_reason}；已停在同一直线的最后可达点"
        )
        return MotionResult(
            direction="target",
            requested_m=distance,
            executed_m=min(distance, executed),
            full_success=full,
            waypoint_count=sent,
            start_xyz=tuple(float(value) for value in start),
            final_xyz=tuple(float(value) for value in last_xyz),
            message=message,
        )
