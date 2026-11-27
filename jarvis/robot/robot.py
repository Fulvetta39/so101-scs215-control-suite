from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Protocol

from .config import RobotSettings

LOGGER = logging.getLogger(__name__)

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class RobotError(RuntimeError):
    """Raised when an arm command cannot be completed safely."""


class ArmBackend(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_positions(self) -> dict[str, float]: ...
    def send_positions(self, positions: dict[str, float]) -> dict[str, float]: ...
    def stream_positions(self, positions: dict[str, float]) -> None: ...
    def hold(self) -> None: ...
    def resume(self) -> None: ...


class PoseStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def load(self) -> dict[str, dict[str, float]]:
        with self._lock:
            if not self.path.exists():
                return {}
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RobotError(f"姿态文件无法读取：{exc}") from exc
        result: dict[str, dict[str, float]] = {}
        for name, pose in payload.items():
            if not isinstance(pose, dict) or not all(joint in pose for joint in JOINTS):
                raise RobotError(f"姿态 {name!r} 不包含全部 SO101 关节。")
            result[str(name)] = {joint: float(pose[joint]) for joint in JOINTS}
        return result

    def names(self) -> list[str]:
        return sorted(self.load())

    def save(self, name: str, positions: dict[str, float]) -> dict[str, float]:
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 64:
            raise RobotError("姿态名称必须为 1 到 64 个字符。")
        pose = {joint: float(positions[joint]) for joint in JOINTS}
        with self._lock:
            current = self.load()
            current[clean_name] = pose
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(current, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        return pose


class MockArm:
    def __init__(self, initial: dict[str, float]):
        self.initial = dict(initial)
        self.positions = dict(initial)
        self._lock = threading.Lock()

    def connect(self) -> None:
        LOGGER.info("Mock SO101 connected")

    def disconnect(self) -> None:
        LOGGER.info("Mock SO101 disconnected")

    def get_positions(self) -> dict[str, float]:
        with self._lock:
            return dict(self.positions)

    def send_positions(self, positions: dict[str, float]) -> dict[str, float]:
        with self._lock:
            for joint, value in positions.items():
                if joint not in JOINTS:
                    raise RobotError(f"未知关节 {joint!r}")
                self.positions[joint] = float(value)
            return dict(self.positions)

    def stream_positions(self, positions: dict[str, float]) -> None:
        self.send_positions(positions)

    def hold(self) -> None:
        return None

    def resume(self) -> None:
        return None


class LeRobotArm:
    def __init__(self, settings: RobotSettings):
        self.settings = settings
        self._robot = None
        self._lock = threading.RLock()

    def connect(self) -> None:
        if self._robot is not None:
            return
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise RobotError(
                "未安装 LeRobot 真机依赖。请运行："
                'pip install -e ".[robot]"'
            ) from exc

        config = SO101FollowerConfig(
            port=self.settings.port,
            id=self.settings.id,
            max_relative_target=self.settings.max_relative_target_degrees,
            disable_torque_on_disconnect=True,
        )
        robot = SO101Follower(config)
        try:
            # Connect the bus without enabling torque. The existing course
            # calibration is loaded from disk and verified before any target is sent.
            robot.bus.connect()
            if not robot.is_calibrated:
                raise RobotError(
                    "SO215 尚未加载校准文件。请先使用课程控制套件完成校准。"
                )
            self._hold_and_enable(robot)
        except Exception:
            try:
                robot.disconnect()
            except Exception:
                pass
            raise
        self._robot = robot
        LOGGER.info("SO101 connected on %s", self.settings.port)

    def disconnect(self) -> None:
        with self._lock:
            if self._robot is not None:
                self._robot.disconnect()
                self._robot = None

    def _require_robot(self):
        if self._robot is None:
            raise RobotError("SO101 尚未连接。")
        return self._robot

    def get_positions(self) -> dict[str, float]:
        with self._lock:
            robot = self._require_robot()
            observation = robot.get_observation()
        return {
            joint: float(observation[f"{joint}.pos"])
            for joint in JOINTS
            if f"{joint}.pos" in observation
        }

    def send_positions(self, positions: dict[str, float]) -> dict[str, float]:
        action = {}
        for joint, value in positions.items():
            if joint not in JOINTS:
                raise RobotError(f"未知关节 {joint!r}")
            action[f"{joint}.pos"] = float(value)
        with self._lock:
            robot = self._require_robot()
            sent = robot.send_action(action)
        return {
            key.removesuffix(".pos"): float(value)
            for key, value in sent.items()
            if key.endswith(".pos")
        }

    def stream_positions(self, positions: dict[str, float]) -> None:
        clean = {
            joint: float(value)
            for joint, value in positions.items()
            if joint in JOINTS
        }
        if len(clean) != len(positions):
            raise RobotError("平滑伺服包含未知关节。")
        with self._lock:
            robot = self._require_robot()
            robot.bus.sync_write("Goal_Position", clean)

    def hold(self) -> None:
        with self._lock:
            robot = self._require_robot()
            robot.bus.disable_torque()

    def resume(self) -> None:
        with self._lock:
            robot = self._require_robot()
            self._hold_and_enable(robot)

    def _hold_and_enable(self, robot) -> None:
        robot.bus.disable_torque()
        hold = {
            joint: robot.bus.read("Present_Position", joint, normalize=False)
            for joint in JOINTS
        }
        robot.bus.sync_write("Goal_Position", hold, normalize=False)
        time.sleep(0.2)
        goal = {
            joint: robot.bus.read(
                "Goal_Position",
                joint,
                normalize=False,
                num_retry=2,
            )
            for joint in JOINTS
        }
        # SCS215 can round one register step while latching the hold target.
        mismatched = {}
        tolerated = {}
        for joint in JOINTS:
            delta = abs(goal[joint] - hold[joint])
            calibration = robot.calibration[joint]
            at_boundary = goal[joint] in (
                calibration.range_min,
                calibration.range_max,
            )
            # With torque disabled, a joint can rest a few SCS register
            # units outside its calibrated range. If the hold target is
            # clamped to that range boundary, allow up to three units so
            # enabling torque makes only a sub-degree correction.
            tolerance = 3 if at_boundary else 1
            if delta > tolerance:
                mismatched[joint] = (hold[joint], goal[joint])
            elif delta > 1:
                tolerated[joint] = (hold[joint], goal[joint])
        if mismatched:
            detail = ", ".join(
                f"{joint}: present={present}, goal={target}"
                for joint, (present, target) in mismatched.items()
            )
            raise RobotError(
                f"当前位置保持目标回读不一致，拒绝开启力矩。{detail}"
            )
        if tolerated:
            detail = ", ".join(
                f"{joint}: present={present}, goal={target}"
                for joint, (present, target) in tolerated.items()
            )
            LOGGER.warning("Accepting clamped hold-target tolerance: %s", detail)
        robot.configure()


def make_backend(settings: RobotSettings) -> ArmBackend:
    if settings.backend == "lerobot":
        return LeRobotArm(settings)
    if settings.backend != "mock":
        raise RobotError(f"未知机械臂后端：{settings.backend!r}")

    poses = PoseStore(settings.poses_file).load()
    initial = poses.get("home")
    if initial is None:
        initial = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -8.0,
            "elbow_flex": 22.0,
            "wrist_flex": -5.0,
            "wrist_roll": 0.0,
            "gripper": 70.0,
        }
    return MockArm(initial)


class ArmController:
    """Joint-space controller with interpolation and conservative motion limits."""

    SMOOTH_RATE_HZ = 30.0
    SMOOTH_MAX_VELOCITY = 10.0
    SMOOTH_ACCELERATION = 30.0
    SMOOTH_TARGET_LEAD = 2.0
    SMOOTH_DEADBAND = 0.03

    def __init__(self, settings: RobotSettings):
        self.settings = settings
        self.backend = make_backend(settings)
        self.poses = PoseStore(settings.poses_file)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._connected = False
        self._last_error: str | None = None
        self._smooth_lock = threading.Lock()
        self._smooth_stop = threading.Event()
        self._smooth_thread: threading.Thread | None = None
        self._smooth_target: dict[str, float] = {}
        self._smooth_command: dict[str, float] = {}
        self._smooth_velocity: dict[str, float] = {}

    def connect(self) -> None:
        with self._lock:
            self.backend.connect()
            self._connected = True
            self._stop.clear()

    def disconnect(self) -> None:
        self.stop_smooth_tracking()
        with self._lock:
            try:
                self.backend.hold()
            except Exception:
                LOGGER.exception("Could not hold pose during disconnect")
            self.backend.disconnect()
            self._connected = False

    def _ensure_ready(self) -> None:
        if not self._connected:
            raise RobotError("机械臂尚未连接。")
        if self._stop.is_set():
            raise RobotError("急停已触发。请检查现场并重启 Agent 后再操作。")

    def clamp_position(self, joint: str, value: float) -> float:
        limits = self.settings.joint_limits.get(joint)
        if limits is None:
            raise RobotError(f"关节 {joint!r} 没有配置安全范围。")
        return max(limits[0], min(limits[1], float(value)))

    def positions(self) -> dict[str, float]:
        with self._lock:
            self._ensure_ready()
            return self.backend.get_positions()

    def update_smooth_target(self, deltas_degrees: dict[str, float]) -> dict[str, object]:
        """Accumulate a visual-servo target consumed by a 30 Hz software loop."""
        allowed = {"shoulder_pan", "shoulder_lift", "wrist_flex"}
        if not deltas_degrees or not set(deltas_degrees).issubset(allowed):
            raise RobotError("平滑伺服只能控制规定的三个关节。")
        with self._lock:
            self._ensure_ready()
            current = self.backend.get_positions()

        with self._smooth_lock:
            thread_alive = self._smooth_thread is not None and self._smooth_thread.is_alive()
            if not thread_alive or self._smooth_stop.is_set():
                self._smooth_target = dict(current)
                self._smooth_command = dict(current)
                self._smooth_velocity = {joint: 0.0 for joint in allowed}
                self._smooth_stop.clear()
                self._smooth_thread = threading.Thread(
                    target=self._smooth_loop,
                    name="so101-smooth-servo",
                    daemon=True,
                )
                self._smooth_thread.start()

            for joint, value in deltas_degrees.items():
                requested = float(value)
                if not math.isfinite(requested):
                    raise RobotError("平滑伺服增量必须是有限数值。")
                requested = max(-6.0, min(6.0, requested))
                units = requested * self.settings.control_units_per_degree
                # Each visual update replaces the short look-ahead target
                # relative to the current command. Accumulating every frame
                # causes the arm to keep chasing stale errors after the face
                # has already returned to the center, producing oscillation.
                target = self.clamp_position(
                    joint,
                    self._smooth_command[joint] + units,
                )
                lower = self._smooth_command[joint] - self.SMOOTH_TARGET_LEAD
                upper = self._smooth_command[joint] + self.SMOOTH_TARGET_LEAD
                self._smooth_target[joint] = max(lower, min(upper, target))
            accepted = {
                joint: round(
                    (self._smooth_target[joint] - current[joint])
                    / self.settings.control_units_per_degree,
                    3,
                )
                for joint in deltas_degrees
            }
        return {
            "ok": True,
            "message": "平滑伺服目标已更新。",
            "tracking": True,
            "accepted_delta_degrees": accepted,
        }

    def stop_smooth_tracking(self) -> None:
        self._smooth_stop.set()
        with self._smooth_lock:
            thread = self._smooth_thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=1.0)
        with self._smooth_lock:
            if self._smooth_thread is thread:
                self._smooth_thread = None
                self._smooth_target.clear()
                self._smooth_command.clear()
                self._smooth_velocity.clear()

    def _smooth_loop(self) -> None:
        period = 1.0 / self.SMOOTH_RATE_HZ
        while not self._smooth_stop.is_set():
            started = time.monotonic()
            with self._smooth_lock:
                target = dict(self._smooth_target)
                command = dict(self._smooth_command)
                velocities = dict(self._smooth_velocity)
            if not target:
                break

            next_positions: dict[str, float] = {}
            next_velocities = dict(velocities)
            for joint, destination in target.items():
                error = destination - command[joint]
                desired_velocity = max(
                    -self.SMOOTH_MAX_VELOCITY,
                    min(self.SMOOTH_MAX_VELOCITY, error * 8.0),
                )
                max_change = self.SMOOTH_ACCELERATION * period
                velocity = velocities.get(joint, 0.0)
                velocity += max(
                    -max_change,
                    min(max_change, desired_velocity - velocity),
                )
                if abs(error) <= self.SMOOTH_DEADBAND:
                    velocity = 0.0
                step = velocity * period
                if abs(step) >= abs(error):
                    step = error
                    velocity = 0.0
                if abs(step) < 1e-6:
                    next_velocities[joint] = velocity
                    continue
                next_positions[joint] = self.clamp_position(
                    joint,
                    command[joint] + step,
                )
                next_velocities[joint] = velocity

            if next_positions:
                try:
                    self.backend.stream_positions(next_positions)
                except Exception:
                    LOGGER.exception("Smooth servo stream failed")
                    self.emergency_stop()
                    break
                with self._smooth_lock:
                    self._smooth_command.update(next_positions)
                    self._smooth_velocity.update(next_velocities)
            else:
                with self._smooth_lock:
                    self._smooth_velocity.update(next_velocities)
            elapsed = time.monotonic() - started
            self._smooth_stop.wait(max(0.001, period - elapsed))

    def move_joint(self, joint: str, delta_degrees: float) -> dict[str, object]:
        self.stop_smooth_tracking()
        if joint not in JOINTS:
            raise RobotError(f"未知关节 {joint!r}")
        requested = float(delta_degrees)
        if not math.isfinite(requested):
            raise RobotError("关节增量必须是有限数值。")
        requested = max(
            -self.settings.max_tool_delta_degrees,
            min(self.settings.max_tool_delta_degrees, requested),
        )
        requested_units = requested * self.settings.control_units_per_degree

        with self._lock:
            self._ensure_ready()
            start = self.backend.get_positions()
            target = self.clamp_position(joint, start[joint] + requested_units)
            actual_units = target - start[joint]
            if abs(actual_units) < 1e-6:
                return self._motion_result(start, joint, 0.0, False, "关节已到达安全边界。")

            steps = max(1, math.ceil(abs(actual_units) / 2.0))
            for step in range(1, steps + 1):
                if self._stop.is_set():
                    self.backend.hold()
                    raise RobotError("运动已由急停中断。")
                positions = {joint: start[joint] + actual_units * step / steps}
                self.backend.send_positions(positions)
                time.sleep(0.08)

            final = self.backend.get_positions()
            actual_degrees = actual_units / self.settings.control_units_per_degree
            return self._motion_result(final, joint, actual_degrees, False, "关节运动完成。")

    def move_joints(self, deltas_degrees: dict[str, float]) -> dict[str, object]:
        """Apply a small simultaneous joint correction for visual servoing."""
        self.stop_smooth_tracking()
        if not deltas_degrees:
            raise RobotError("关节修正不能为空。")
        if any(joint not in JOINTS for joint in deltas_degrees):
            raise RobotError("包含未知关节。")
        clean: dict[str, float] = {}
        for joint, value in deltas_degrees.items():
            if joint == "gripper":
                raise RobotError("视觉伺服接口不能直接控制夹爪。")
            requested = float(value)
            if not math.isfinite(requested):
                raise RobotError("关节增量必须是有限数值。")
            requested = max(
                -self.settings.max_tool_delta_degrees,
                min(self.settings.max_tool_delta_degrees, requested),
            )
            if abs(requested) >= 1e-4:
                clean[joint] = requested
        if not clean:
            return {
                "ok": True,
                "message": "修正量位于死区内。",
                "positions": self.positions(),
                "applied_degrees": {},
            }

        with self._lock:
            self._ensure_ready()
            start = self.backend.get_positions()
            target = dict(start)
            requested_units = {
                joint: delta * self.settings.control_units_per_degree
                for joint, delta in clean.items()
            }
            for joint, delta in requested_units.items():
                target[joint] = self.clamp_position(joint, start[joint] + delta)
            actual_units = {
                joint: target[joint] - start[joint] for joint in requested_units
            }
            largest = max(abs(value) for value in actual_units.values())
            if largest < 1e-6:
                return {
                    "ok": True,
                    "message": "修正量已到达安全边界。",
                    "positions": {
                        key: round(value, 2) for key, value in start.items()
                    },
                    "applied_degrees": {},
                }

            steps = max(1, math.ceil(largest / 1.0))
            steps = min(steps, 8)
            for step in range(1, steps + 1):
                if self._stop.is_set():
                    self.backend.hold()
                    raise RobotError("视觉伺服已由急停中断。")
                ratio = step / steps
                positions = {
                    joint: start[joint] + actual_units[joint] * ratio
                    for joint in actual_units
                }
                self.backend.send_positions(positions)
                time.sleep(0.06)
            time.sleep(0.12)
            final = self.backend.get_positions()
            applied = {
                joint: round(
                    (final[joint] - start[joint])
                    / self.settings.control_units_per_degree,
                    3,
                )
                for joint in actual_units
            }
            if all(abs(value) < 0.15 for value in applied.values()):
                return {
                    "ok": False,
                    "message": "目标已发送，但关节位置没有发生变化。",
                    "positions": {
                        key: round(value, 2) for key, value in final.items()
                    },
                    "applied_degrees": applied,
                }
            return {
                "ok": True,
                "message": "视觉伺服修正完成。",
                "positions": {key: round(value, 2) for key, value in final.items()},
                "applied_degrees": applied,
            }

    def move_to_pose(self, name: str) -> dict[str, object]:
        self.stop_smooth_tracking()
        pose = self.poses.load().get(name)
        if pose is None:
            known = ", ".join(self.poses.names()) or "无"
            raise RobotError(f"未知姿态 {name!r}。已校准姿态：{known}")

        with self._lock:
            self._ensure_ready()
            start = self.backend.get_positions()
            target = {joint: self.clamp_position(joint, value) for joint, value in pose.items()}
            largest_units = max(abs(target[joint] - start[joint]) for joint in JOINTS)
            steps = max(1, math.ceil(largest_units / 3.0))
            steps = min(steps, 60)
            for step in range(1, steps + 1):
                if self._stop.is_set():
                    self.backend.hold()
                    raise RobotError("姿态运动已由急停中断。")
                ratio = step / steps
                positions = {
                    joint: start[joint] + (target[joint] - start[joint]) * ratio
                    for joint in JOINTS
                }
                self.backend.send_positions(positions)
                time.sleep(0.06)
            final = self.backend.get_positions()
            largest_degrees = largest_units / self.settings.control_units_per_degree
            return self._motion_result(
                final, None, largest_degrees, False, f"已到达姿态 {name}。"
            )

    def set_gripper(self, open_percent: float) -> dict[str, object]:
        self.stop_smooth_tracking()
        value = max(0.0, min(100.0, float(open_percent)))
        with self._lock:
            self._ensure_ready()
            start = self.backend.get_positions()
            steps = max(1, math.ceil(abs(value - start["gripper"]) / 4.0))
            for step in range(1, steps + 1):
                if self._stop.is_set():
                    self.backend.hold()
                    raise RobotError("夹爪运动已由急停中断。")
                positions = {
                    "gripper": start["gripper"]
                    + (value - start["gripper"]) * step / steps
                }
                self.backend.send_positions(positions)
                time.sleep(0.06)
            final = self.backend.get_positions()
            return self._motion_result(final, "gripper", value - start["gripper"], False, "夹爪已调整。")

    def emergency_stop(self) -> None:
        self._smooth_stop.set()
        self._stop.set()
        with self._lock:
            if self._connected:
                try:
                    self.backend.hold()
                except Exception:
                    LOGGER.exception("Hold command failed during emergency stop")
        LOGGER.warning("SO101 emergency stop triggered")

    def resume_from_emergency(self) -> dict[str, object]:
        with self._lock:
            if not self._connected:
                raise RobotError("机械臂尚未连接。")
            self.backend.resume()
            self._stop.clear()
            self._last_error = None
        return {
            "ok": True,
            "message": "机械臂已恢复力矩，并保持当前位置。",
            "stopped": False,
        }

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def save_current_pose(self, name: str) -> dict[str, float]:
        return self.poses.save(name, self.positions())

    def _motion_result(
        self,
        positions: dict[str, float],
        joint: str | None,
        delta: float,
        refresh_image: bool,
        message: str,
    ) -> dict[str, object]:
        return {
            "ok": True,
            "message": message,
            "joint": joint,
            "delta_degrees": round(delta, 2),
            "positions": {key: round(value, 2) for key, value in positions.items()},
            "refresh_image": refresh_image,
        }

    def status(self) -> dict[str, object]:
        with self._smooth_lock:
            smooth_tracking = (
                self._smooth_thread is not None and self._smooth_thread.is_alive()
            )
        return {
            "backend": self.settings.backend,
            "connected": self._connected,
            "stopped": self._stop.is_set(),
            "port": self.settings.port,
            "poses": self.poses.names(),
            "last_error": self._last_error,
            "smooth_tracking": smooth_tracking,
        }

