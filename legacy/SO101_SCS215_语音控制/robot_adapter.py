"""Dry-run and persistent real-hardware executors for voice commands."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Sequence

from command_parser import DIRECTION_LABELS, VoiceCommand


ROOT = Path(__file__).resolve().parent.parent
IK_DIR = ROOT / "SO101_SCS215_IK键盘直线控制"
IK_FILE = IK_DIR / "ik_keyboard_control.py"
DEFAULT_URDF = ROOT / "SO101" / "so101_new_calib.urdf"
DEFAULT_MAPPING = IK_DIR / "joint_mapping.json"
DEFAULT_VOICE_CALIBRATION = Path(__file__).resolve().with_name("voice_calibration.json")


def load_direction_biases(path: Path = DEFAULT_VOICE_CALIBRATION) -> dict[str, tuple[int, ...]]:
    if not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError("语音标定文件必须是 schema 1")
    result: dict[str, tuple[int, ...]] = {}
    for direction, values in data.get("direction_load_bias_raw", {}).items():
        if direction not in {"forward", "back", "left", "right", "up", "down"}:
            raise ValueError(f"语音标定中存在未知方向：{direction}")
        if (
            not isinstance(values, list)
            or len(values) != 5
            or any(not isinstance(value, int) for value in values)
        ):
            raise ValueError(f"{direction} 负载偏置必须是 5 个整数")
        result[direction] = tuple(values)
    return result


def _load_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise RuntimeError(f"找不到现有控制器：{path}")
    path_text = str(path.parent)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载控制器：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DryRunExecutor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.history: list[VoiceCommand] = []

    def __enter__(self) -> "DryRunExecutor":
        print("当前是模拟执行：不连接串口，不写目标，不改变扭矩。")
        return self

    def __exit__(self, *_: object) -> None:
        print(f"模拟会话结束，共记录 {len(self.history)} 条可执行指令。")

    def execute(self, command: VoiceCommand) -> None:
        self.history.append(command)
        if command.kind == "pose":
            pose = self.config["poses"][command.target]
            print(
                f"[模拟] 移动到 {command.target}：{pose['positions']}，"
                f"建议切换 {pose.get('transition_s', 1.5):.2f}s"
            )
        elif command.kind == "action":
            steps = self.config["actions"][command.target]["steps"]
            print(f"[模拟] 执行 {command.target}，共 {len(steps)} 步。")
            for index, step in enumerate(steps, start=1):
                print(
                    f"  {index}. {step['pose']}，切换 {step['transition_s']:.2f}s，"
                    f"停留 {step['hold_s']:.2f}s"
                )
        elif command.kind == "move":
            print(
                f"[模拟] 末端向{DIRECTION_LABELS[command.direction]}"
                f"移动 {command.distance_cm:.2f} cm。"
            )
        elif command.kind == "stop":
            print("[模拟] 停止并保持当前位置。")


class RealRobotSession:
    """One checked serial/torque session shared by every recognized command."""

    def __init__(
        self,
        config: dict,
        *,
        port: str | None,
        urdf: Path = DEFAULT_URDF,
        mapping: Path = DEFAULT_MAPPING,
        voice_calibration: Path = DEFAULT_VOICE_CALIBRATION,
        interval_s: float = 0.05,
        max_pose_speed_raw_s: float = 90.0,
    ) -> None:
        self.config = config
        self.port = port
        self.urdf = Path(urdf)
        self.mapping = Path(mapping)
        self.voice_calibration = Path(voice_calibration)
        self.interval_s = float(interval_s)
        self.max_pose_speed_raw_s = float(max_pose_speed_raw_s)
        self.ik = None
        self.bus = None
        self.driver = None
        self.controller = None

    def __enter__(self) -> "RealRobotSession":
        self.ik = _load_module("voice_ik_keyboard_control", IK_FILE)
        selected_port = self.ik.find_port(self.port)
        kinematics = self.ik.URDFKinematics(self.urdf)
        mappings = self.ik.load_mappings(self.mapping)
        self.bus = self.ik.SCS215Bus(selected_port)
        self.bus.__enter__()
        try:
            states = self.bus.snapshot()
            print(f"真机只读检查：{selected_port} @ {self.ik.BAUD}")
            self.ik.print_states(states)
            self.ik.validate_models(states)
            self.driver = self.ik.SCS215ArmDriver(
                self.bus, states, mappings, kinematics.joint_limits_deg
            )
            direction_biases = load_direction_biases(self.voice_calibration)
            for direction, values in direction_biases.items():
                for index, value in enumerate(values):
                    limit = (
                        self.ik.ID3_LOAD_BIAS_MAX_RAW
                        if index == 2
                        else self.ik.LOAD_BIAS_MAX_RAW
                    )
                    if abs(value) > limit:
                        raise ValueError(
                            f"{direction} ID{index + 1} 负载偏置 {value} 超过上限 {limit}"
                        )
                self.driver.direction_load_bias[direction] = values
            if direction_biases:
                print(f"已加载语音方向标定：{direction_biases}")
            print(
                "软件安全原始范围："
                + "，".join(
                    f"ID{i + 1} {minimum}~{maximum}"
                    for i, (minimum, maximum) in enumerate(
                        self.driver.safe_raw_ranges
                    )
                )
            )
            self._validate_real_pose_config()
            self.driver.hold_then_enable()
            args = Namespace(
                resolution_mm=2.0,
                tolerance_mm=0.8,
                max_joint_step=12.0,
                interval=self.interval_s,
            )
            self.controller = self.ik.make_controller(
                kinematics,
                self.driver,
                self.driver.effective_joint_limits_deg,
                args,
            )
            self.controller.initialize()
            self._refresh_cartesian_reference()
            print("保持目标已回读，六个舵机扭矩已开启；语音会话可用。")
            return self
        except Exception:
            self._safe_shutdown()
            raise

    def __exit__(self, *_: object) -> None:
        self._safe_shutdown()

    def _safe_shutdown(self) -> None:
        if self.driver is not None:
            try:
                self.driver.shutdown()
                print("六个舵机已卸力。")
            except Exception as exc:
                print(f"警告：普通卸力失败：{exc}", file=sys.stderr)
                if self.bus is not None:
                    try:
                        self.bus.broadcast_torque_off()
                        print("已改用广播卸力。")
                    except Exception as fallback:
                        print(f"警告：广播卸力也失败：{fallback}", file=sys.stderr)
        if self.bus is not None:
            try:
                self.bus.__exit__(None, None, None)
            finally:
                self.bus = None
        self.driver = None
        self.controller = None

    def _validate_real_pose_config(self) -> None:
        simulated = [
            name
            for name, pose in self.config["poses"].items()
            if pose.get("source") != "hardware_readback"
        ]
        if simulated:
            raise RuntimeError("真机模式拒绝模拟/手填姿态：" + "、".join(simulated))
        for name, pose in self.config["poses"].items():
            self._validate_raw_target(pose["positions"], name)

    def _validate_raw_target(self, target: Sequence[int], label: str) -> None:
        if len(target) != 6:
            raise ValueError(f"{label} 必须包含六个舵机位置")
        ranges = tuple(self.driver.safe_raw_ranges) + (
            (self.driver.states[5].minimum, self.driver.states[5].maximum),
        )
        bad = [
            f"ID{i}={value} 不在 {minimum}~{maximum}"
            for i, (value, (minimum, maximum)) in enumerate(
                zip(target, ranges, strict=True), start=1
            )
            if not minimum <= int(value) <= maximum
        ]
        if bad:
            raise RuntimeError(f"{label} 超出安全范围：" + "；".join(bad))

    def _refresh_cartesian_reference(self) -> None:
        self.controller.initialize()
        self.controller.cartesian_reference_xyz = self.controller.kinematics.position(
            self.controller.joints_deg
        )

    def _interpolate(
        self, start: Sequence[int], target: Sequence[int], requested_s: float
    ) -> tuple[list[tuple[int, ...]], float]:
        largest = max(
            abs(int(wanted) - int(current))
            for current, wanted in zip(start, target, strict=True)
        )
        actual_s = max(float(requested_s), largest / self.max_pose_speed_raw_s)
        count = max(1, math.ceil(actual_s / self.interval_s))
        points = []
        for index in range(1, count + 1):
            phase = index / count
            eased = phase * phase * (3.0 - 2.0 * phase)
            points.append(
                tuple(
                    int(round(a + (b - a) * eased))
                    for a, b in zip(start, target, strict=True)
                )
            )
        points[-1] = tuple(int(value) for value in target)
        return points, actual_s

    def _move_pose(self, name: str) -> None:
        pose = self.config["poses"][name]
        target = tuple(int(value) for value in pose["positions"])
        self._validate_raw_target(target, name)
        start = self.bus.positions()
        if abs(target[0] - start[0]) > 511:
            raise RuntimeError(
                f"ID1 路径将跨越 0/1022 接缝（{start[0]} -> {target[0]}），已拒绝"
            )
        points, actual_s = self._interpolate(
            start, target, float(pose.get("transition_s", 1.5))
        )
        print(f"开始移动到 {name}，计划 {actual_s:.2f}s，路点 {len(points)}。")
        deadline = time.monotonic()
        self.driver.set_streaming(True)
        try:
            for point in points:
                joints = tuple(
                    mapping.raw_to_degrees(raw)
                    for mapping, raw in zip(
                        self.driver.mappings, point[:5], strict=True
                    )
                )
                self.driver.send_joint_target(joints, float(point[5]))
                deadline += actual_s / len(points)
                time.sleep(max(0.0, deadline - time.monotonic()))
        finally:
            self.driver.set_streaming(False)
        targets, goals, actual = self.driver.verify_last_target()
        self._refresh_cartesian_reference()
        print(f"姿态完成：{name}")
        print(f"发送目标：{targets}")
        print(f"目标回读：{goals}")
        print(f"现场位置：{actual}")

    def _run_action(self, name: str) -> None:
        print(f"开始组合动作：{name}")
        for index, step in enumerate(self.config["actions"][name]["steps"], start=1):
            self._move_pose(step["pose"])
            hold_s = float(step.get("hold_s", 0.0))
            print(f"  {index}. {step['pose']}，停留 {hold_s:.2f}s")
            if hold_s > 0:
                time.sleep(hold_s)
        print(f"组合动作完成：{name}")

    def execute(self, command: VoiceCommand) -> None:
        if command.kind == "pose":
            self._move_pose(command.target)
        elif command.kind == "action":
            self._run_action(command.target)
        elif command.kind == "move":
            self.ik.move_hardware_distance(
                self.controller,
                self.driver,
                command.direction,
                command.distance_cm / 100.0,
                show_recovery_notice=False,
                endpoint_correction=True,
            )
            self._refresh_cartesian_reference()
        elif command.kind == "stop":
            joints, gripper = self.driver.hold_current_position()
            self.controller.joints_deg = joints
            self.controller.gripper_raw = gripper
            self._refresh_cartesian_reference()
            print("已用现场回读位置刷新保持目标。")


def diagnose_hardware(
    *,
    port: str | None,
    urdf: Path = DEFAULT_URDF,
    mapping: Path = DEFAULT_MAPPING,
) -> None:
    ik = _load_module("voice_ik_keyboard_diagnose", IK_FILE)
    args = Namespace(port=port, urdf=str(urdf), mapping=str(mapping))
    ik.command_diagnose(args)
