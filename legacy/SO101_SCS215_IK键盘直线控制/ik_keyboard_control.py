"""Keyboard Cartesian control for a six-servo SO-101 with SCS215 servos."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import serial
from serial.tools import list_ports

from ik_core import (
    ARM_JOINT_NAMES,
    CartesianController,
    DIRECTIONS,
    DIRECTION_JOINT_MOTION_WEIGHTS,
    DryRunDriver,
    MotionResult,
    URDFKinematics,
)


BAUD = 1_000_000
SERVO_IDS = tuple(range(1, 7))
EXPECTED_MODEL = 0x0523
RAW_PERIOD = 1023
RAW_MAX = 1022
SERVO_RANGE_DEG = 300.0
SPEED_UNIT_RPM = 0.059
MAX_SPEED_REGISTER = 1500
DEFAULT_SPEED_RAW_S = 90.0
CONTINUOUS_STEP_MIN_CM = 0.1
CONTINUOUS_STEP_DEFAULT_CM = 0.2
CONTINUOUS_STEP_MAX_CM = 0.4
CONTINUOUS_STEP_CHANGE_CM = 0.05
CONTINUOUS_PERIOD_S = 1.0 / 30.0
CONTINUOUS_LOAD_BIAS_RAMP_STEPS = 6
ENDPOINT_CORRECTION_THRESHOLD_M = 0.002
REFERENCE_MAX_ERROR_M = 0.020
VOICE_CORRECTION_MIN_COMMAND_M = 0.020
VOICE_CORRECTION_THRESHOLD_M = 0.003
VOICE_CORRECTION_MAX_M = 0.020
LOAD_BIAS_LEARN_GAIN = 0.5
LOAD_BIAS_MAX_RAW = 12
ID3_LOAD_BIAS_MAX_RAW = 16
LOAD_BIAS_SOURCE_MAX_RAW = 25
ID3_LOAD_DEADZONE_MIN_RAW = 8
ID3_LOAD_DEADZONE_TRIGGER_RAW = 3
DIRECTION_LOAD_BIAS_DEFAULTS = {
    # Share horizontal load correction across ID2/ID3/ID4 instead of asking
    # the elbow servo (ID3) to absorb nearly all of it.
    "forward": (4, -2, -10, -10, 0),
    "back": (-4, -3, -5, 0, 0),
    # Lateral motion is mostly ID1 rotation, but ID2-ID4 must still hold the
    # same gravity load while the base is turning.
    "left": (0, -2, -10, -10, 0),
    "right": (0, -2, -10, -10, 0),
}
FORWARD_HEIGHT_GUARD_M = 0.0008
FORWARD_ID3_BIAS_FLOOR = -10
HORIZONTAL_ID3_LOAD_BIAS_MAX_RAW = 10
HORIZONTAL_DIRECTIONS = frozenset(("forward", "back", "left", "right"))
DISTANCE_FEEDFORWARD = {
    # Re-measure only after the physical joint mapping is calibrated.  The
    # previous gains were obtained with placeholder zero positions.
    "up": 1.0,
    "down": 1.0,
}

INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_WRITE = 0x83
ADDR_MODEL = 3
ADDR_MIN_LIMIT = 9
ADDR_MAX_LIMIT = 11
ADDR_TORQUE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_TIME = 44
ADDR_GOAL_SPEED = 46
ADDR_LOCK = 48
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_VOLTAGE = 62
ADDR_STATUS = 65

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = PROJECT_DIR.parent / "SO101" / "so101_new_calib.urdf"
DEFAULT_CALIBRATION = PROJECT_DIR / "joint_mapping.json"

KEYS = {
    "w": "forward",
    "s": "back",
    "a": "left",
    "d": "right",
}

DIRECTION_ALIASES = {
    "w": "forward",
    "前": "forward",
    "前方": "forward",
    "forward": "forward",
    "s": "back",
    "后": "back",
    "后方": "back",
    "back": "back",
    "backward": "back",
    "a": "left",
    "左": "left",
    "left": "left",
    "d": "right",
    "右": "right",
    "right": "right",
    "上": "up",
    "↑": "up",
    "up": "up",
    "下": "down",
    "↓": "down",
    "down": "down",
}

LABELS = {
    "forward": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "up": "上",
    "down": "下",
}


@dataclass(frozen=True)
class ServoState:
    id: int
    model: int
    position: int
    minimum: int
    maximum: int
    torque: int
    voltage: int = 74


@dataclass(frozen=True)
class JointMapping:
    servo_id: int
    name: str
    zero_raw: float
    direction: int
    wrapped: bool = False
    guard_min: int = 0
    guard_max: int = RAW_MAX

    def raw_to_degrees(self, raw: int) -> float:
        delta = float(raw) - self.zero_raw
        if self.wrapped:
            delta = (delta + RAW_PERIOD / 2.0) % RAW_PERIOD - RAW_PERIOD / 2.0
        return self.direction * delta * SERVO_RANGE_DEG / RAW_PERIOD

    def degrees_to_raw(self, degrees: float) -> int:
        value = self.zero_raw + self.direction * float(degrees) * RAW_PERIOD / SERVO_RANGE_DEG
        if self.wrapped:
            value %= RAW_PERIOD
        return int(round(value))


def packet(servo_id: int, instruction: int, parameters: Iterable[int]) -> bytes:
    params = list(parameters)
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def raw_speed_to_register(raw_units_per_second: float) -> int:
    if raw_units_per_second <= 0:
        raise ValueError("舵机速度必须大于 0")
    degrees_per_second = raw_units_per_second * SERVO_RANGE_DEG / RAW_PERIOD
    rpm = degrees_per_second / 6.0
    return max(1, min(MAX_SPEED_REGISTER, int(round(rpm / SPEED_UNIT_RPM))))


def sync_position_packet(positions: Sequence[int]) -> bytes:
    if len(positions) != 6:
        raise ValueError("同步位置必须包含 ID1~6")
    # Match the proven preset-pose controller: only update the two-byte goal
    # position. The Cartesian waypoint schedule controls motion timing.
    params = [ADDR_GOAL_POSITION, 2]
    for servo_id, value in zip(SERVO_IDS, positions, strict=True):
        if not 0 <= int(value) <= RAW_MAX:
            raise ValueError(f"ID{servo_id} 目标 {value} 超出 0~1022")
        params.extend(
            (
                servo_id,
                (int(value) >> 8) & 0xFF,
                int(value) & 0xFF,
            )
        )
    return packet(0xFE, INST_SYNC_WRITE, params)


class SCS215Bus:
    def __init__(self, port: str, timeout: float = 0.15) -> None:
        self.serial = serial.Serial(port, BAUD, timeout=timeout)
        self.port = port

    def __enter__(self) -> "SCS215Bus":
        time.sleep(0.12)
        return self

    def __exit__(self, *_: object) -> None:
        self.serial.close()

    def _read_status(self, expected_id: int) -> bytes:
        previous = None
        deadline = time.monotonic() + max(0.3, self.serial.timeout or 0.0)
        while time.monotonic() < deadline:
            value = self.serial.read(1)
            if not value:
                continue
            current = value[0]
            if previous == 0xFF and current == 0xFF:
                break
            previous = current
        else:
            raise RuntimeError(f"ID{expected_id} 无回复")
        prefix = self.serial.read(2)
        if len(prefix) != 2:
            raise RuntimeError(f"ID{expected_id} 回复头不完整")
        packet_id, length = prefix
        remaining = self.serial.read(length)
        if len(remaining) != length:
            raise RuntimeError(f"ID{expected_id} 回复长度不足")
        payload, checksum = remaining[:-1], remaining[-1]
        expected_checksum = (~(packet_id + length + sum(payload))) & 0xFF
        if packet_id != expected_id or checksum != expected_checksum:
            raise RuntimeError(f"ID{expected_id} 回复校验失败")
        if not payload or payload[0]:
            code = payload[0] if payload else 0xFF
            raise RuntimeError(f"ID{expected_id} 返回错误 0x{code:02X}")
        return bytes(payload[1:])

    def exchange(self, servo_id: int, instruction: int, parameters: Iterable[int]) -> bytes:
        self.serial.reset_input_buffer()
        self.serial.write(packet(servo_id, instruction, parameters))
        self.serial.flush()
        return self._read_status(servo_id)

    def read(self, servo_id: int, address: int, size: int) -> bytes:
        result = self.exchange(servo_id, INST_READ, (address, size))
        if len(result) != size:
            raise RuntimeError(f"ID{servo_id} 寄存器 0x{address:02X} 长度错误")
        return result

    def read_u16(self, servo_id: int, address: int) -> int:
        return int.from_bytes(self.read(servo_id, address, 2), "big")

    def write(self, servo_id: int, address: int, data: Iterable[int]) -> None:
        self.exchange(servo_id, INST_WRITE, (address, *data))
        time.sleep(0.025)

    def write_u8(self, servo_id: int, address: int, value: int) -> None:
        self.write(servo_id, address, (value,))

    def snapshot(self) -> list[ServoState]:
        return [
            ServoState(
                id=servo_id,
                model=self.read_u16(servo_id, ADDR_MODEL),
                position=self.read_u16(servo_id, ADDR_PRESENT_POSITION),
                minimum=self.read_u16(servo_id, ADDR_MIN_LIMIT),
                maximum=self.read_u16(servo_id, ADDR_MAX_LIMIT),
                torque=self.read(servo_id, ADDR_TORQUE, 1)[0],
                voltage=self.read(servo_id, ADDR_PRESENT_VOLTAGE, 1)[0],
            )
            for servo_id in SERVO_IDS
        ]

    def positions(self) -> tuple[int, ...]:
        return tuple(self.read_u16(servo_id, ADDR_PRESENT_POSITION) for servo_id in SERVO_IDS)

    def goals(self) -> tuple[int, ...]:
        return tuple(self.read_u16(servo_id, ADDR_GOAL_POSITION) for servo_id in SERVO_IDS)

    def send_positions(self, positions: Sequence[int]) -> None:
        self.serial.reset_input_buffer()
        written = self.serial.write(sync_position_packet(positions))
        self.serial.flush()
        if written <= 0:
            raise RuntimeError("同步位置指令发送失败")

    def restore_native_motion_profile(self) -> None:
        """Restore the runtime time/speed defaults used by the pose controller."""
        for servo_id in SERVO_IDS:
            self.write(servo_id, ADDR_GOAL_TIME, (0, 0, 0, 0))
        bad = [
            servo_id
            for servo_id in SERVO_IDS
            if self.read_u16(servo_id, ADDR_GOAL_TIME) != 0
            or self.read_u16(servo_id, ADDR_GOAL_SPEED) != 0
        ]
        if bad:
            raise RuntimeError(f"运行时间/速度寄存器未恢复为 0：{bad}")

    def broadcast_torque_off(self) -> None:
        params = [ADDR_TORQUE, 1]
        for servo_id in SERVO_IDS:
            params.extend((servo_id, 0))
        self.serial.reset_input_buffer()
        self.serial.write(packet(0xFE, INST_SYNC_WRITE, params))
        self.serial.flush()
        time.sleep(0.08)

    def set_torque(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        completed: list[int] = []
        try:
            for servo_id in SERVO_IDS:
                self.write_u8(servo_id, ADDR_TORQUE, value)
                completed.append(servo_id)
            bad = [row.id for row in self.snapshot() if row.torque != value]
            if bad:
                raise RuntimeError(f"扭矩回读不一致：{bad}")
        except Exception:
            if enabled:
                try:
                    self.broadcast_torque_off()
                except Exception:
                    pass
            raise


def find_port(requested: str | None) -> str:
    if requested:
        return requested
    matches = []
    for item in list_ports.comports():
        description = f"{item.device} {item.description} {item.hwid}".upper()
        if "CH343" in description or "VID:PID=1A86:55D3" in description:
            matches.append(item.device)
    if len(matches) != 1:
        raise RuntimeError(f"需要明确指定唯一 CH343 串口；当前候选：{matches or '无'}")
    return matches[0]


def load_mappings(path: Path) -> tuple[JointMapping, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mappings = tuple(JointMapping(**item) for item in data["joints"])
    if tuple(item.servo_id for item in mappings) != (1, 2, 3, 4, 5):
        raise ValueError("joint_mapping.json 必须按 ID1~5 排列")
    if any(item.direction not in {-1, 1} for item in mappings):
        raise ValueError("关节方向只能是 -1 或 1")
    return mappings


def validate_models(states: Sequence[ServoState]) -> None:
    if tuple(row.id for row in states) != SERVO_IDS:
        raise RuntimeError("必须连接底部到顶部 ID1~6 的完整机械臂")
    bad = [row.id for row in states if row.model != EXPECTED_MODEL]
    if bad:
        raise RuntimeError(f"以下舵机不是 SCS215 / 0x0523：{bad}")


def print_states(states: Sequence[ServoState]) -> None:
    print("ID  型号    位置  EEPROM限位  扭矩  电压")
    for row in states:
        print(
            f"{row.id:>2}  0x{row.model:04X}  {row.position:>4}  "
            f"{row.minimum:>4}~{row.maximum:<4}  {row.torque}    {row.voltage / 10:.1f}V"
        )


class SCS215ArmDriver:
    """Degree-level driver with an explicit no-cross ID1 branch guard."""

    def __init__(
        self,
        bus: SCS215Bus,
        states: Sequence[ServoState],
        mappings: Sequence[JointMapping],
        urdf_limits_deg: Sequence[tuple[float, float]],
        *,
        max_tracking_error_raw: int = 60,
    ) -> None:
        self.bus = bus
        self.states = tuple(states)
        self.mappings = tuple(mappings)
        self.urdf_limits_deg = tuple(urdf_limits_deg)
        self.max_tracking_error_raw = int(max_tracking_error_raw)
        self.gripper_raw = float(states[5].position)
        self.command_count = 0
        self.streaming = False
        self.enabled = False
        self.last_targets: tuple[int, ...] | None = None
        self.last_nominal_targets: tuple[int, ...] | None = None
        self.trajectory_load_bias = (0, 0, 0, 0, 0)
        self.trajectory_total_steps = 0
        self.trajectory_step = 0
        self.direction_load_bias = {
            name: tuple(values) for name, values in DIRECTION_LOAD_BIAS_DEFAULTS.items()
        }
        self.safe_raw_ranges = self._safe_ranges()
        self.effective_joint_limits_deg = tuple(
            self._angle_range(index) for index in range(5)
        )

    def _safe_ranges(self) -> tuple[tuple[int, int], ...]:
        ranges: list[tuple[int, int]] = []
        for index, (mapping, state, urdf_limit) in enumerate(
            zip(self.mappings, self.states[:5], self.urdf_limits_deg, strict=True)
        ):
            valid = []
            for raw in range(max(0, state.minimum, mapping.guard_min), min(RAW_MAX, state.maximum, mapping.guard_max) + 1):
                angle = mapping.raw_to_degrees(raw)
                if urdf_limit[0] <= angle <= urdf_limit[1]:
                    valid.append(raw)
            if not valid:
                raise RuntimeError(f"ID{index + 1} 没有 EEPROM/映射/URDF 共同安全范围")
            ranges.append((min(valid), max(valid)))
        return tuple(ranges)

    def _angle_range(self, index: int) -> tuple[float, float]:
        minimum, maximum = self.safe_raw_ranges[index]
        values = [self.mappings[index].raw_to_degrees(raw) for raw in (minimum, maximum)]
        return min(values), max(values)

    def validate_current(self) -> None:
        problems = []
        for index, (state, safe_range) in enumerate(zip(self.states[:5], self.safe_raw_ranges, strict=True)):
            if not safe_range[0] <= state.position <= safe_range[1]:
                problems.append(
                    f"ID{index + 1} 当前 {state.position}，安全范围 {safe_range[0]}~{safe_range[1]}"
                )
        gripper = self.states[5]
        if not gripper.minimum <= gripper.position <= gripper.maximum:
            problems.append(
                f"ID6 当前 {gripper.position}，EEPROM 范围 {gripper.minimum}~{gripper.maximum}"
            )
        if problems:
            raise RuntimeError("当前位置不安全，拒绝开启扭矩：" + "；".join(problems))

    def get_joint_state(self) -> tuple[tuple[float, ...], float]:
        positions = self.bus.positions()
        joints = tuple(
            mapping.raw_to_degrees(raw)
            for mapping, raw in zip(self.mappings, positions[:5], strict=True)
        )
        self.gripper_raw = float(positions[5])
        return joints, self.gripper_raw

    def hold_then_enable(self) -> None:
        if any(row.torque != 0 for row in self.states):
            raise RuntimeError(
                "启动前检测到扭矩已经开启。请先退出其他控制程序，托住机械臂并安全卸力后重试。"
            )
        self.validate_current()
        positions = tuple(row.position for row in self.states)
        self.bus.restore_native_motion_profile()
        self.bus.send_positions(positions)
        time.sleep(0.08)
        if self.bus.goals() != positions:
            raise RuntimeError("保持当前位置的目标回读不一致，拒绝开启扭矩")
        self.bus.set_torque(True)
        self.enabled = True

    def _joint_targets_to_raw(self, joints_deg: Sequence[float]) -> tuple[int, ...]:
        if len(joints_deg) != 5:
            raise ValueError("关节目标必须包含 5 个角度")
        targets = []
        for index, (name, mapping, angle, safe_range) in enumerate(
            zip(ARM_JOINT_NAMES, self.mappings, joints_deg, self.safe_raw_ranges, strict=True)
        ):
            raw = mapping.degrees_to_raw(float(angle))
            if not safe_range[0] <= raw <= safe_range[1]:
                raise ValueError(
                    f"{name} 目标 {angle:.2f}° -> {raw}，超出 {safe_range[0]}~{safe_range[1]}"
                )
            if index == 0 and not mapping.guard_min <= raw <= mapping.guard_max:
                raise ValueError(
                    f"ID1 目标 {raw} 将离开新标定范围 "
                    f"{mapping.guard_min}~{mapping.guard_max}"
                )
            targets.append(raw)
        targets.append(int(round(self.gripper_raw)))
        return tuple(targets)

    def send_joint_target(self, joints_deg: Sequence[float], gripper_raw: float) -> None:
        if not self.enabled:
            raise RuntimeError("真机扭矩尚未通过安全流程开启")
        self.gripper_raw = float(gripper_raw)
        nominal = self._joint_targets_to_raw(joints_deg)
        targets = self._apply_trajectory_load_bias(nominal)
        self.bus.send_positions(targets)
        self.last_nominal_targets = nominal
        self.last_targets = targets
        self.command_count += 1
        if not self.streaming and self.command_count % 5 == 0:
            actual = self.bus.positions()
            errors = [abs(a - b) for a, b in zip(actual, targets, strict=True)]
            worst = max(errors)
            if worst > self.max_tracking_error_raw:
                servo_id = errors.index(worst) + 1
                raise RuntimeError(
                    f"ID{servo_id} 跟随误差过大：目标 {targets[servo_id - 1]}，实际 {actual[servo_id - 1]}"
                )
            statuses = [self.bus.read(servo_id, ADDR_STATUS, 1)[0] for servo_id in SERVO_IDS]
            bad = [servo_id for servo_id, status in zip(SERVO_IDS, statuses, strict=True) if status]
            if bad:
                raise RuntimeError(f"舵机报警，已停止：{bad}，状态={statuses}")

    def _apply_trajectory_load_bias(
        self, nominal: Sequence[int]
    ) -> tuple[int, ...]:
        if self.trajectory_total_steps <= 0:
            return tuple(int(value) for value in nominal)
        self.trajectory_step += 1
        progress = min(1.0, self.trajectory_step / self.trajectory_total_steps)
        blend = progress * progress * (3.0 - 2.0 * progress)
        targets = []
        for wanted, bias, safe_range in zip(
            nominal[:5], self.trajectory_load_bias, self.safe_raw_ranges, strict=True
        ):
            shifted = int(round(int(wanted) + int(bias) * blend))
            targets.append(max(safe_range[0], min(safe_range[1], shifted)))
        targets.append(int(nominal[5]))
        return tuple(targets)

    def begin_trajectory_load_bias(
        self, direction: str, total_steps: int
    ) -> tuple[int, ...]:
        bias = self.direction_load_bias.get(direction, (0, 0, 0, 0, 0))
        self.trajectory_load_bias = tuple(int(value) for value in bias)
        self.trajectory_total_steps = max(1, int(total_steps))
        self.trajectory_step = 0
        return self.trajectory_load_bias

    def end_trajectory_load_bias(self) -> None:
        self.trajectory_load_bias = (0, 0, 0, 0, 0)
        self.trajectory_total_steps = 0
        self.trajectory_step = 0

    def learn_direction_load_bias(
        self,
        direction: str,
        nominal: Sequence[int],
        actual: Sequence[int],
    ) -> tuple[int, ...]:
        current = self.direction_load_bias.get(direction, (0, 0, 0, 0, 0))
        updated = []
        for index, (bias, wanted, measured) in enumerate(
            zip(current, nominal[:5], actual[:5], strict=True)
        ):
            residual = int(wanted) - int(measured)
            value = int(bias) + int(round(residual * LOAD_BIAS_LEARN_GAIN))
            if (
                index == 2
                and abs(residual) >= ID3_LOAD_DEADZONE_TRIGGER_RAW
                and abs(value) < ID3_LOAD_DEADZONE_MIN_RAW
                and (int(bias) == 0 or int(bias) * residual > 0)
            ):
                value = int(
                    math.copysign(
                        max(ID3_LOAD_DEADZONE_MIN_RAW, abs(value)), residual
                    )
                )
            if index == 2 and direction in HORIZONTAL_DIRECTIONS:
                limit = HORIZONTAL_ID3_LOAD_BIAS_MAX_RAW
            else:
                limit = ID3_LOAD_BIAS_MAX_RAW if index == 2 else LOAD_BIAS_MAX_RAW
            value = max(-limit, min(limit, value))
            updated.append(value)
        result = tuple(updated)
        self.direction_load_bias[direction] = result
        return result

    def set_streaming(self, enabled: bool) -> None:
        self.streaming = bool(enabled)

    def verify_last_target(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if self.last_targets is None:
            raise RuntimeError("尚未发送关节目标")
        goals = self.bus.goals()
        if goals != self.last_targets:
            raise RuntimeError(
                f"最终目标寄存器回读不一致：发送 {self.last_targets}，回读 {goals}"
            )
        deadline = time.monotonic() + 4.0
        while True:
            actual = self.bus.positions()
            errors = [abs(a - b) for a, b in zip(actual, self.last_targets, strict=True)]
            worst = max(errors)
            if worst <= 2:
                break
            if time.monotonic() >= deadline:
                if worst > self.max_tracking_error_raw:
                    servo_id = errors.index(worst) + 1
                    raise RuntimeError(
                        f"最终等待超时：ID{servo_id} 跟随误差过大，目标 "
                        f"{self.last_targets[servo_id - 1]}，实际 {actual[servo_id - 1]}"
                    )
                break
            time.sleep(0.1)
        statuses = [self.bus.read(servo_id, ADDR_STATUS, 1)[0] for servo_id in SERVO_IDS]
        bad = [
            servo_id
            for servo_id, status in zip(SERVO_IDS, statuses, strict=True)
            if status
        ]
        if bad:
            raise RuntimeError(f"舵机报警，已停止：{bad}，状态={statuses}")
        return self.last_targets, goals, actual

    def hold_current_position(self) -> tuple[tuple[float, ...], float]:
        """Stop a live jog by making the measured pose the new hold target."""
        actual = self.bus.positions()
        for index, (raw, safe_range) in enumerate(
            zip(actual[:5], self.safe_raw_ranges, strict=True)
        ):
            if not safe_range[0] <= raw <= safe_range[1]:
                raise RuntimeError(
                    f"自由控制停止时 ID{index + 1}={raw} 超出安全范围 "
                    f"{safe_range[0]}~{safe_range[1]}"
                )
        gripper = self.states[5]
        if not gripper.minimum <= actual[5] <= gripper.maximum:
            raise RuntimeError("自由控制停止时夹爪位置超出 EEPROM 限位")
        self.bus.send_positions(actual)
        self.last_targets = actual
        joints = tuple(
            mapping.raw_to_degrees(raw)
            for mapping, raw in zip(self.mappings, actual[:5], strict=True)
        )
        self.gripper_raw = float(actual[5])
        return joints, self.gripper_raw

    def shutdown(self) -> None:
        if not self.enabled:
            return
        try:
            self.bus.set_torque(False)
        except Exception:
            self.bus.broadcast_torque_off()
        finally:
            self.enabled = False


def dry_limits(
    mappings: Sequence[JointMapping], urdf_limits: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float], ...]:
    limits = list(urdf_limits)
    id1_values = [
        mappings[0].raw_to_degrees(raw)
        for raw in (mappings[0].guard_min, mappings[0].guard_max)
    ]
    limits[0] = (
        max(limits[0][0], min(id1_values)),
        min(limits[0][1], max(id1_values)),
    )
    return tuple(limits)


def show_motion(result: MotionResult) -> None:
    status = "IK 规划完成" if result.full_success else "IK 规划部分完成"
    print(
        f"{status}：向{LABELS[result.direction]}请求 {result.requested_m * 100:.2f} cm，"
        f"沿原直线执行 {result.executed_m * 100:.2f} cm，路点 {result.waypoint_count}。"
    )
    print(result.message)
    print(
        "末端位置："
        f"x={result.final_xyz[0]:.4f} m, y={result.final_xyz[1]:.4f} m, z={result.final_xyz[2]:.4f} m"
    )


def actual_directional_distance(
    result: MotionResult, actual_xyz: Sequence[float]
) -> float:
    direction = DIRECTIONS[result.direction]
    return max(
        0.0,
        sum(
            (float(actual) - start) * axis
            for actual, start, axis in zip(
                actual_xyz, result.start_xyz, direction, strict=True
            )
        ),
    )


def synchronize_and_report_hardware(
    controller: CartesianController,
    driver: SCS215ArmDriver,
    result: MotionResult,
) -> tuple[tuple[float, float, float], float, tuple[int, ...]]:
    _, _, actual = driver.verify_last_target()
    actual_joints, _ = driver.get_joint_state()
    controller.joints_deg = actual_joints
    actual_xyz = controller.kinematics.position(actual_joints)
    actual_distance = actual_directional_distance(result, actual_xyz)
    return actual_xyz, actual_distance, actual


def guard_forward_height_bias(
    driver: SCS215ArmDriver,
    direction: str,
    desired_z: float,
    actual_z: float,
) -> bool:
    """Avoid learning a stronger lift bias when a forward move already ends high."""
    if direction != "forward" or actual_z <= desired_z + FORWARD_HEIGHT_GUARD_M:
        return False
    current = list(driver.direction_load_bias.get(direction, (0, 0, 0, 0, 0)))
    current[2] = max(current[2], FORWARD_ID3_BIAS_FLOOR)
    driver.direction_load_bias[direction] = tuple(current)
    return True


def move_hardware_distance(
    controller: CartesianController,
    driver: SCS215ArmDriver,
    direction: str,
    distance_m: float,
    *,
    show_recovery_notice: bool = True,
    endpoint_correction: bool = False,
) -> tuple[float, float, float]:
    origin = controller.kinematics.position(controller.joints_deg)
    axis = DIRECTIONS[direction]
    command_origin = controller.cartesian_reference_xyz or origin
    desired_xyz = tuple(
        start + component * distance_m
        for start, component in zip(command_origin, axis, strict=True)
    )
    distance_tolerance = max(0.0002, min(0.001, distance_m * 0.1))
    feedforward = DISTANCE_FEEDFORWARD.get(direction, 1.0)
    planned_distance = distance_m * feedforward
    if feedforward != 1.0:
        print(
            f"现场标定前馈：请求 {distance_m * 100:.2f} cm，"
            f"连续轨迹规划 {planned_distance * 100:.2f} cm（{feedforward:.2f}×）。"
        )
    path_distance = math.dist(origin, desired_xyz)
    expected_waypoints = max(
        1, math.ceil(path_distance / controller.cartesian_resolution_m)
    )
    driver.begin_trajectory_load_bias(direction, expected_waypoints)
    try:
        absolute_result = controller.move_to_xyz(
            desired_xyz,
            stream_targets=True,
            joint_motion_weights=DIRECTION_JOINT_MOTION_WEIGHTS.get(direction),
        )
    finally:
        driver.end_trajectory_load_bias()
    planned_along_axis = max(
        0.0,
        sum(
            (final - start) * component
            for final, start, component in zip(
                absolute_result.final_xyz, absolute_result.start_xyz, axis, strict=True
            )
        ),
    )
    result = MotionResult(
        direction=direction,
        requested_m=distance_m,
        executed_m=min(distance_m, planned_along_axis),
        full_success=absolute_result.full_success,
        waypoint_count=absolute_result.waypoint_count,
        start_xyz=absolute_result.start_xyz,
        final_xyz=absolute_result.final_xyz,
        message=(
            "已沿单段直线同时执行指定方向移动与锁定平面恢复"
            if absolute_result.full_success
            else absolute_result.message
        ),
    )
    desired_raw = driver.last_nominal_targets
    actual_xyz, _, actual_raw = synchronize_and_report_hardware(controller, driver, result)
    endpoint_error = math.sqrt(
        sum((target - actual) ** 2 for target, actual in zip(desired_xyz, actual_xyz, strict=True))
    )
    if (
        endpoint_correction
        and result.full_success
        and distance_m >= VOICE_CORRECTION_MIN_COMMAND_M
        and VOICE_CORRECTION_THRESHOLD_M < endpoint_error <= VOICE_CORRECTION_MAX_M
    ):
        print(f"检测到末端偏差 {endpoint_error * 100:.2f} cm，正在平滑修正。")
        controller.cartesian_reference_xyz = actual_xyz
        correction_waypoints = max(
            1, math.ceil(endpoint_error / controller.cartesian_resolution_m)
        )
        driver.begin_trajectory_load_bias(direction, correction_waypoints)
        try:
            correction_result = controller.move_to_xyz(
                desired_xyz, stream_targets=True
            )
        finally:
            driver.end_trajectory_load_bias()
        planned_along_axis = max(
            0.0,
            sum(
                (final - start) * component
                for final, start, component in zip(
                    correction_result.final_xyz, origin, axis, strict=True
                )
            ),
        )
        result = MotionResult(
            direction=direction,
            requested_m=distance_m,
            executed_m=min(distance_m, planned_along_axis),
            full_success=correction_result.full_success,
            waypoint_count=result.waypoint_count + correction_result.waypoint_count,
            start_xyz=result.start_xyz,
            final_xyz=correction_result.final_xyz,
            message=correction_result.message,
        )
        desired_raw = driver.last_nominal_targets
        actual_xyz, _, actual_raw = synchronize_and_report_hardware(
            controller, driver, result
        )
        endpoint_error = math.sqrt(
            sum(
                (target - actual) ** 2
                for target, actual in zip(desired_xyz, actual_xyz, strict=True)
            )
        )
    raw_errors = (
        tuple(
            int(wanted) - int(measured)
            for wanted, measured in zip(desired_raw[:5], actual_raw[:5], strict=True)
        )
        if desired_raw is not None
        else ()
    )
    worst_raw_error = max((abs(value) for value in raw_errors), default=0)
    height_guarded = guard_forward_height_bias(
        driver, direction, desired_xyz[2], actual_xyz[2]
    )
    if (
        not height_guarded
        and result.full_success
        and endpoint_error > ENDPOINT_CORRECTION_THRESHOLD_M
        and (
        2 < worst_raw_error <= LOAD_BIAS_SOURCE_MAX_RAW
        )
    ):
        driver.learn_direction_load_bias(
            direction, desired_raw, actual_raw
        )
    if result.full_success and endpoint_error <= REFERENCE_MAX_ERROR_M:
        controller.cartesian_reference_xyz = desired_xyz
    else:
        controller.cartesian_reference_xyz = actual_xyz
    total = max(
        0.0,
        sum(
            (actual - start) * component
            for actual, start, component in zip(actual_xyz, origin, axis, strict=True)
        ),
    )
    distance_ok = abs(total - distance_m) <= distance_tolerance
    status = "完成" if result.full_success and distance_ok else "未达标"
    height_error_mm = (actual_xyz[2] - desired_xyz[2]) * 1000.0
    print(
        f"{status}：{LABELS[direction]} 请求 {distance_m * 100:.2f} cm，"
        f"实际 {total * 100:.2f} cm；终点误差 {endpoint_error * 100:.2f} cm，"
        f"高度偏差 {height_error_mm:+.1f} mm。"
    )
    if not result.full_success:
        print(f"原因：{result.message}")
    elif not distance_ok and show_recovery_notice:
        print(
            "已安全停止且不做终点突跳补偿；下一次将从现场回读姿态继续。"
        )
    return tuple(float(value) for value in controller.cartesian_reference_xyz)


def decode_control_key(first: str, second: str | None = None) -> str:
    """Convert Windows console key sequences into controller actions."""
    if first in {"\x00", "\xe0"}:
        return {"H": "up", "P": "down"}.get(second or "", "")
    key = first.lower()
    return KEYS.get(key, key)


WINDOWS_DIRECTION_KEYS = (
    (0x57, "forward"),  # W
    (0x53, "back"),     # S
    (0x41, "left"),     # A
    (0x44, "right"),    # D
    (0x26, "up"),       # Up arrow
    (0x28, "down"),     # Down arrow
)


def pressed_direction(is_down) -> str | None:
    pressed = [direction for key, direction in WINDOWS_DIRECTION_KEYS if is_down(key)]
    if len(pressed) != 1:
        return None
    return pressed[0]


def print_live_pose(controller: CartesianController) -> None:
    if isinstance(controller.driver, SCS215ArmDriver):
        controller.joints_deg, controller.gripper_raw = (
            controller.driver.get_joint_state()
        )
    xyz = controller.kinematics.position(controller.joints_deg)
    print(f"末端：x={xyz[0]:.4f}, y={xyz[1]:.4f}, z={xyz[2]:.4f} m")


def parse_direction(value: str) -> str | None:
    return DIRECTION_ALIASES.get(value.strip().lower())


def choose_start_mode(getch=None) -> str | None:
    if getch is None:
        if os.name != "nt":
            raise RuntimeError("即时模式选择目前用于 Windows")
        import msvcrt

        getch = msvcrt.getwch
    print("\n请选择控制模式（直接按键，无需回车）：")
    print("  A：输入方向和距离（厘米）")
    print("  B：自由键盘控制（WASD + 方向键上下）")
    print("  Esc：退出")
    while True:
        key = getch()
        if key in {"\x00", "\xe0"}:
            getch()
            continue
        if key == "\x1b":
            return None
        choice = key.lower()
        if choice == "a":
            print("A")
            return "distance"
        if choice == "b":
            print("B")
            return "free"


def run_distance_mode(controller: CartesianController, input_fn=input) -> None:
    print("\nA 定距模式：方向可输入 前/后/左/右/上/下。")
    print("输入 P 查看现场末端位置，输入 Q 返回 A/B 选择。")
    while True:
        raw_direction = input_fn("\n移动方向：").strip()
        if raw_direction.lower() in {"q", "quit", "exit", "b"}:
            print("返回 A/B 模式选择。")
            return
        if raw_direction.lower() in {"p", "position", "status", "状态"}:
            print_live_pose(controller)
            continue
        direction = parse_direction(raw_direction)
        if direction is None:
            print("方向无效，请输入 前、后、左、右、上或下。")
            continue

        raw_distance = input_fn("移动距离（cm）：").strip()
        if raw_distance.lower() in {"q", "quit", "exit", "b"}:
            print("返回 A/B 模式选择。")
            return
        try:
            distance_cm = float(raw_distance)
        except ValueError:
            print("距离无效，请输入正数，例如 5。")
            continue
        if not math.isfinite(distance_cm) or distance_cm <= 0:
            print("距离必须是大于 0 的有限数值。")
            continue

        if isinstance(controller.driver, SCS215ArmDriver):
            move_hardware_distance(
                controller,
                controller.driver,
                direction,
                distance_cm / 100.0,
            )
        else:
            show_motion(controller.move(direction, distance_cm / 100.0))
        return


def run_interactive(controller: CartesianController, step_cm: float) -> str:
    if os.name != "nt":
        raise RuntimeError("键盘即时模式目前用于 Windows")
    import ctypes
    import msvcrt

    step = max(
        CONTINUOUS_STEP_MIN_CM,
        min(CONTINUOUS_STEP_MAX_CM, float(step_cm)),
    )
    is_down = lambda key: bool(ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000)
    print("\n按住 W/S 前后，按住 A/D 左右，按住方向键 ↑/↓ 上下；松手立即停止。")
    print("+/- 调整速度，P 打印位置，Q 返回模式选择，Esc 退出。")
    print(
        f"当前速度约 {step / CONTINUOUS_PERIOD_S:.1f} cm/s "
        f"（每 {CONTINUOUS_PERIOD_S:.2f} s 前进 {step:.2f} cm）。"
    )
    active_direction: str | None = None
    blocked_direction: str | None = None
    previous_plus = previous_minus = previous_print = False
    next_move_at = 0.0
    while True:
        while msvcrt.kbhit():
            msvcrt.getwch()
        escape_pressed = is_down(0x1B)
        menu_pressed = is_down(0x51)
        if escape_pressed or menu_pressed:
            if active_direction and isinstance(controller.driver, SCS215ArmDriver):
                controller.driver.end_trajectory_load_bias()
                controller.driver.set_streaming(False)
                controller.joints_deg, controller.gripper_raw = (
                    controller.driver.hold_current_position()
                )
            if escape_pressed:
                print("退出键盘控制。")
                return "exit"
            print("返回 A/B 模式选择。")
            return "menu"

        plus = is_down(0xBB) or is_down(0x6B)
        minus = is_down(0xBD) or is_down(0x6D)
        print_key = is_down(0x50)
        if plus and not previous_plus:
            step = min(CONTINUOUS_STEP_MAX_CM, step + CONTINUOUS_STEP_CHANGE_CM)
            print(f"连续控制速度：约 {step / CONTINUOUS_PERIOD_S:.1f} cm/s")
        if minus and not previous_minus:
            step = max(CONTINUOUS_STEP_MIN_CM, step - CONTINUOUS_STEP_CHANGE_CM)
            print(f"连续控制速度：约 {step / CONTINUOUS_PERIOD_S:.1f} cm/s")
        if print_key and not previous_print:
            print_live_pose(controller)
        previous_plus, previous_minus, previous_print = plus, minus, print_key

        direction = pressed_direction(is_down)
        if direction is None:
            blocked_direction = None
            if active_direction is not None:
                if isinstance(controller.driver, SCS215ArmDriver):
                    controller.driver.end_trajectory_load_bias()
                    controller.driver.set_streaming(False)
                    controller.joints_deg, controller.gripper_raw = (
                        controller.driver.hold_current_position()
                    )
                print(f"已停止向{LABELS[active_direction]}运动。")
                active_direction = None
            time.sleep(0.015)
            continue

        if direction == blocked_direction:
            time.sleep(0.015)
            continue
        if active_direction is not None and direction != active_direction:
            if isinstance(controller.driver, SCS215ArmDriver):
                controller.driver.end_trajectory_load_bias()
                controller.driver.set_streaming(False)
                controller.joints_deg, controller.gripper_raw = (
                    controller.driver.hold_current_position()
                )
            active_direction = None

        now = time.monotonic()
        if now < next_move_at:
            time.sleep(min(0.01, next_move_at - now))
            continue

        if active_direction is None and isinstance(controller.driver, SCS215ArmDriver):
            controller.driver.begin_trajectory_load_bias(
                direction, CONTINUOUS_LOAD_BIAS_RAMP_STEPS
            )
            controller.driver.set_streaming(True)
        move_started_at = time.monotonic()
        saved_interval = controller.command_interval_s
        controller.command_interval_s = 0.0
        try:
            result = controller.move(direction, step / 100.0, final_target_only=True)
        finally:
            controller.command_interval_s = saved_interval
        if not result.full_success:
            show_motion(result)
            blocked_direction = direction
            if isinstance(controller.driver, SCS215ArmDriver):
                controller.driver.end_trajectory_load_bias()
                controller.driver.set_streaming(False)
                controller.joints_deg, controller.gripper_raw = (
                    controller.driver.hold_current_position()
                )
            active_direction = None
            print("已到可达边界；松开该按键后可选择其他方向。")
            continue
        if active_direction is None:
            print(f"开始向{LABELS[direction]}连续运动。")
        active_direction = direction
        next_move_at = move_started_at + CONTINUOUS_PERIOD_S


def make_controller(
    kinematics: URDFKinematics,
    driver: DryRunDriver | SCS215ArmDriver,
    limits: Sequence[tuple[float, float]],
    args: argparse.Namespace,
) -> CartesianController:
    return CartesianController(
        kinematics,
        driver,
        joint_limits_deg=limits,
        cartesian_resolution_m=args.resolution_mm / 1000.0,
        tolerance_m=args.tolerance_mm / 1000.0,
        max_joint_step_deg=args.max_joint_step,
        command_interval_s=args.interval,
    )


def execute_dry(args: argparse.Namespace) -> None:
    kinematics = URDFKinematics(args.urdf)
    mappings = load_mappings(Path(args.mapping))
    limits = dry_limits(mappings, kinematics.joint_limits_deg)
    joints = tuple(float(value) for value in args.initial_joints.split(","))
    if len(joints) != 5:
        raise ValueError("--initial-joints 需要 5 个逗号分隔角度")
    driver = DryRunDriver(joints)
    controller = make_controller(kinematics, driver, limits, args)
    controller.initialize()
    if args.direction:
        show_motion(controller.move(args.direction, args.distance_cm / 100.0))
    else:
        run_interactive(controller, args.step_cm)


def run_control_session(
    controller: CartesianController,
    initial_mode: str,
    step_cm: float,
) -> None:
    """Keep one checked hardware session alive while switching A/B modes."""
    controller.cartesian_reference_xyz = controller.kinematics.position(
        controller.joints_deg
    )
    mode: str | None = initial_mode
    while mode is not None:
        if mode == "distance":
            run_distance_mode(controller)
        else:
            if run_interactive(controller, step_cm) == "exit":
                return
            controller.cartesian_reference_xyz = controller.kinematics.position(
                controller.joints_deg
            )
        mode = choose_start_mode()


def execute_real(args: argparse.Namespace) -> None:
    if not args.real:
        raise RuntimeError("真机运动必须明确增加 --real")
    port = find_port(args.port)
    kinematics = URDFKinematics(args.urdf)
    mappings = load_mappings(Path(args.mapping))
    with SCS215Bus(port) as bus:
        states = bus.snapshot()
        print_states(states)
        validate_models(states)
        driver = SCS215ArmDriver(bus, states, mappings, kinematics.joint_limits_deg)
        print("软件安全原始范围：" + "，".join(
            f"ID{i + 1} {minimum}~{maximum}"
            for i, (minimum, maximum) in enumerate(driver.safe_raw_ranges)
        ))
        driver.hold_then_enable()
        print("当前位置保持目标已回读，六个舵机扭矩已开启。")
        controller = make_controller(
            kinematics, driver, driver.effective_joint_limits_deg, args
        )
        try:
            controller.initialize()
            if getattr(args, "control_mode", None) == "distance":
                run_control_session(controller, "distance", args.step_cm)
            elif getattr(args, "control_mode", None) == "free":
                run_control_session(controller, "free", args.step_cm)
            elif args.direction:
                move_hardware_distance(
                    controller,
                    driver,
                    args.direction,
                    args.distance_cm / 100.0,
                )
            else:
                run_control_session(controller, "free", args.step_cm)
        except Exception:
            print("检测到异常，正在停止指令并安全卸力。")
            raise
        finally:
            driver.shutdown()
            print("六个舵机已卸力。")


def execute_menu(args: argparse.Namespace, input_fn=input, getch=None) -> None:
    mode = choose_start_mode(getch)
    if mode is None:
        print("已退出，机械臂未连接、未开启扭矩。")
        return
    confirmation = input_fn(
        "请托住机械臂并确认周围无障碍物；输入 Y 后开始只读检查并启动："
    ).strip().lower()
    if confirmation not in {"y", "yes"}:
        print("已取消，机械臂未连接、未开启扭矩。")
        return
    args.real = True
    args.direction = None
    args.distance_cm = None
    args.step_cm = CONTINUOUS_STEP_DEFAULT_CM
    args.control_mode = mode
    execute_real(args)


def command_diagnose(args: argparse.Namespace) -> None:
    port = find_port(args.port)
    kinematics = URDFKinematics(args.urdf)
    mappings = load_mappings(Path(args.mapping))
    with SCS215Bus(port) as bus:
        states = bus.snapshot()
    validate_models(states)
    print(f"只读诊断完成：{port} @ {BAUD}")
    print_states(states)
    driver = SCS215ArmDriver.__new__(SCS215ArmDriver)
    # Build a range report without opening a second serial connection.
    driver.bus = None
    driver.states = tuple(states)
    driver.mappings = tuple(mappings)
    driver.urdf_limits_deg = tuple(kinematics.joint_limits_deg)
    ranges = SCS215ArmDriver._safe_ranges(driver)
    print("软件安全原始范围：" + "，".join(
        f"ID{i + 1} {minimum}~{maximum}" for i, (minimum, maximum) in enumerate(ranges)
    ))
    unsafe = [
        f"ID{i + 1}={state.position}，应在 {minimum}~{maximum}"
        for i, (state, (minimum, maximum)) in enumerate(
            zip(states[:5], ranges, strict=True)
        )
        if not minimum <= state.position <= maximum
    ]
    gripper = states[5]
    if not gripper.minimum <= gripper.position <= gripper.maximum:
        unsafe.append(
            f"ID6={gripper.position}，应在 {gripper.minimum}~{gripper.maximum}"
        )
    if unsafe:
        print("禁止真机启动：" + "；".join(unsafe))
        print("请保持扭矩关闭，托住机械臂并手动移回安全范围后重新诊断。")
    else:
        print("当前位置通过启动范围检查。")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SO-101 / SCS215 IK 键盘直线控制")
    result.add_argument("--port", help="CH343 串口；省略时自动识别唯一设备")
    result.add_argument("--urdf", default=str(DEFAULT_URDF))
    result.add_argument("--mapping", default=str(DEFAULT_CALIBRATION))
    result.add_argument("--resolution-mm", type=float, default=2.0, help="直线路点间距")
    result.add_argument("--tolerance-mm", type=float, default=0.8, help="IK 位置容差")
    result.add_argument("--max-joint-step", type=float, default=12.0)
    result.add_argument(
        "--interval",
        type=float,
        default=CONTINUOUS_PERIOD_S,
        help="路点发送间隔秒",
    )

    result.set_defaults(
        func=execute_menu,
        real=False,
        direction=None,
        distance_cm=None,
        step_cm=CONTINUOUS_STEP_DEFAULT_CM,
        control_mode=None,
    )
    sub = result.add_subparsers(dest="command")
    start = sub.add_parser("start", help="显示真机双模式启动菜单")
    start.set_defaults(func=execute_menu)

    diagnose = sub.add_parser("diagnose", help="只读检查，不移动、不改扭矩")
    diagnose.set_defaults(func=command_diagnose)

    dry = sub.add_parser("dry-run", help="模拟 IK 和直线轨迹")
    dry.add_argument("--direction", choices=tuple(DIRECTIONS))
    dry.add_argument("--distance-cm", type=float, default=5.0)
    dry.add_argument("--step-cm", type=float, default=1.0)
    dry.add_argument("--initial-joints", default="-45,45,-65,25,0")
    dry.set_defaults(func=execute_dry)

    move = sub.add_parser("move", help="真机单次直线移动")
    move.add_argument("--direction", required=True, choices=tuple(DIRECTIONS))
    move.add_argument("--distance-cm", required=True, type=float)
    move.add_argument("--real", action="store_true")
    move.set_defaults(func=execute_real)

    run = sub.add_parser("run", help="真机键盘控制")
    run.add_argument("--step-cm", type=float, default=CONTINUOUS_STEP_DEFAULT_CM)
    run.add_argument("--real", action="store_true")
    run.set_defaults(func=execute_real, direction=None)
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        if args.resolution_mm <= 0 or args.tolerance_mm <= 0 or args.interval < 0:
            raise ValueError("分辨率、容差必须大于 0，发送间隔不能小于 0")
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\n已由用户中止。")
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
