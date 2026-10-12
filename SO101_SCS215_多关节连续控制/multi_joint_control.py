"""SO-101 / SCS215 multi-joint preset and continuous-action controller.

Safety model:
- SCS215 only (model register 0x0523), IDs 1..6, SCSCL, 0..1023.
- Diagnosis and pose recording are read-only.
- Simulation poses are rejected in real-hardware mode.
- Before torque-on, hold the measured pose, read the targets back, then enable.
- Every interpolated target must remain inside the live EEPROM limits.
- Torque is disabled on exit or on any exception.

This controller intentionally does not claim table/self-collision avoidance.  A
separate coupled-joint collision model is required for that.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import serial
from serial.tools import list_ports


BAUD = 1_000_000
SERVO_IDS = tuple(range(1, 7))
EXPECTED_MODEL = 0x0523
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_WRITE = 0x83
ADDR_MODEL = 3
ADDR_MIN_LIMIT = 9
ADDR_MAX_LIMIT = 11
ADDR_TORQUE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
@dataclass(frozen=True)
class ServoState:
    id: int
    model: int
    position: int
    minimum: int
    maximum: int
    torque: int


def make_packet(servo_id: int, instruction: int, parameters: Iterable[int]) -> bytes:
    params = list(parameters)
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def make_sync_position_packet(positions: Sequence[int]) -> bytes:
    if len(positions) != len(SERVO_IDS):
        raise ValueError("位置必须包含 ID 1~6 共六个数值")
    params = [ADDR_GOAL_POSITION, 2]
    for servo_id, value in zip(SERVO_IDS, positions, strict=True):
        if not 0 <= int(value) <= 1023:
            raise ValueError(f"ID {servo_id} 目标 {value} 超出 0~1023")
        params.extend((servo_id, (int(value) >> 8) & 0xFF, int(value) & 0xFF))
    return make_packet(0xFE, INST_SYNC_WRITE, params)


class SCS215Bus:
    def __init__(self, port: str, timeout: float = 0.15) -> None:
        self.port_name = port
        self.serial = serial.Serial(port, BAUD, timeout=timeout)

    def __enter__(self) -> "SCS215Bus":
        time.sleep(0.12)
        return self

    def __exit__(self, *_: object) -> None:
        self.serial.close()

    def _read_status(self, expected_id: int) -> bytes:
        previous = None
        deadline = time.monotonic() + max(0.3, self.serial.timeout or 0.0)
        while time.monotonic() < deadline:
            byte = self.serial.read(1)
            if not byte:
                continue
            value = byte[0]
            if previous == 0xFF and value == 0xFF:
                break
            previous = value
        else:
            raise RuntimeError(f"ID {expected_id} 无回复")
        prefix = self.serial.read(2)
        if len(prefix) != 2:
            raise RuntimeError(f"ID {expected_id} 回复头不完整")
        packet_id, length = prefix
        remaining = self.serial.read(length)
        if len(remaining) != length:
            raise RuntimeError(f"ID {expected_id} 回复长度不足")
        payload, checksum = remaining[:-1], remaining[-1]
        expected_checksum = (~(packet_id + length + sum(payload))) & 0xFF
        if packet_id != expected_id or checksum != expected_checksum:
            raise RuntimeError(f"ID {expected_id} 回复校验失败")
        if payload[0]:
            raise RuntimeError(f"ID {expected_id} 返回错误 0x{payload[0]:02X}")
        return bytes(payload[1:])

    def exchange(self, servo_id: int, instruction: int, parameters: Iterable[int]) -> bytes:
        self.serial.reset_input_buffer()
        self.serial.write(make_packet(servo_id, instruction, parameters))
        self.serial.flush()
        return self._read_status(servo_id)

    def read(self, servo_id: int, address: int, size: int) -> bytes:
        result = self.exchange(servo_id, INST_READ, (address, size))
        if len(result) != size:
            raise RuntimeError(f"ID {servo_id} 寄存器 0x{address:02X} 长度错误")
        return result

    def read_u16(self, servo_id: int, address: int) -> int:
        return int.from_bytes(self.read(servo_id, address, 2), "big")

    def write(self, servo_id: int, address: int, data: Iterable[int]) -> None:
        self.exchange(servo_id, INST_WRITE, (address, *data))
        time.sleep(0.025)

    def write_u8(self, servo_id: int, address: int, value: int) -> None:
        self.write(servo_id, address, (value,))

    def write_u16(self, servo_id: int, address: int, value: int) -> None:
        self.write(servo_id, address, ((value >> 8) & 0xFF, value & 0xFF))

    def snapshot(self) -> list[ServoState]:
        rows = []
        for servo_id in SERVO_IDS:
            rows.append(
                ServoState(
                    id=servo_id,
                    model=self.read_u16(servo_id, ADDR_MODEL),
                    position=self.read_u16(servo_id, ADDR_PRESENT_POSITION),
                    minimum=self.read_u16(servo_id, ADDR_MIN_LIMIT),
                    maximum=self.read_u16(servo_id, ADDR_MAX_LIMIT),
                    torque=self.read(servo_id, ADDR_TORQUE, 1)[0],
                )
            )
        return rows

    def get_positions(self) -> tuple[int, ...]:
        return tuple(self.read_u16(servo_id, ADDR_PRESENT_POSITION) for servo_id in SERVO_IDS)

    def get_goal_positions(self) -> tuple[int, ...]:
        return tuple(self.read_u16(servo_id, ADDR_GOAL_POSITION) for servo_id in SERVO_IDS)

    def send_positions(self, positions: Sequence[int]) -> None:
        self.serial.reset_input_buffer()
        written = self.serial.write(make_sync_position_packet(positions))
        self.serial.flush()
        if written <= 0:
            raise RuntimeError("同步位置指令发送失败")

    def broadcast_torque_off(self) -> None:
        params = [ADDR_TORQUE, 1]
        for servo_id in SERVO_IDS:
            params.extend((servo_id, 0))
        self.serial.reset_input_buffer()
        written = self.serial.write(make_packet(0xFE, INST_SYNC_WRITE, params))
        self.serial.flush()
        time.sleep(0.08)
        if written <= 0:
            raise RuntimeError("广播卸力指令发送失败")

    def set_torque(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        completed = []
        try:
            for servo_id in SERVO_IDS:
                self.write_u8(servo_id, ADDR_TORQUE, value)
                completed.append(servo_id)
            states = self.snapshot()
            bad = [row.id for row in states if row.torque != value]
            if bad:
                raise RuntimeError(f"扭矩回读不一致：{bad}")
        except Exception:
            if enabled:
                for servo_id in completed:
                    try:
                        self.write_u8(servo_id, ADDR_TORQUE, 0)
                    except Exception:
                        pass
            else:
                try:
                    self.broadcast_torque_off()
                except Exception:
                    pass
            raise


class DryRunBus:
    def __init__(self, initial: Sequence[int] = (512, 512, 512, 512, 512, 512)) -> None:
        self.positions = tuple(int(value) for value in initial)
        self.history: list[tuple[int, ...]] = []

    def get_positions(self) -> tuple[int, ...]:
        return self.positions

    def send_positions(self, positions: Sequence[int]) -> None:
        self.positions = tuple(int(value) for value in positions)
        self.history.append(self.positions)


def find_port(requested: str | None) -> str:
    if requested:
        return requested
    matches = []
    for item in list_ports.comports():
        text = f"{item.device} {item.description} {item.hwid}".upper()
        if "CH343" in text or "VID:PID=1A86:55D3" in text:
            matches.append(item.device)
    if len(matches) != 1:
        raise RuntimeError(f"需要明确指定 CH343 串口；当前候选：{matches or '无'}")
    return matches[0]


def validate_hardware(states: Sequence[ServoState]) -> None:
    if tuple(row.id for row in states) != SERVO_IDS:
        raise RuntimeError("必须连接底部到顶部 ID 1~6 的完整机械臂")
    wrong = [row.id for row in states if row.model != EXPECTED_MODEL]
    if wrong:
        raise RuntimeError(f"以下舵机不是 SCS215 / 0x0523：{wrong}")
    unsafe = [
        f"ID{row.id}={row.position} 不在 {row.minimum}~{row.maximum}"
        for row in states
        if not row.minimum <= row.position <= row.maximum
    ]
    if unsafe:
        raise RuntimeError("当前位置超出 EEPROM 限位：" + "；".join(unsafe))


def empty_config() -> dict:
    return {"schema": 1, "servo_ids": list(SERVO_IDS), "poses": {}, "actions": {}}


def load_config(path: Path) -> dict:
    if not path.exists():
        return empty_config()
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_config(data)
    return data


def save_config(path: Path, config: dict) -> None:
    validate_config(config)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_config(config: dict) -> None:
    if config.get("schema") != 1 or config.get("servo_ids") != list(SERVO_IDS):
        raise ValueError("配置必须是 ID 1~6 的 schema 1 文件")
    pose_keys: set[str] = set()
    for name, pose in config.get("poses", {}).items():
        key = str(pose.get("key", ""))
        positions = pose.get("positions")
        if key not in "0123456789" or key in pose_keys:
            raise ValueError(f"姿态 {name} 的按键无效或重复：{key}")
        pose_keys.add(key)
        if not isinstance(positions, list) or len(positions) != 6:
            raise ValueError(f"姿态 {name} 必须包含六个位置")
        if any(not isinstance(value, int) or not 0 <= value <= 1023 for value in positions):
            raise ValueError(f"姿态 {name} 的位置必须是 0~1023 整数")
    action_keys: set[str] = set()
    for name, action in config.get("actions", {}).items():
        key = str(action.get("key", ""))
        if key not in "123456789" or key in action_keys:
            raise ValueError(f"动作 {name} 的按键无效或重复：{key}")
        action_keys.add(key)
        steps = action.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"动作 {name} 至少需要一个步骤")
        for step in steps:
            if step.get("pose") not in config.get("poses", {}):
                raise ValueError(f"动作 {name} 引用了不存在的姿态：{step.get('pose')}")
            transition = step.get("transition_s")
            hold = step.get("hold_s")
            if not isinstance(transition, (int, float)) or transition <= 0:
                raise ValueError(f"动作 {name} 的切换时间必须大于 0")
            if not isinstance(hold, (int, float)) or hold < 0:
                raise ValueError(f"动作 {name} 的停留时间不能小于 0")


def interpolate_trajectory(
    start: Sequence[int],
    target: Sequence[int],
    requested_s: float,
    max_speed_raw_s: float,
    interval_s: float,
) -> tuple[list[tuple[int, ...]], float]:
    if len(start) != 6 or len(target) != 6:
        raise ValueError("轨迹端点必须包含六个关节")
    if requested_s <= 0 or max_speed_raw_s <= 0 or interval_s <= 0:
        raise ValueError("时间和速度参数必须大于 0")
    largest = max(abs(int(b) - int(a)) for a, b in zip(start, target, strict=True))
    actual_s = max(float(requested_s), largest / float(max_speed_raw_s))
    steps = max(1, math.ceil(actual_s / interval_s))
    points = []
    for index in range(1, steps + 1):
        phase = index / steps
        eased = phase * phase * (3.0 - 2.0 * phase)
        point = tuple(
            int(round(a + (b - a) * eased))
            for a, b in zip(start, target, strict=True)
        )
        points.append(point)
    points[-1] = tuple(int(value) for value in target)
    return points, actual_s


class MotionController:
    def __init__(
        self,
        bus: SCS215Bus | DryRunBus,
        limits: Sequence[tuple[int, int]],
        *,
        max_speed_raw_s: float = 220.0,
        interval_s: float = 0.05,
        real_time: bool = True,
        max_tracking_error_raw: int = 60,
    ) -> None:
        self.bus = bus
        self.limits = tuple(limits)
        self.max_speed_raw_s = max_speed_raw_s
        self.interval_s = interval_s
        self.real_time = real_time
        self.max_tracking_error_raw = int(max_tracking_error_raw)

    def check_target(self, target: Sequence[int]) -> None:
        if len(target) != 6:
            raise ValueError("目标必须包含六个关节")
        bad = [
            f"ID{i}={value} 不在 {minimum}~{maximum}"
            for i, (value, (minimum, maximum)) in enumerate(
                zip(target, self.limits, strict=True), start=1
            )
            if not minimum <= int(value) <= maximum
        ]
        if bad:
            raise ValueError("目标超出现场限位：" + "；".join(bad))

    def move_to(
        self,
        target: Sequence[int],
        transition_s: float,
        cancel: Callable[[], bool] | None = None,
    ) -> float:
        self.check_target(target)
        start = self.bus.get_positions()
        if abs(int(target[0]) - int(start[0])) > 511:
            raise RuntimeError(
                f"ID1 路径将跨越 0/1022 接缝（{start[0]} -> {target[0]}）；"
                "SCS215 位置模式无法安全走短路径，请重新录制同一数值区间的姿态"
            )
        points, actual_s = interpolate_trajectory(
            start, target, transition_s, self.max_speed_raw_s, self.interval_s
        )
        deadline = time.monotonic()
        for index, point in enumerate(points, start=1):
            if cancel and cancel():
                raise KeyboardInterrupt("动作已停止")
            self.check_target(point)
            self.bus.send_positions(point)
            if isinstance(self.bus, SCS215Bus) and (index % 5 == 0 or index == len(points)):
                actual = self.bus.get_positions()
                errors = [abs(current - commanded) for current, commanded in zip(actual, point, strict=True)]
                if max(errors) > self.max_tracking_error_raw:
                    servo_id = errors.index(max(errors)) + 1
                    raise RuntimeError(
                        f"ID {servo_id} 跟随误差过大：目标 {point[servo_id - 1]}，"
                        f"实际 {actual[servo_id - 1]}"
                    )
                statuses = [self.bus.read(servo_id, 65, 1)[0] for servo_id in SERVO_IDS]
                bad_status = [servo_id for servo_id, status in zip(SERVO_IDS, statuses, strict=True) if status]
                if bad_status:
                    raise RuntimeError(f"舵机报警，已停止：{bad_status}，状态={statuses}")
            if self.real_time:
                deadline += actual_s / len(points)
                time.sleep(max(0.0, deadline - time.monotonic()))
        return actual_s

    def run_action(self, config: dict, name: str, cancel: Callable[[], bool] | None = None) -> None:
        action = config["actions"][name]
        print(f"开始动作：{name}")
        for index, step in enumerate(action["steps"], start=1):
            pose = config["poses"][step["pose"]]
            actual = self.move_to(pose["positions"], float(step["transition_s"]), cancel)
            print(f"  {index}. {step['pose']}，切换 {actual:.2f}s，停留 {step['hold_s']:.2f}s")
            if self.real_time and step["hold_s"]:
                end = time.monotonic() + float(step["hold_s"])
                while time.monotonic() < end:
                    if cancel and cancel():
                        raise KeyboardInterrupt("动作已停止")
                    time.sleep(min(0.05, end - time.monotonic()))
        print(f"动作完成：{name}")


def print_states(states: Sequence[ServoState]) -> None:
    print("ID  型号    位置  EEPROM限位  扭矩")
    for row in states:
        print(f"{row.id:>2}  0x{row.model:04X}  {row.position:>4}  {row.minimum:>4}~{row.maximum:<4}  {row.torque}")


def require_real_poses(config: dict) -> None:
    simulated = [name for name, pose in config["poses"].items() if pose.get("source") != "hardware_readback"]
    if simulated:
        raise RuntimeError("真机模式拒绝模拟/手填姿态：" + "、".join(simulated))


def hold_then_enable(bus: SCS215Bus, states: Sequence[ServoState]) -> None:
    positions = tuple(row.position for row in states)
    bus.send_positions(positions)
    time.sleep(0.08)
    if bus.get_goal_positions() != positions:
        raise RuntimeError("保持当前位置的目标回读不一致，拒绝开启扭矩")
    bus.set_torque(True)


def stop_key_requested() -> bool:
    if os.name != "nt":
        return False
    import msvcrt

    if not msvcrt.kbhit():
        return False
    return msvcrt.getwch() == "\x1b"


def parse_pose_sequence(value: str, available_keys: set[str]) -> list[str]:
    normalized = value.replace("，", ",").strip()
    if not normalized:
        raise ValueError("姿态序列不能为空")
    if re.search(r"[^0-9,\s]", normalized):
        raise ValueError("姿态序列只能包含数字 0~9、空格或逗号")
    keys = re.findall(r"[0-9]", normalized)
    if not keys:
        raise ValueError("姿态序列不能为空")
    missing = sorted({key for key in keys if key not in available_keys})
    if missing:
        raise ValueError("这些数字还没有保存姿态：" + "、".join(missing))
    return keys


def parse_transition_times(value: str, step_count: int) -> list[float]:
    tokens = [token for token in re.split(r"[,，\s]+", value.strip()) if token]
    if not tokens:
        raise ValueError("执行时长不能为空")
    try:
        durations = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("执行时长必须是数字") from exc
    if any(duration <= 0 for duration in durations):
        raise ValueError("执行时长必须大于 0 秒")
    if len(durations) == 1:
        return durations * step_count
    if len(durations) != step_count:
        raise ValueError(f"请输入 1 个时长，或与 {step_count} 个姿态一一对应的时长")
    return durations


def read_line_or_escape(prompt: str) -> str | None:
    import msvcrt

    print(prompt, end="", flush=True)
    characters: list[str] = []
    while True:
        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            msvcrt.getwch()
            continue
        if key == "\x1b":
            print()
            return None
        if key in {"\r", "\n"}:
            print()
            return "".join(characters).strip()
        if key == "\b":
            if characters:
                characters.pop()
                print("\b \b", end="", flush=True)
            continue
        if key.isprintable():
            characters.append(key)
            print(key, end="", flush=True)


def prompt_and_run_sequence(
    controller: MotionController,
    config: dict,
    poses_by_key: dict[str, str],
) -> bool:
    print("\nB 组合动作：按 Esc 可随时退出程序。")
    while True:
        raw_sequence = read_line_or_escape("输入姿态顺序（例如 0 1 2 3）：")
        if raw_sequence is None:
            return False
        try:
            keys = parse_pose_sequence(raw_sequence, set(poses_by_key))
            break
        except ValueError as exc:
            print(f"输入错误：{exc}")

    while True:
        raw_times = read_line_or_escape(
            "输入每段时长（单个值用于全部步骤；也可逐段输入，例如 1,2,1）："
        )
        if raw_times is None:
            return False
        try:
            durations = parse_transition_times(raw_times, len(keys))
            break
        except ValueError as exc:
            print(f"输入错误：{exc}")

    print("开始组合动作：" + " → ".join(keys))
    for index, (key, requested_s) in enumerate(zip(keys, durations, strict=True), start=1):
        name = poses_by_key[key]
        pose = config["poses"][name]
        actual_s = controller.move_to(pose["positions"], requested_s, stop_key_requested)
        print(
            f"  {index}/{len(keys)}  姿态 {key}（{name}），"
            f"请求 {requested_s:.2f}s，实际 {actual_s:.2f}s"
        )
    print("组合动作完成。")
    return True


def keyboard_loop(controller: MotionController, config: dict) -> None:
    if os.name != "nt":
        raise RuntimeError("交互键盘模式目前用于 Windows；可用 --action 直接运行动作")
    import msvcrt

    poses_by_key = {str(pose["key"]): name for name, pose in config["poses"].items()}
    print("\n请选择控制方式：")
    print("  A：单个姿态（进入后可连续输入 0~9）")
    print("  B：临时组合动作（输入姿态顺序和每段时长）")
    print("  Esc：退出并关闭舵机扭矩")
    single_mode = False
    while True:
        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            msvcrt.getwch()
            continue
        if key == "\x1b":
            return
        command = key.lower()
        if command == "a":
            single_mode = True
            print("\nA 单个姿态：直接按 0~9；按 B 输入组合动作；按 Esc 退出。")
            continue
        if command == "b":
            if not prompt_and_run_sequence(controller, config, poses_by_key):
                return
            single_mode = True
            print("可继续按 0~9，按 B 再建组合动作，按 Esc 退出。")
            continue
        if single_mode and key in poses_by_key:
            name = poses_by_key[key]
            pose = config["poses"][name]
            controller.move_to(pose["positions"], float(pose.get("transition_s", 1.5)), stop_key_requested)
            print(f"姿态 {key}：{name}。可继续输入下一个数字。")
        elif single_mode and key.isdigit():
            print(f"数字键 {key} 还没有保存姿态。")


def command_diagnose(args: argparse.Namespace) -> None:
    with SCS215Bus(find_port(args.port)) as bus:
        states = bus.snapshot()
    print_states(states)
    validate_hardware(states)
    print("只读检查通过；未写目标位置、未改变扭矩。")


def command_record(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    with SCS215Bus(find_port(args.port)) as bus:
        states = bus.snapshot()
    confirmed_wrap_ids = set(args.confirm_wrap_id or [])
    if tuple(row.id for row in states) != SERVO_IDS:
        raise RuntimeError("必须连接底部到顶部 ID 1~6 的完整机械臂")
    wrong = [row.id for row in states if row.model != EXPECTED_MODEL]
    if wrong:
        raise RuntimeError(f"以下舵机不是 SCS215 / 0x0523：{wrong}")
    outside = [row for row in states if not row.minimum <= row.position <= row.maximum]
    unconfirmed = [row.id for row in outside if row.id not in confirmed_wrap_ids]
    if unconfirmed:
        details = "；".join(
            f"ID{row.id}={row.position} 不在 {row.minimum}~{row.maximum}"
            for row in outside
            if row.id in unconfirmed
        )
        raise RuntimeError("当前位置超出 EEPROM 限位：" + details)
    if any(name != args.name and str(pose["key"]) == args.key for name, pose in config["poses"].items()):
        raise ValueError(f"姿态按键 {args.key} 已被占用")
    config["poses"][args.name] = {
        "key": args.key,
        "positions": [row.position for row in states],
        "transition_s": args.transition,
        "source": "hardware_readback",
        "wrap_confirmed_ids": sorted(confirmed_wrap_ids),
    }
    save_config(config_path, config)
    print(f"已只读保存姿态 {args.name} 到数字键 {args.key}：{[row.position for row in states]}")
    print("本操作没有开启扭矩，也没有命令机械臂运动。")


def parse_step(value: str) -> dict:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("步骤格式应为 姿态名:切换秒数:停留秒数")
    try:
        return {"pose": parts[0], "transition_s": float(parts[1]), "hold_s": float(parts[2])}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("切换和停留时间必须是数字") from exc


def command_action_set(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    if any(name != args.name and str(action["key"]) == args.key for name, action in config["actions"].items()):
        raise ValueError(f"动作按键 Shift+{args.key} 已被占用")
    config["actions"][args.name] = {"key": args.key, "steps": args.step}
    save_config(config_path, config)
    print(f"已保存动作 {args.name} 到 Shift+{args.key}，共 {len(args.step)} 步。")


def command_run(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if not config["poses"]:
        raise RuntimeError("配置中还没有姿态")
    if args.dry_run:
        limits = ((0, 1023),) * 6
        bus = DryRunBus()
        controller = MotionController(
            bus, limits, max_speed_raw_s=args.max_speed, interval_s=args.interval, real_time=False
        )
        if args.action:
            controller.run_action(config, args.action)
        else:
            keyboard_loop(controller, config)
        print(f"模拟完成，共发送 {len(bus.history)} 个插值航点。")
        return

    if not args.real:
        raise RuntimeError("真机运动必须明确增加 --real")
    require_real_poses(config)
    port = find_port(args.port)
    with SCS215Bus(port) as bus:
        states = bus.snapshot()
        validate_hardware(states)
        limits = tuple((row.minimum, row.maximum) for row in states)
        controller = MotionController(
            bus, limits, max_speed_raw_s=args.max_speed, interval_s=args.interval, real_time=True
        )
        for pose in config["poses"].values():
            controller.check_target(pose["positions"])
        try:
            hold_then_enable(bus, states)
            if args.action:
                controller.run_action(config, args.action, stop_key_requested)
            else:
                keyboard_loop(controller, config)
        finally:
            try:
                bus.set_torque(False)
                print("已关闭六个舵机扭矩。")
            except Exception as exc:
                try:
                    bus.broadcast_torque_off()
                    print(f"普通卸力应答异常，已改用广播卸力：{exc}", file=sys.stderr)
                except Exception as fallback_exc:
                    print(
                        f"警告：关闭扭矩失败：{exc}；广播卸力也失败：{fallback_exc}",
                        file=sys.stderr,
                    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO-101 / SCS215 多关节预设姿态与连续动作")
    parser.add_argument("--port", help="CH343 串口；省略时自动识别唯一设备")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().with_name("poses.json")),
        help="姿态与动作 JSON",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    diagnose = sub.add_parser("diagnose", help="只读检查六个舵机")
    diagnose.set_defaults(func=command_diagnose)

    record = sub.add_parser("record", help="只读保存当前真机姿态")
    record.add_argument("--name", required=True)
    record.add_argument("--key", required=True, choices=list("0123456789"))
    record.add_argument("--transition", type=float, default=1.5)
    record.add_argument(
        "--confirm-wrap-id",
        action="append",
        type=int,
        choices=list(SERVO_IDS),
        help="仅记录时接受已明确确认的跨 1023/0 安全舵机；可重复使用",
    )
    record.set_defaults(func=command_record)

    action = sub.add_parser("action-set", help="创建或更新连续动作")
    action.add_argument("--name", required=True)
    action.add_argument("--key", required=True, choices=list("123456789"))
    action.add_argument("--step", action="append", type=parse_step, required=True)
    action.set_defaults(func=command_action_set)

    run = sub.add_parser("run", help="键盘控制或直接运行动作")
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    run.add_argument("--action", help="直接运行指定动作；省略则进入键盘模式")
    run.add_argument("--max-speed", type=float, default=220.0, help="最大原始位置单位/秒")
    run.add_argument("--interval", type=float, default=0.05, help="插值发送间隔秒数")
    run.set_defaults(func=command_run)
    return parser


def main() -> int:
    try:
        if len(sys.argv) == 1:
            args = make_parser().parse_args(["run", "--real"])
        else:
            args = make_parser().parse_args()
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("已停止。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
