"""Safe interactive calibration helper for a six-servo SO-101 using SCS215.

The measurement workflow never commands motion.  The operator moves joints by
hand while torque is disabled.  EEPROM writes are gated behind an explicit
``apply --confirm`` command and live read-back checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import serial
from serial.tools import list_ports


BAUD = 1_000_000
SERVO_IDS = tuple(range(1, 7))
EXPECTED_MODEL = 0x0523
RAW_MIN = 0
RAW_MAX = 1023
DEFAULT_PROPOSAL = Path(__file__).resolve().with_name("records") / "limit_proposal.json"

INST_READ = 0x02
INST_WRITE = 0x03
ADDR_MODEL = 3
ADDR_MIN_LIMIT = 9
ADDR_MAX_LIMIT = 11
ADDR_TORQUE = 40
ADDR_LOCK = 48
ADDR_POSITION = 56


@dataclass(frozen=True)
class ServoState:
    id: int
    model: int
    position: int
    minimum: int
    maximum: int
    torque: int


def packet(servo_id: int, instruction: int, parameters: Iterable[int]) -> bytes:
    params = list(parameters)
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


class Bus:
    def __init__(self, port: str, timeout: float = 0.15) -> None:
        self.port_name = port
        self.serial = serial.Serial(port, BAUD, timeout=timeout)

    def close(self) -> None:
        self.serial.close()

    def __enter__(self) -> "Bus":
        time.sleep(0.12)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_status(self, expected_id: int) -> bytes:
        # Search for a complete 0xFF 0xFF header so a stale byte cannot shift
        # the packet parser.
        previous = None
        deadline = time.monotonic() + max(0.3, self.serial.timeout or 0.0)
        while time.monotonic() < deadline:
            current = self.serial.read(1)
            if not current:
                continue
            value = current[0]
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
        if packet_id != expected_id:
            raise RuntimeError(f"期望 ID {expected_id}，收到 ID {packet_id}")
        if checksum != expected_checksum:
            raise RuntimeError(f"ID {expected_id} 回复校验失败")
        error = payload[0]
        if error:
            raise RuntimeError(f"ID {expected_id} 返回错误 0x{error:02X}")
        return bytes(payload[1:])

    def exchange(self, servo_id: int, instruction: int, parameters: Iterable[int]) -> bytes:
        self.serial.reset_input_buffer()
        self.serial.write(packet(servo_id, instruction, parameters))
        self.serial.flush()
        return self._read_status(servo_id)

    def read(self, servo_id: int, address: int, size: int) -> bytes:
        data = self.exchange(servo_id, INST_READ, (address, size))
        if len(data) != size:
            raise RuntimeError(f"ID {servo_id} 寄存器 0x{address:02X} 长度错误")
        return data

    def write(self, servo_id: int, address: int, data: Iterable[int]) -> None:
        self.exchange(servo_id, INST_WRITE, (address, *data))
        time.sleep(0.04)

    def read_u16(self, servo_id: int, address: int) -> int:
        # SCSCL transmits multi-byte control-table values high byte first.
        return int.from_bytes(self.read(servo_id, address, 2), "big")

    def write_u8(self, servo_id: int, address: int, value: int) -> None:
        self.write(servo_id, address, (value,))

    def snapshot_one(self, servo_id: int) -> ServoState:
        return ServoState(
            id=servo_id,
            model=self.read_u16(servo_id, ADDR_MODEL),
            position=self.read_u16(servo_id, ADDR_POSITION),
            minimum=self.read_u16(servo_id, ADDR_MIN_LIMIT),
            maximum=self.read_u16(servo_id, ADDR_MAX_LIMIT),
            torque=self.read(servo_id, ADDR_TORQUE, 1)[0],
        )

    def snapshot(self) -> list[ServoState]:
        return [self.snapshot_one(servo_id) for servo_id in SERVO_IDS]


def find_ch343_ports() -> list[str]:
    matches = []
    for item in list_ports.comports():
        text = f"{item.device} {item.description} {item.hwid}".upper()
        if "CH343" in text or "VID:PID=1A86:55D3" in text:
            matches.append(item.device)
    return matches


def resolve_port(requested: str | None) -> str:
    if requested:
        return requested
    matches = find_ch343_ports()
    if len(matches) != 1:
        raise RuntimeError(f"需要明确指定 CH343 串口；当前候选：{matches or '无'}")
    return matches[0]


def require_models(states: list[ServoState]) -> None:
    bad = [state for state in states if state.model != EXPECTED_MODEL]
    if bad:
        details = ", ".join(f"ID{s.id}=0x{s.model:04X}" for s in bad)
        raise RuntimeError(f"舵机型号不匹配，应全部为 0x{EXPECTED_MODEL:04X}：{details}")


def require_torque_off(states: list[ServoState]) -> None:
    enabled = [state.id for state in states if state.torque != 0]
    if enabled:
        raise RuntimeError(f"以下舵机扭矩仍开启，已停止：{enabled}")


def print_states(states: list[ServoState]) -> None:
    print("ID  型号    位置  EEPROM限位  扭矩")
    for state in states:
        print(
            f"{state.id:>2}  0x{state.model:04X}  {state.position:>4}  "
            f"{state.minimum:>4}~{state.maximum:<4}  {state.torque}"
        )


def command_diagnose(args: argparse.Namespace) -> None:
    port = resolve_port(args.port)
    with Bus(port) as bus:
        states = bus.snapshot()
    print(f"只读诊断完成：{port} @ {BAUD}")
    print_states(states)
    require_models(states)


def command_torque_off(args: argparse.Namespace) -> None:
    port = resolve_port(args.port)
    with Bus(port) as bus:
        before = bus.snapshot()
        require_models(before)
        for servo_id in SERVO_IDS:
            bus.write_u8(servo_id, ADDR_TORQUE, 0)
        after = bus.snapshot()
    require_torque_off(after)
    print("六个舵机已卸力，回读确认 torque=0。")
    print_states(after)


def command_measure(args: argparse.Namespace) -> None:
    if args.seconds < 10:
        raise ValueError("测量时间至少为 10 秒")
    if not 0 <= args.margin <= 50:
        raise ValueError("安全余量必须在 0~50 原始单位之间")
    port = resolve_port(args.port)
    output = Path(args.output)
    selected_ids = tuple(
        sorted({int(value.strip()) for value in args.ids.split(",") if value.strip()})
    )
    if not selected_ids or any(servo_id not in SERVO_IDS for servo_id in selected_ids):
        raise ValueError("--ids 只能包含 1~6，例如 --ids 6 或 --ids 4,5,6")
    seed = load_proposal(Path(args.resume)) if args.resume else None
    seed_by_id = {item["id"]: item for item in seed["servos"]} if seed else {}
    with Bus(port, timeout=0.08) as bus:
        initial = bus.snapshot()
        require_models(initial)
        require_torque_off(initial)
        for state in initial:
            old = seed_by_id.get(state.id)
            if old and (state.minimum, state.maximum) != (old["old_min"], old["old_max"]):
                raise RuntimeError(f"ID {state.id} 的 EEPROM 限位已变化，不能续测")
        minima = {
            state.id: int(seed_by_id.get(state.id, {}).get("measured_min", state.position))
            for state in initial
        }
        maxima = {
            state.id: int(seed_by_id.get(state.id, {}).get("measured_max", state.position))
            for state in initial
        }
        samples = 0
        started = time.monotonic()
        next_report = started
        print(
            f"开始记录 ID {','.join(map(str, selected_ids))}。请缓慢移动到两端，"
            "不要用力顶机械止挡；结束前回到中间姿态。",
            flush=True,
        )
        while time.monotonic() - started < args.seconds:
            for servo_id in selected_ids:
                position = bus.read_u16(servo_id, ADDR_POSITION)
                minima[servo_id] = min(minima[servo_id], position)
                maxima[servo_id] = max(maxima[servo_id], position)
            samples += 1
            now = time.monotonic()
            if now >= next_report:
                elapsed = int(now - started)
                summary = " ".join(
                    f"ID{i}:{minima[i]}~{maxima[i]}" for i in selected_ids
                )
                print(f"[{elapsed:>3}s] {summary}", flush=True)
                next_report = now + 2.0
        final = bus.snapshot()

    require_torque_off(final)
    proposal = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "port_observed": port,
        "model": EXPECTED_MODEL,
        "margin": args.margin,
        "samples": samples,
        "write_performed": False,
        "servos": [],
    }
    for state in initial:
        if state.id not in selected_ids:
            continue
        measured_min = minima[state.id]
        measured_max = maxima[state.id]
        proposed_min = max(RAW_MIN, measured_min + args.margin)
        proposed_max = min(RAW_MAX, measured_max - args.margin)
        if proposed_min >= proposed_max:
            raise RuntimeError(f"ID {state.id} 测量跨度过小，无法生成限位")
        proposal["servos"].append(
            {
                "id": state.id,
                "old_min": state.minimum,
                "old_max": state.maximum,
                "measured_min": measured_min,
                "measured_max": measured_max,
                "proposed_min": proposed_min,
                "proposed_max": proposed_max,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"测量完成，提案已保存：{output.resolve()}")
    for item in proposal["servos"]:
        print(
            f"ID {item['id']}: 实测 {item['measured_min']}~{item['measured_max']} -> "
            f"提议 {item['proposed_min']}~{item['proposed_max']}"
        )


def load_proposal(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("servos", [])
    ids = [int(item.get("id", -1)) for item in items]
    if (
        data.get("model") != EXPECTED_MODEL
        or not items
        or len(ids) != len(set(ids))
        or any(servo_id not in SERVO_IDS for servo_id in ids)
    ):
        raise RuntimeError("提案不是有效的 SCS215 限位文件")
    return data


def validate_proposal(states: list[ServoState], proposal: dict) -> None:
    require_models(states)
    require_torque_off(states)
    by_id = {state.id: state for state in states}
    for item in proposal["servos"]:
        state = by_id[item["id"]]
        proposed_min = int(item["proposed_min"])
        proposed_max = int(item["proposed_max"])
        if (state.minimum, state.maximum) != (item["old_min"], item["old_max"]):
            raise RuntimeError(
                f"ID {state.id} 旧限位已变化：提案为 {item['old_min']}~{item['old_max']}，"
                f"现场为 {state.minimum}~{state.maximum}"
            )
        if not RAW_MIN <= proposed_min < proposed_max <= RAW_MAX:
            raise RuntimeError(f"ID {state.id} 提议限位无效")
        if not proposed_min <= state.position <= proposed_max:
            raise RuntimeError(
                f"ID {state.id} 当前 {state.position} 不在提议限位 "
                f"{proposed_min}~{proposed_max} 内"
            )


def command_apply(args: argparse.Namespace) -> None:
    path = Path(args.proposal)
    proposal = load_proposal(path)
    port = resolve_port(args.port)
    with Bus(port) as bus:
        before = bus.snapshot()
        validate_proposal(before, proposal)
        print("预览检查通过：型号、旧限位、当前位置和扭矩状态均符合。")
        for item in proposal["servos"]:
            state = before[item["id"] - 1]
            print(
                f"ID {state.id}: {state.minimum}~{state.maximum} -> "
                f"{item['proposed_min']}~{item['proposed_max']}，当前 {state.position}"
            )
        if not args.confirm:
            print("未写入。确认无误后，重新运行并增加 --confirm。")
            return

        for item in proposal["servos"]:
            servo_id = int(item["id"])
            minimum = int(item["proposed_min"])
            maximum = int(item["proposed_max"])
            bus.write_u8(servo_id, ADDR_LOCK, 0)
            try:
                bus.write(
                    servo_id,
                    ADDR_MIN_LIMIT,
                    (
                        (minimum >> 8) & 0xFF,
                        minimum & 0xFF,
                        (maximum >> 8) & 0xFF,
                        maximum & 0xFF,
                    ),
                )
            finally:
                bus.write_u8(servo_id, ADDR_LOCK, 1)
        for attempt in range(3):
            after = bus.snapshot()
            expected = {
                item["id"]: (item["proposed_min"], item["proposed_max"])
                for item in proposal["servos"]
            }
            bad = [
                state.id
                for state in after
                if state.id in expected
                and (state.minimum, state.maximum) != expected[state.id]
            ]
            if bad:
                raise RuntimeError(f"第 {attempt + 1} 次回读失败，异常 ID：{bad}")
        require_torque_off(after)

    proposal["write_performed"] = True
    proposal["written_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("写入完成：所选舵机连续三次回读一致，六个舵机扭矩仍为 0。")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO-101 / SCS215 安全校准与限位工具")
    parser.add_argument("--port", help="串口，例如 COM5；省略时自动选择唯一 CH343")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser("diagnose", help="只读型号、位置、限位和扭矩")
    diagnose.set_defaults(func=command_diagnose)

    torque_off = subparsers.add_parser("torque-off", help="关闭六个舵机扭矩并回读")
    torque_off.set_defaults(func=command_torque_off)

    measure = subparsers.add_parser("measure", help="扭矩关闭时记录手动运动范围")
    measure.add_argument("--seconds", type=float, default=90)
    measure.add_argument("--margin", type=int, default=5)
    measure.add_argument("--ids", default="1,2,3,4,5,6", help="本轮记录的舵机 ID")
    measure.add_argument("--resume", help="从上一轮测量 JSON 续测并合并范围")
    measure.add_argument("--output", default=str(DEFAULT_PROPOSAL))
    measure.set_defaults(func=command_measure)

    apply = subparsers.add_parser("apply", help="预览或确认写入 EEPROM 限位")
    apply.add_argument("--proposal", default=str(DEFAULT_PROPOSAL))
    apply.add_argument("--confirm", action="store_true")
    apply.set_defaults(func=command_apply)
    return parser


def main() -> int:
    try:
        args = make_parser().parse_args()
        args.func(args)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
