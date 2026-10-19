"""Guarded eye-in-hand checkerboard data collection.

The script reuses the established multi-joint controller for every hardware
write.  It only explores a small envelope around the measured start pose,
records actual encoder positions at exposure time, and always disables torque
on exit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MULTI_DIR = ROOT / "SO101_SCS215_多关节连续控制"
IK_DIR = ROOT / "SO101_SCS215_IK键盘直线控制"
for module_dir in (MULTI_DIR, IK_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from ik_keyboard_control import load_mappings  # noqa: E402
from multi_joint_control import (  # noqa: E402
    MotionController,
    SCS215Bus,
    find_port,
    hold_then_enable,
    validate_hardware,
)
from aruco_visual_servo import (  # noqa: E402
    _write_image,
    find_chessboard,
    rotate_frame,
)


DEFAULT_MAPPING = IK_DIR / "joint_mapping.json"
DEFAULT_OUTPUT_DIR = HERE / "data" / "hand_eye"
DEFAULT_MANIFEST = HERE / "data" / "hand_eye_manifest.json"
MAX_START_DELTA_RAW = 35
LIMIT_MARGIN_RAW = 6

# Excite yaw, pitch, elbow/wrist pitch and wrist roll independently, then in
# combinations.  Values are raw-position offsets from an interior center pose.
OFFSETS = (
    (0, 0, 0, 0, 0),
    (12, 0, 0, 0, 0),
    (-12, 0, 0, 0, 0),
    (0, 12, 0, 0, 0),
    (0, 24, 0, 0, 0),
    (0, 0, 12, 0, 0),
    (0, 0, -12, 0, 0),
    (0, 0, 0, 6, 0),
    (0, 0, 0, -12, 0),
    (0, 0, 0, 0, 12),
    (0, 0, 0, 0, -12),
    (8, 8, -8, -8, 8),
    (-8, 16, 8, -4, -8),
    (6, 18, 6, -10, -6),
    (-6, 10, -6, -6, 6),
    (20, 0, 0, 0, 0),
    (-20, 0, 0, 0, 0),
    (0, 0, 20, 0, 0),
    (0, 0, -20, 0, 0),
)


def make_capture_targets(
    start: Sequence[int], limits: Sequence[tuple[int, int]]
) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
    if len(start) != 6 or len(limits) != 6:
        raise ValueError("起始位置和限位都必须包含 ID1~6")
    center = list(map(int, start))
    for index in range(5):
        minimum, maximum = limits[index]
        center[index] = min(
            maximum - LIMIT_MARGIN_RAW,
            max(minimum + LIMIT_MARGIN_RAW, center[index]),
        )
    center[5] = int(start[5])

    targets: list[tuple[int, ...]] = []
    for offsets in OFFSETS:
        target = center.copy()
        for index, offset in enumerate(offsets):
            target[index] += offset
        for index, (value, (minimum, maximum)) in enumerate(
            zip(target, limits, strict=True)
        ):
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"候选姿态 ID{index + 1}={value} 超出现场限位 {minimum}~{maximum}"
                )
            if index < 5 and abs(value - int(start[index])) > MAX_START_DELTA_RAW:
                raise ValueError(
                    f"候选姿态 ID{index + 1} 相对起点变化 {value - int(start[index])}，"
                    f"超过 {MAX_START_DELTA_RAW}"
                )
        targets.append(tuple(target))
    return tuple(center), targets


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开摄像头 {index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def read_fresh_frame(capture: cv2.VideoCapture, rotate: int) -> object:
    frame = None
    for _ in range(8):
        ok, candidate = capture.read()
        if ok:
            frame = candidate
        time.sleep(0.025)
    if frame is None:
        raise RuntimeError("摄像头读取失败")
    return rotate_frame(frame, rotate)


def wait_for_settle(
    bus: SCS215Bus,
    target: Sequence[int],
    max_tracking_error: int = 35,
    stable_delta: int = 2,
    timeout_s: float = 4.0,
) -> tuple[int, ...]:
    deadline = time.monotonic() + timeout_s
    previous = bus.get_positions()
    stable_reads = 0
    while True:
        time.sleep(0.08)
        actual = bus.get_positions()
        tracking_error = max(
            abs(a - b) for a, b in zip(actual, target, strict=True)
        )
        if tracking_error > max_tracking_error:
            raise RuntimeError(
                f"姿态跟随误差过大：目标 {tuple(target)}，实际 {actual}"
            )
        movement = max(abs(a - b) for a, b in zip(actual, previous, strict=True))
        stable_reads = stable_reads + 1 if movement <= stable_delta else 0
        if stable_reads >= 3:
            return actual
        if time.monotonic() >= deadline:
            raise RuntimeError(f"姿态在 {timeout_s:.1f}s 内未稳定：实际 {actual}")
        previous = actual


def load_existing_samples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("现有手眼清单的 samples 无效")
    return samples


def is_distinct_sample(actual: Sequence[int], samples: Sequence[dict]) -> bool:
    for sample in samples:
        previous = sample.get("raw_positions", [])
        if len(previous) == 6 and max(
            abs(int(a) - int(b))
            for a, b in zip(actual[:5], previous[:5], strict=True)
        ) < 5:
            return False
    return True


def write_manifest(path: Path, samples: list[dict]) -> None:
    payload = {
        "schema": 1,
        "notes": "自动采集；图片已按倒置相机旋转 180 度，棋盘为 8x6 内角点、25 mm 方格。",
        "samples": samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command_check(args: argparse.Namespace) -> int:
    with SCS215Bus(find_port(args.port)) as bus:
        states = bus.snapshot()
    validate_hardware(states)
    if any(item.torque for item in states):
        raise RuntimeError("离线规划前要求所有舵机扭矩为 0")
    start = tuple(item.position for item in states)
    limits = tuple((item.minimum, item.maximum) for item in states)
    center, targets = make_capture_targets(start, limits)
    print(f"起点：{start}")
    print(f"安全中心：{center}")
    print(f"候选姿态：{len(targets)} 个；最大起点偏移不超过 {MAX_START_DELTA_RAW}")
    for index, target in enumerate(targets, 1):
        print(f"{index:02d}: {target}")
    print("检查完成：未开启扭矩，未发送位置指令。")
    return 0


def command_real(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    mappings = load_mappings(Path(args.mapping))
    capture = open_camera(args.camera, args.width, args.height)
    enabled = False
    try:
        preflight = read_fresh_frame(capture, args.rotate)
        if find_chessboard(preflight, args.columns, args.rows) is None:
            raise RuntimeError("运动前图像未找到完整棋盘格，拒绝开启扭矩")

        with SCS215Bus(find_port(args.port)) as bus:
            states = bus.snapshot()
            validate_hardware(states)
            if any(item.torque for item in states):
                raise RuntimeError("启动前扭矩已开启，拒绝接管")
            start = tuple(item.position for item in states)
            limits = tuple((item.minimum, item.maximum) for item in states)
            center, targets = make_capture_targets(start, limits)
            controller = MotionController(
                bus,
                limits,
                max_speed_raw_s=args.max_speed,
                interval_s=0.05,
                real_time=True,
                max_tracking_error_raw=35,
            )
            for target in targets:
                controller.check_target(target)

            output_dir.mkdir(parents=True, exist_ok=True)
            hold_then_enable(bus, states)
            enabled = True
            samples = load_existing_samples(manifest_path)
            used_candidates = {
                int(sample["candidate_index"])
                for sample in samples
                if "candidate_index" in sample
            }
            if samples:
                print(f"续采：保留现有 {len(samples)} 个有效样本。", flush=True)
            try:
                controller.move_to(center, args.transition)
                wait_for_settle(bus, center)
                for candidate_index, target in enumerate(targets, 1):
                    if len(samples) >= args.target_count:
                        break
                    if candidate_index in used_candidates:
                        continue
                    controller.move_to(target, args.transition)
                    actual = wait_for_settle(bus, target)
                    frame = read_fresh_frame(capture, args.rotate)
                    corners = find_chessboard(frame, args.columns, args.rows)
                    if corners is None:
                        print(f"候选 {candidate_index:02d} 未识别完整棋盘，跳过。", flush=True)
                    elif not is_distinct_sample(actual, samples):
                        print(
                            f"候选 {candidate_index:02d} 与已有实际姿态过近，跳过 {actual}。",
                            flush=True,
                        )
                    else:
                        sample_number = len(samples) + 1
                        image_path = output_dir / f"pose_{sample_number:03d}.png"
                        _write_image(image_path, frame)
                        joints_deg = [
                            mapping.raw_to_degrees(raw)
                            for mapping, raw in zip(mappings, actual[:5], strict=True)
                        ]
                        samples.append(
                            {
                                "image": image_path.relative_to(manifest_path.parent).as_posix(),
                                "joints_deg": joints_deg,
                                "raw_positions": list(actual),
                                "candidate_index": candidate_index,
                            }
                        )
                        write_manifest(manifest_path, samples)
                        print(
                            f"已采集 {sample_number:02d}/{args.target_count}: "
                            f"候选 {candidate_index:02d}，实际 {actual}",
                            flush=True,
                        )
                    if target != center:
                        controller.move_to(center, args.transition)
                        wait_for_settle(bus, center)

                if len(samples) < args.target_count:
                    raise RuntimeError(
                        f"仅获得 {len(samples)} 个有效样本，少于目标 {args.target_count}"
                    )
                controller.move_to(center, args.transition)
                wait_for_settle(bus, center)
                print(f"采集完成：{len(samples)} 个有效样本，清单 {manifest_path.resolve()}")
            finally:
                try:
                    bus.set_torque(False)
                    enabled = False
                    print("已关闭六个舵机扭矩。")
                except Exception:
                    bus.broadcast_torque_off()
                    enabled = False
                    print("普通卸力失败，已广播关闭扭矩。")
    finally:
        capture.release()
        if enabled:
            print("警告：退出时无法确认扭矩状态，请立即断开舵机电源。", file=sys.stderr)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="眼在手上相机的手眼标定数据采集")
    parser.add_argument("--port", help="CH343 串口；省略时自动识别唯一设备")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--max-speed", type=float, default=35.0)
    parser.add_argument("--transition", type=float, default=1.2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--real", action="store_true")
    return parser


def main() -> int:
    try:
        args = make_parser().parse_args()
        if not 12 <= args.target_count <= len(OFFSETS):
            raise ValueError(f"--target-count 必须在 12~{len(OFFSETS)} 之间")
        if not 5.0 <= args.max_speed <= 45.0:
            raise ValueError("--max-speed 必须在 5~45 原始单位/秒之间")
        if not 0.8 <= args.transition <= 5.0:
            raise ValueError("--transition 必须在 0.8~5.0 秒之间")
        return command_check(args) if args.check else command_real(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
