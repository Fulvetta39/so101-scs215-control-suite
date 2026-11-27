"""Persistent microphone/text voice control entry point for SO-101 SCS215."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

from asr_engine import MicrophoneASR, list_input_devices
from command_parser import CommandParser, VoiceCommand
from robot_adapter import (
    DEFAULT_MAPPING,
    DEFAULT_URDF,
    DryRunExecutor,
    RealRobotSession,
    diagnose_hardware,
)


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parent
DEFAULT_POSE_CONFIG = ROOT / "SO101_SCS215_多关节连续控制" / "poses.json"
DEFAULT_MODEL_DIR = (
    PROJECT_DIR / "models" / "sherpa-onnx-paraformer-zh-small-2024-03-09"
)
DEFAULT_VAD_MODEL = PROJECT_DIR / "models" / "silero_vad.onnx"


def load_pose_config(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("servo_ids") != [1, 2, 3, 4, 5, 6]:
        raise ValueError("姿态配置必须是 ID1~6 的 schema 1 文件")
    if not isinstance(data.get("poses"), dict) or not data["poses"]:
        raise ValueError("姿态配置中没有可用姿态")
    if not isinstance(data.get("actions"), dict):
        raise ValueError("姿态配置中的 actions 无效")
    return data


def build_command_parser(args: argparse.Namespace, config: dict) -> CommandParser:
    return CommandParser(
        config["poses"],
        config["actions"],
        default_distance_cm=args.default_distance_cm,
        max_distance_cm=args.max_distance_cm,
    )


def parse_and_report(text: str, parser: CommandParser) -> VoiceCommand:
    print(f"\n语音转录文本：{text}")
    command = parser.parse(text)
    print(f"指令解析：{command.describe()}")
    return command


def handle_text(
    text: str,
    parser: CommandParser,
    executor,
    *,
    asr_only: bool,
) -> bool:
    command = parse_and_report(text, parser)
    if command.kind == "exit":
        return False
    if asr_only:
        print("当前为仅识别模式，不执行动作。")
        return True
    if command.kind == "none":
        print("未通过指令白名单，机械臂保持不动。")
        return True
    executor.execute(command)
    return True


def choose_executor(args: argparse.Namespace, config: dict):
    if args.mode == "asr-only":
        return nullcontext(None)
    if args.mode == "dry-run":
        return DryRunExecutor(config)
    confirmation = input(
        "真机模式将在只读检查通过后开启扭矩。\n"
        "请托住机械臂，确认周围无障碍物且没有其他串口程序；输入 Y 继续："
    ).strip().lower()
    if confirmation not in {"y", "yes"}:
        raise RuntimeError("已取消真机语音控制；未连接、未开启扭矩")
    return RealRobotSession(
        config,
        port=args.port,
        urdf=Path(args.urdf),
        mapping=Path(args.mapping),
    )


def run_text(args: argparse.Namespace, config: dict) -> None:
    parser = build_command_parser(args, config)
    print("文本输入会话已启动。可输入“姿态三”、“往返演示”、“向前移动一厘米”。")
    print("输入“退出程序”结束。")
    with choose_executor(args, config) as executor:
        while True:
            try:
                text = input("\n指令> ").strip()
            except EOFError:
                break
            if not handle_text(
                text,
                parser,
                executor,
                asr_only=args.mode == "asr-only",
            ):
                break


def run_listen(args: argparse.Namespace, config: dict) -> None:
    parser = build_command_parser(args, config)
    recognizer = MicrophoneASR(
        Path(args.model_dir),
        Path(args.vad_model),
        device=args.device,
        num_threads=args.asr_threads,
    )
    with choose_executor(args, config) as executor:
        for text in recognizer.listen():
            keep_listening = handle_text(
                text,
                parser,
                executor,
                asr_only=args.mode == "asr-only",
            )
            if not keep_listening:
                break
            print("请输入语音指令：")


def show_devices() -> None:
    devices = list_input_devices()
    if not devices:
        print("没有发现可用的麦克风输入设备。")
        return
    print("麦克风输入设备：")
    for index, name, channels, sample_rate in devices:
        print(f"  {index}: {name}，输入通道 {channels}，默认 {sample_rate:.0f} Hz")


def add_shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pose-config", default=str(DEFAULT_POSE_CONFIG))
    parser.add_argument("--default-distance-cm", type=float, default=0.5)
    parser.add_argument("--max-distance-cm", type=float, default=20.0)
    parser.add_argument("--port", help="CH343 串口；真机时可省略以自动识别")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))


def add_mode_options(parser: argparse.ArgumentParser, *, include_asr_only: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    if include_asr_only:
        group.add_argument("--asr-only", dest="mode", action="store_const", const="asr-only")
    group.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run")
    group.add_argument("--real", dest="mode", action="store_const", const="real")
    parser.set_defaults(mode="asr-only" if include_asr_only else "dry-run")


def make_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SO-101 / SCS215 安全语音控制")
    add_shared_options(result)
    sub = result.add_subparsers(dest="command", required=True)

    devices = sub.add_parser("devices", help="列出麦克风输入设备")
    devices.set_defaults(func=lambda args, config: show_devices())

    diagnose = sub.add_parser("diagnose", help="只读诊断机械臂，不移动不改扭矩")
    diagnose.set_defaults(
        func=lambda args, config: diagnose_hardware(
            port=args.port, urdf=Path(args.urdf), mapping=Path(args.mapping)
        )
    )

    text = sub.add_parser("text", help="用键盘文本验证解析和动作层")
    add_mode_options(text, include_asr_only=False)
    text.set_defaults(func=run_text)

    listen = sub.add_parser("listen", help="从麦克风持续识别语音")
    add_mode_options(listen, include_asr_only=True)
    listen.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    listen.add_argument("--vad-model", default=str(DEFAULT_VAD_MODEL))
    listen.add_argument("--device", type=int, help="devices 命令显示的输入设备编号")
    listen.add_argument("--asr-threads", type=int, default=2)
    listen.set_defaults(func=run_listen)
    return result


def main() -> int:
    try:
        args = make_parser().parse_args()
        config = load_pose_config(Path(args.pose_config))
        args.func(args, config)
        return 0
    except KeyboardInterrupt:
        print("\n已由用户中止。")
        return 130
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
