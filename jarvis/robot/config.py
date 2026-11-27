from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(slots=True)
class RobotSettings:
    backend: str = "mock"
    port: str = "COM3"
    id: str = "so101-follower"
    poses_file: Path = Path("data/poses.json")
    max_relative_target_degrees: float = 12.0
    max_tool_delta_degrees: float = 8.0
    max_turn_motion_degrees: float = 30.0
    control_units_per_degree: float = 1.0
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-100.0, 100.0),
            "shoulder_lift": (-20.0, 80.0),
            "elbow_flex": (-90.0, 90.0),
            "wrist_flex": (-90.0, 90.0),
            "wrist_roll": (-180.0, 180.0),
            "gripper": (0.0, 100.0),
        }
    )


@dataclass(slots=True)
class CameraSettings:
    backend: str = "device"
    index: int = 0
    width: int = 960
    height: int = 540
    fps: int = 15
    jpeg_quality: int = 82
    fallback_to_demo: bool = True


@dataclass(slots=True)
class LLMSettings:
    enabled: bool = True
    backend: str = "qwen"
    base_url: str = "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1"
    model: str = "qwen3.8-omni-flash"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 120.0
    max_tool_rounds: int = 6
    history_turns: int = 4


@dataclass(slots=True)
class Settings:
    server: ServerSettings = field(default_factory=ServerSettings)
    robot: RobotSettings = field(default_factory=RobotSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    config_path: Path = field(default_factory=lambda: ROOT / "config.toml")
    root: Path = ROOT


def _construct(cls: type[Any], values: dict[str, Any]) -> Any:
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("SO101_CONFIG", "config.toml"))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件 {config_path}。请先复制 config.toml 或指定 --config。"
        )

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    robot_values = dict(raw.get("robot", {}))
    limits = robot_values.pop("joint_limits", None)
    robot = _construct(RobotSettings, robot_values)
    if limits:
        robot.joint_limits = {
            name: (float(values[0]), float(values[1])) for name, values in limits.items()
        }

    camera = _construct(CameraSettings, raw.get("camera", {}))
    llm = _construct(LLMSettings, raw.get("llm", {}))
    server = _construct(ServerSettings, raw.get("server", {}))

    poses_file = Path(robot.poses_file)
    if not poses_file.is_absolute():
        poses_file = config_path.parent / poses_file
    robot.poses_file = poses_file.resolve()

    return Settings(
        server=server,
        robot=robot,
        camera=camera,
        llm=llm,
        config_path=config_path,
        root=config_path.parent,
    )

