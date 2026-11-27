from pathlib import Path

from so101_agent.config import load_settings


def test_load_default_settings() -> None:
    settings = load_settings("config.toml")
    assert settings.robot.backend == "mock"
    assert settings.llm.model == "qwen3.8-omni-flash"
    assert settings.robot.poses_file.is_absolute()
    assert settings.robot.poses_file.name == "poses.json"


def test_load_custom_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"
    config_path.write_text(
        """
[robot]
backend = "mock"
poses_file = "poses/custom.json"

[camera]
backend = "demo"
""",
        encoding="utf-8",
    )
    settings = load_settings(config_path)
    assert settings.robot.poses_file == (tmp_path / "poses/custom.json").resolve()
    assert settings.camera.backend == "demo"

