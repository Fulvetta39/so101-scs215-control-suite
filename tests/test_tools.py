from pathlib import Path

from so101_agent.camera import CameraService
from so101_agent.config import load_settings
from so101_agent.robot import ArmController
from so101_agent.tools import ArmToolbox


def test_toolbox_moves_and_reports_refresh() -> None:
    settings = load_settings("config.toml")
    settings.camera.backend = "demo"
    arm = ArmController(settings.robot)
    camera = CameraService(settings.camera)
    arm.connect()
    camera.start()
    try:
        toolbox = ArmToolbox(arm, camera, max_turn_motion_degrees=30)
        toolbox.begin_turn()
        result = toolbox.execute("move_joint", {"joint": "elbow_flex", "delta_degrees": 3})
        assert result["ok"] is True
        assert result["refresh_image"] is True
        assert toolbox.events[-1].tool == "move_joint"
    finally:
        camera.stop()
        arm.disconnect()


def test_toolbox_rejects_cumulative_motion() -> None:
    settings = load_settings("config.toml")
    settings.camera.backend = "demo"
    arm = ArmController(settings.robot)
    camera = CameraService(settings.camera)
    arm.connect()
    camera.start()
    try:
        toolbox = ArmToolbox(arm, camera, max_turn_motion_degrees=5)
        toolbox.begin_turn()
        first = toolbox.execute("move_joint", {"joint": "elbow_flex", "delta_degrees": 4})
        second = toolbox.execute("move_joint", {"joint": "elbow_flex", "delta_degrees": 4})
        assert first["ok"] is True
        assert second["ok"] is False
        assert "累计动作" in second["message"]
    finally:
        camera.stop()
        arm.disconnect()

