from pathlib import Path
import time

import pytest

from jarvis.robot.config import load_settings
from jarvis.robot.robot import ArmController, PoseStore, RobotError


def test_pose_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "poses.json"
    store = PoseStore(path)
    pose = {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "wrist_roll": 5,
        "gripper": 60,
    }
    store.save("test", pose)
    loaded = store.load()
    assert loaded["test"] == {key: float(value) for key, value in pose.items()}


def test_controller_clamps_joint_delta_and_limits() -> None:
    settings = load_settings("config.toml")
    controller = ArmController(settings.robot)
    controller.connect()
    before = controller.positions()
    result = controller.move_joint("shoulder_pan", 100)
    after = controller.positions()
    assert result["ok"] is True
    assert after["shoulder_pan"] - before["shoulder_pan"] == pytest.approx(
        settings.robot.max_tool_delta_degrees
    )
    controller.disconnect()


def test_emergency_stop_blocks_further_motion() -> None:
    settings = load_settings("config.toml")
    controller = ArmController(settings.robot)
    controller.connect()
    controller.emergency_stop()
    with pytest.raises(RobotError, match="急停"):
        controller.move_joint("shoulder_pan", 2)
    resumed = controller.resume_from_emergency()
    assert resumed["ok"] is True
    assert resumed["stopped"] is False
    assert controller.move_joint("shoulder_pan", 2)["ok"] is True
    controller.disconnect()


def test_named_pose_motion() -> None:
    settings = load_settings("config.toml")
    controller = ArmController(settings.robot)
    controller.connect()
    result = controller.move_to_pose("observe")
    assert result["ok"] is True
    pose = controller.poses.load()["observe"]
    assert controller.positions()["shoulder_pan"] == pytest.approx(pose["shoulder_pan"], abs=0.1)
    controller.disconnect()


def test_multi_joint_visual_servo_step() -> None:
    settings = load_settings("config.toml")
    controller = ArmController(settings.robot)
    controller.connect()
    before = controller.positions()
    result = controller.move_joints(
        {"shoulder_pan": 2.0, "wrist_flex": -2.0}
    )
    after = controller.positions()
    assert result["ok"] is True
    assert after["shoulder_pan"] > before["shoulder_pan"]
    assert after["wrist_flex"] < before["wrist_flex"]
    assert after["wrist_roll"] == before["wrist_roll"]
    assert after["gripper"] == before["gripper"]
    controller.disconnect()


def test_smooth_visual_servo_accumulates_without_blocking() -> None:
    settings = load_settings("config.toml")
    controller = ArmController(settings.robot)
    controller.connect()
    before = controller.positions()
    result = controller.update_smooth_target({"shoulder_pan": 6.0})
    assert result["ok"] is True
    assert result["tracking"] is True
    deadline = time.monotonic() + 1.0
    moved = before
    while time.monotonic() < deadline:
        moved = controller.positions()
        if moved["shoulder_pan"] - before["shoulder_pan"] > 0.2:
            break
        time.sleep(0.02)
    assert moved["shoulder_pan"] > before["shoulder_pan"]
    assert controller.status()["smooth_tracking"] is True
    controller.disconnect()


def test_smooth_visual_servo_does_not_accumulate_stale_error() -> None:
    settings = load_settings("config.toml")
    controller = ArmController(settings.robot)
    controller.connect()
    controller.update_smooth_target({"shoulder_pan": 6.0})
    controller.update_smooth_target({"shoulder_pan": 6.0})
    with controller._smooth_lock:
        lead = (
            controller._smooth_target["shoulder_pan"]
            - controller._smooth_command["shoulder_pan"]
        )
    assert abs(lead) <= controller.SMOOTH_TARGET_LEAD + 1e-6
    controller.disconnect()

