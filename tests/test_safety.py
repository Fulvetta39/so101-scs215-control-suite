"""Stop/hold semantics, per-joint degree conversion and follow-error watchdog."""

import time
from types import SimpleNamespace

import pytest

from jarvis.robot.config import load_settings
from jarvis.robot.robot import (
    JOINTS,
    SCS215_DEGREES_PER_STEP,
    ArmController,
    LeRobotArm,
    MockArm,
    RobotError,
)

# The handoff calibration of the real arm (calibration/so101-follower.json).
RANGES = {
    "shoulder_pan": (92, 807),
    "shoulder_lift": (453, 1022),
    "elbow_flex": (0, 517),
    "wrist_flex": (31, 660),
    "wrist_roll": (427, 1022),
    "gripper": (127, 518),
}


class RecordingArm(MockArm):
    """Mock arm that records hardware-level stop calls."""

    def __init__(self, initial, units=None):
        super().__init__(initial)
        self.calls: list[str] = []
        self._units = units
        self.lag: dict[str, float] = {}

    def hold(self) -> None:
        self.calls.append("hold")

    def freeze(self) -> None:
        self.calls.append("freeze")

    def resume(self) -> None:
        self.calls.append("resume")

    def units_per_degree(self):
        return self._units

    def get_positions(self):
        positions = super().get_positions()
        for joint, offset in self.lag.items():
            positions[joint] -= offset
        return positions


def _controller(units=None, conversion="fixed"):
    settings = load_settings("config.toml")
    settings.robot.degree_conversion = conversion
    controller = ArmController(settings.robot)
    controller.backend = RecordingArm(controller.backend.get_positions(), units)
    controller.connect()
    return controller


def test_halt_keeps_torque_and_resume_does_not_cycle_torque() -> None:
    controller = _controller()
    controller.halt()
    assert controller.stopped
    assert controller.status()["torque_off"] is False
    assert controller.backend.calls == ["freeze"]
    with pytest.raises(RobotError, match="急停"):
        controller.move_joint("shoulder_pan", 2)
    controller.resume_from_emergency()
    assert "resume" not in controller.backend.calls
    assert controller.move_joint("shoulder_pan", 2)["ok"] is True
    controller.disconnect()


def test_emergency_stop_removes_torque_and_resume_reenables() -> None:
    controller = _controller()
    controller.emergency_stop()
    assert controller.status()["torque_off"] is True
    assert controller.backend.calls == ["hold"]
    controller.halt()  # a later halt must not try to freeze a limp arm
    assert "freeze" not in controller.backend.calls
    controller.resume_from_emergency()
    assert controller.backend.calls[-1] == "resume"
    assert controller.status()["torque_off"] is False
    controller.disconnect()


def _calibrated_units():
    arm = LeRobotArm.__new__(LeRobotArm)
    calibration = {
        joint: SimpleNamespace(range_min=low, range_max=high)
        for joint, (low, high) in RANGES.items()
    }
    arm._robot = SimpleNamespace(calibration=calibration)
    return arm.units_per_degree()


def test_units_per_degree_follow_calibration_range() -> None:
    units = _calibrated_units()
    # shoulder_pan: 715 steps * 300/1024 deg = 209.5 deg mapped onto 200 units.
    assert units["shoulder_pan"] == pytest.approx(200 / (715 * SCS215_DEGREES_PER_STEP))
    assert units["shoulder_pan"] == pytest.approx(0.955, abs=1e-3)
    assert units["gripper"] == pytest.approx(100 / (391 * SCS215_DEGREES_PER_STEP))
    assert set(units) == set(JOINTS)


def test_calibrated_conversion_turns_the_requested_real_angle() -> None:
    units = _calibrated_units()
    controller = _controller(units, conversion="calibration")
    before = controller.positions()["shoulder_pan"]
    result = controller.move_joint("shoulder_pan", 3.0)
    moved_units = controller.positions()["shoulder_pan"] - before
    assert moved_units == pytest.approx(3.0 * units["shoulder_pan"])
    assert result["delta_degrees"] == pytest.approx(3.0)
    assert controller.status()["degree_conversion"] == "calibration"
    controller.disconnect()


def test_follow_error_watchdog_halts_smooth_servo(monkeypatch) -> None:
    controller = _controller()
    monkeypatch.setattr(ArmController, "SMOOTH_FOLLOW_ERROR_S", 0.1)
    # Measured position trails the command by 20 degrees: a blocked joint.
    controller.backend.lag["shoulder_pan"] = 20.0
    controller.update_smooth_target({"shoulder_pan": 1.0})
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not controller.stopped:
        time.sleep(0.02)
    assert controller.stopped
    assert controller.status()["torque_off"] is False
    assert "freeze" in controller.backend.calls
    assert "跟随误差" in (controller.status()["last_error"] or "")
    controller.disconnect()
