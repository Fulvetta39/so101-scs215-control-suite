from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from ik_core import (
    DIRECTION_JOINT_MOTION_WEIGHTS,
    CartesianController,
    DryRunDriver,
    URDFKinematics,
)
from ik_keyboard_control import (
    DIRECTION_LOAD_BIAS_DEFAULTS,
    EXPECTED_MODEL,
    DISTANCE_FEEDFORWARD,
    JointMapping,
    SCS215ArmDriver,
    ServoState,
    actual_directional_distance,
    choose_start_mode,
    decode_control_key,
    guard_forward_height_bias,
    parser,
    packet,
    parse_direction,
    pressed_direction,
    raw_speed_to_register,
    sync_position_packet,
)


ROOT = Path(__file__).resolve().parent
URDF = ROOT.parent / "SO101" / "so101_new_calib.urdf"


class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = URDFKinematics(URDF)

    def test_urdf_chain_and_finite_forward_position(self) -> None:
        self.assertEqual(len(self.kinematics.active_joints), 5)
        xyz = self.kinematics.position((0, 0, 0, 0, 0))
        self.assertEqual(len(xyz), 3)
        self.assertTrue(all(math.isfinite(value) for value in xyz))

    def test_urdf_limits_include_confirmed_folded_storage_pose(self) -> None:
        limits = self.kinematics.joint_limits_deg
        self.assertLessEqual(limits[1][0], -109.86)
        self.assertGreaterEqual(limits[2][1], 97.55)

    def test_inverse_recovers_nearby_position(self) -> None:
        initial = (-45, 45, -65, 25, 0)
        start = np.asarray(self.kinematics.position(initial))
        target = start + np.asarray((0.0, 0.0, 0.01))
        result = self.kinematics.inverse(target, initial)
        self.assertTrue(result.success, result.reason)
        self.assertLessEqual(result.residual_m, 0.0008)

    def test_cartesian_waypoints_stay_on_requested_line(self) -> None:
        initial = (-45, 45, -65, 25, 0)
        driver = DryRunDriver(initial)
        controller = CartesianController(
            self.kinematics,
            driver,
            command_interval_s=0,
            cartesian_resolution_m=0.002,
        )
        controller.initialize()
        start = np.asarray(self.kinematics.position(initial))
        result = controller.move("down", 0.02)
        self.assertTrue(result.full_success, result.message)
        for joints in driver.history:
            xyz = np.asarray(self.kinematics.position(joints))
            self.assertLess(abs(xyz[0] - start[0]), 0.001)
            self.assertLess(abs(xyz[1] - start[1]), 0.001)

    def test_unreachable_move_keeps_partial_same_direction_path(self) -> None:
        driver = DryRunDriver((-45, 45, -65, 25, 0))
        controller = CartesianController(self.kinematics, driver, command_interval_s=0)
        controller.initialize()
        result = controller.move("forward", 0.50)
        self.assertFalse(result.full_success)
        self.assertIn("IK 无解", result.message)
        self.assertGreater(result.executed_m, 0)
        self.assertLess(result.executed_m, result.requested_m)

    def test_actual_directional_distance_uses_requested_axis(self) -> None:
        driver = DryRunDriver((-45, 45, -65, 25, 0))
        controller = CartesianController(self.kinematics, driver, command_interval_s=0)
        controller.initialize()
        result = controller.move("up", 0.01)
        actual = (result.start_xyz[0] + 0.02, result.start_xyz[1], result.start_xyz[2] + 0.006)
        self.assertAlmostEqual(actual_directional_distance(result, actual), 0.006)

    def test_physical_up_uses_positive_urdf_z_after_joint_calibration(self) -> None:
        from ik_core import DIRECTIONS

        self.assertEqual(DIRECTIONS["up"], (0.0, 0.0, 1.0))
        self.assertEqual(DIRECTIONS["down"], (0.0, 0.0, -1.0))

    def test_final_target_only_sends_one_synchronized_goal(self) -> None:
        initial = (-45, 45, -65, 25, 0)
        driver = DryRunDriver(initial)
        controller = CartesianController(
            self.kinematics,
            driver,
            command_interval_s=0,
            cartesian_resolution_m=0.002,
        )
        controller.initialize()
        result = controller.move("down", 0.01, final_target_only=True)
        self.assertTrue(result.full_success, result.message)
        self.assertGreater(result.waypoint_count, 1)
        self.assertEqual(len(driver.history), 1)

    def test_stream_targets_preplans_then_sends_all_waypoints(self) -> None:
        initial = (-45, 45, -65, 25, 0)
        driver = DryRunDriver(initial)
        controller = CartesianController(
            self.kinematics,
            driver,
            command_interval_s=0,
            cartesian_resolution_m=0.002,
        )
        controller.initialize()
        result = controller.move("down", 0.01, stream_targets=True)
        self.assertTrue(result.full_success, result.message)
        self.assertEqual(result.waypoint_count, 5)
        self.assertEqual(len(driver.history), 5)
        self.assertIn("固定节拍", result.message)

    def test_absolute_xyz_correction_locks_all_three_axes(self) -> None:
        initial = (-45, 45, -65, 25, 0)
        driver = DryRunDriver(initial)
        controller = CartesianController(
            self.kinematics,
            driver,
            command_interval_s=0,
            cartesian_resolution_m=0.002,
        )
        controller.initialize()
        start = np.asarray(self.kinematics.position(initial))
        target = start + np.asarray((0.006, -0.003, 0.005))
        result = controller.move_to_xyz(target, stream_targets=True)
        self.assertTrue(result.full_success, result.message)
        final = np.asarray(self.kinematics.position(controller.joints_deg))
        self.assertLess(float(np.linalg.norm(final - target)), 0.001)
        self.assertEqual(len(driver.history), result.waypoint_count)

    def test_downward_ik_uses_wrist_before_consuming_elbow_range(self) -> None:
        initial = (0.0, -80.0, 30.0, 20.0, 0.0)
        start = np.asarray(self.kinematics.position(initial))
        target = start + np.asarray((0.0, 0.0, -0.05))

        plain_driver = DryRunDriver(initial)
        plain = CartesianController(
            self.kinematics,
            plain_driver,
            command_interval_s=0,
            cartesian_resolution_m=0.002,
        )
        plain.initialize()
        plain_result = plain.move_to_xyz(target, stream_targets=True)

        balanced_driver = DryRunDriver(initial)
        balanced = CartesianController(
            self.kinematics,
            balanced_driver,
            command_interval_s=0,
            cartesian_resolution_m=0.002,
        )
        balanced.initialize()
        balanced_result = balanced.move_to_xyz(
            target,
            stream_targets=True,
            joint_motion_weights=DIRECTION_JOINT_MOTION_WEIGHTS["down"],
        )

        self.assertTrue(plain_result.full_success, plain_result.message)
        self.assertTrue(balanced_result.full_success, balanced_result.message)
        plain_change = np.abs(np.asarray(plain.joints_deg) - np.asarray(initial))
        balanced_change = np.abs(np.asarray(balanced.joints_deg) - np.asarray(initial))
        self.assertLess(balanced_change[2], plain_change[2] * 0.5)
        self.assertGreater(balanced_change[3], plain_change[3])
        for joints in balanced_driver.history:
            xyz = np.asarray(self.kinematics.position(joints))
            self.assertLess(abs(xyz[0] - start[0]), 0.001)
            self.assertLess(abs(xyz[1] - start[1]), 0.001)


class MappingAndSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.id1 = JointMapping(
            servo_id=1,
            name="shoulder_pan",
            zero_raw=0,
            direction=1,
            wrapped=True,
            guard_min=600,
            guard_max=1022,
        )

    def test_id1_uses_new_non_wrapping_calibration(self) -> None:
        self.id1 = JointMapping(1, "shoulder_pan", 449.5, 1, False, 92, 807)
        self.assertAlmostEqual(self.id1.raw_to_degrees(449.5), 0.0)
        for raw in (92, 200, 422, 600, 807):
            self.assertEqual(self.id1.degrees_to_raw(self.id1.raw_to_degrees(raw)), raw)

    def test_safe_ranges_use_new_id1_eeprom_limits(self) -> None:
        self.id1 = JointMapping(1, "shoulder_pan", 449.5, 1, False, 92, 807)
        mappings = (
            self.id1,
            JointMapping(2, "shoulder_lift", 511.5, 1),
            JointMapping(3, "elbow_flex", 511.5, 1),
            JointMapping(4, "wrist_flex", 511.5, 1),
            JointMapping(5, "wrist_roll", 511.5, 1),
        )
        states = tuple(
            ServoState(i, EXPECTED_MODEL, p, low, high, 0)
            for i, p, low, high in (
                (1, 422, 92, 807),
                (2, 753, 5, 1017),
                (3, 238, 5, 1017),
                (4, 595, 35, 595),
                (5, 493, 5, 1017),
                (6, 157, 128, 515),
            )
        )
        arm = object.__new__(SCS215ArmDriver)
        arm.states = states
        arm.mappings = mappings
        arm.urdf_limits_deg = URDFKinematics(URDF).joint_limits_deg
        ranges = SCS215ArmDriver._safe_ranges(arm)
        self.assertEqual(ranges[0], (92, 807))
        self.assertEqual(ranges[3][1], 595)

    def test_hold_rejects_already_enabled_torque(self) -> None:
        arm = object.__new__(SCS215ArmDriver)
        arm.states = tuple(
            ServoState(i, EXPECTED_MODEL, 700, 0, 1022, 1) for i in range(1, 7)
        )
        with self.assertRaisesRegex(RuntimeError, "扭矩已经开启"):
            SCS215ArmDriver.hold_then_enable(arm)

    def test_load_bias_blends_through_trajectory_and_learns_by_direction(self) -> None:
        arm = object.__new__(SCS215ArmDriver)
        arm.safe_raw_ranges = (
            (92, 807),
            (453, 1022),
            (0, 516),
            (31, 656),
            (200, 720),
        )
        arm.direction_load_bias = {
            "forward": (2, 1, -10, -6, 0),
            "back": (-2, 1, -8, 0, 0),
        }
        nominal = (430, 566, 319, 448, 473, 136)
        self.assertEqual(arm.begin_trajectory_load_bias("forward", 4), (2, 1, -10, -6, 0))
        waypoints = [arm._apply_trajectory_load_bias(nominal) for _ in range(4)]
        self.assertEqual(waypoints[0], (430, 566, 317, 447, 473, 136))
        self.assertEqual(waypoints[-1], (432, 567, 309, 442, 473, 136))
        self.assertEqual(
            [row[2] for row in waypoints],
            sorted((row[2] for row in waypoints), reverse=True),
        )
        arm.end_trajectory_load_bias()
        self.assertEqual(arm._apply_trajectory_load_bias(nominal), nominal)

        back_nominal = (423, 485, 338, 500, 473, 136)
        back_actual = (425, 485, 343, 500, 473, 136)
        learned = arm.learn_direction_load_bias("back", back_nominal, back_actual)
        self.assertEqual(learned, (-3, 1, -10, 0, 0))

        forward_actual = (430, 566, 316, 448, 473, 136)
        learned = arm.learn_direction_load_bias("forward", nominal, forward_actual)
        self.assertEqual(learned[2], -8)

    def test_forward_height_guard_prevents_stronger_id3_lift_bias(self) -> None:
        arm = object.__new__(SCS215ArmDriver)
        arm.direction_load_bias = {"forward": (4, -2, -12, -10, 0)}
        guarded = guard_forward_height_bias(arm, "forward", 0.2754, 0.2764)
        self.assertTrue(guarded)
        self.assertEqual(arm.direction_load_bias["forward"], (4, -2, -10, -10, 0))

        guarded = guard_forward_height_bias(arm, "back", 0.2754, 0.2800)
        self.assertFalse(guarded)
        self.assertEqual(DIRECTION_LOAD_BIAS_DEFAULTS["forward"], (4, -2, -10, -10, 0))

    def test_forward_learning_keeps_id3_load_share_cap(self) -> None:
        arm = object.__new__(SCS215ArmDriver)
        arm.direction_load_bias = {"forward": (4, -2, -10, -10, 0)}
        nominal = (431, 548, 358, 410, 492, 136)
        actual = (431, 548, 374, 410, 492, 136)
        learned = arm.learn_direction_load_bias("forward", nominal, actual)
        self.assertEqual(learned[2], -10)

    def test_lateral_motion_keeps_gravity_support_without_id1_bias(self) -> None:
        self.assertEqual(DIRECTION_LOAD_BIAS_DEFAULTS["left"], (0, -2, -10, -10, 0))
        self.assertEqual(DIRECTION_LOAD_BIAS_DEFAULTS["right"], (0, -2, -10, -10, 0))
        arm = object.__new__(SCS215ArmDriver)
        arm.direction_load_bias = {"left": (0, -2, -10, -10, 0)}
        nominal = (500, 548, 358, 410, 492, 136)
        actual = (500, 548, 374, 410, 492, 136)
        learned = arm.learn_direction_load_bias("left", nominal, actual)
        self.assertEqual(learned[0], 0)
        self.assertEqual(learned[2], -10)

class ProtocolTests(unittest.TestCase):
    def test_packet_checksum(self) -> None:
        data = packet(1, 2, (56, 2))
        self.assertEqual(data[:2], b"\xff\xff")
        self.assertEqual(data[-1], (~sum(data[2:-1])) & 0xFF)

    def test_sync_packet_contains_all_six_ids(self) -> None:
        data = sync_position_packet((600, 500, 400, 300, 200, 128))
        self.assertEqual(data[2], 0xFE)
        self.assertEqual(data[6], 2)
        for servo_id in range(1, 7):
            self.assertIn(servo_id, data[7:-1])

    def test_speed_register_conversion_is_bounded(self) -> None:
        self.assertGreater(raw_speed_to_register(90), 0)
        self.assertEqual(raw_speed_to_register(1e9), 1500)


class UserInterfaceTests(unittest.TestCase):
    def test_default_motion_period_is_slightly_faster(self) -> None:
        args = parser().parse_args([])
        self.assertAlmostEqual(args.interval, 1.0 / 30.0)

    def test_vertical_distance_feedforward_is_neutral_after_recalibration(self) -> None:
        self.assertAlmostEqual(DISTANCE_FEEDFORWARD["up"], 1.0)
        self.assertAlmostEqual(DISTANCE_FEEDFORWARD["down"], 1.0)

    def test_wasd_and_arrow_keys_have_requested_directions(self) -> None:
        self.assertEqual(decode_control_key("w"), "forward")
        self.assertEqual(decode_control_key("S"), "back")
        self.assertEqual(decode_control_key("a"), "left")
        self.assertEqual(decode_control_key("D"), "right")
        self.assertEqual(decode_control_key("\xe0", "H"), "up")
        self.assertEqual(decode_control_key("\xe0", "P"), "down")

    def test_pressed_direction_uses_live_key_state(self) -> None:
        held = {0x57}
        self.assertEqual(pressed_direction(lambda key: key in held), "forward")
        held = {0x26}
        self.assertEqual(pressed_direction(lambda key: key in held), "up")
        held = {0x57, 0x44}
        self.assertIsNone(pressed_direction(lambda key: key in held))
        self.assertIsNone(pressed_direction(lambda key: False))

    def test_distance_mode_accepts_chinese_directions(self) -> None:
        self.assertEqual(parse_direction("前"), "forward")
        self.assertEqual(parse_direction("上"), "up")
        self.assertEqual(parse_direction("↓"), "down")
        self.assertIsNone(parse_direction("斜着"))

    def test_start_menu_reprompts_invalid_choice(self) -> None:
        answers = iter(("x", "b"))
        self.assertEqual(choose_start_mode(lambda: next(answers)), "free")

    def test_start_menu_escape_exits(self) -> None:
        self.assertIsNone(choose_start_mode(lambda: "\x1b"))


if __name__ == "__main__":
    unittest.main()
