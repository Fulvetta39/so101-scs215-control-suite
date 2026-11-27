import unittest

from multi_joint_control import (
    DryRunBus,
    MotionController,
    interpolate_trajectory,
    make_packet,
    make_sync_position_packet,
    parse_pose_sequence,
    parse_transition_times,
    validate_config,
)


class ProtocolTests(unittest.TestCase):
    def test_read_packet_checksum(self):
        self.assertEqual(make_packet(1, 2, (3, 2)), bytes.fromhex("ff ff 01 04 02 03 02 f3"))

    def test_sync_packet_contains_six_positions(self):
        result = make_sync_position_packet((1, 2, 3, 4, 5, 1023))
        self.assertEqual(result[:7], bytes.fromhex("ff ff fe 16 83 2a 02"))
        self.assertEqual(result[-4:-1], bytes.fromhex("06 03 ff"))
        self.assertEqual(sum(result[2:]) & 0xFF, 0xFF)


class MotionTests(unittest.TestCase):
    def test_interpolation_reaches_target_and_extends_duration(self):
        points, actual = interpolate_trajectory((0,) * 6, (100,) * 6, 0.2, 50, 0.05)
        self.assertEqual(actual, 2.0)
        self.assertEqual(points[-1], (100,) * 6)
        self.assertGreater(len(points), 1)

    def test_controller_rejects_id1_zero_seam_transition(self):
        controller = MotionController(
            DryRunBus((990, 512, 512, 512, 512, 512)),
            ((0, 1022),) * 6,
            real_time=False,
        )
        with self.assertRaisesRegex(RuntimeError, "0/1022 接缝"):
            controller.move_to((0, 512, 512, 512, 512, 512), 1.5)

    def test_all_joints_use_global_220_speed_limit(self):
        controller = MotionController(
            DryRunBus((500, 500, 500, 600, 500, 500)),
            ((0, 1023),) * 6,
            max_speed_raw_s=220,
            real_time=False,
        )
        actual = controller.move_to((500, 500, 500, 150, 500, 500), 1.5)
        self.assertAlmostEqual(actual, 450 / 220)

    def test_controller_rejects_live_limit_violation(self):
        controller = MotionController(DryRunBus(), ((10, 900),) * 6, real_time=False)
        with self.assertRaisesRegex(ValueError, "现场限位"):
            controller.move_to((5, 512, 512, 512, 512, 512), 1.0)

    def test_dry_run_sends_smoothed_points(self):
        bus = DryRunBus()
        controller = MotionController(bus, ((0, 1023),) * 6, max_speed_raw_s=100, real_time=False)
        controller.move_to((600,) * 6, 1.0)
        self.assertGreater(len(bus.history), 1)
        self.assertEqual(bus.history[-1], (600,) * 6)


class ConfigTests(unittest.TestCase):
    def test_rejects_action_with_missing_pose(self):
        config = {
            "schema": 1,
            "servo_ids": [1, 2, 3, 4, 5, 6],
            "poses": {},
            "actions": {
                "bad": {
                    "key": "1",
                    "steps": [{"pose": "missing", "transition_s": 1.0, "hold_s": 0.0}],
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "不存在"):
            validate_config(config)


class InteractiveInputTests(unittest.TestCase):
    def test_pose_sequence_accepts_compact_and_separated_digits(self):
        available = {"0", "1", "2", "3"}
        self.assertEqual(parse_pose_sequence("0123", available), ["0", "1", "2", "3"])
        self.assertEqual(parse_pose_sequence("0, 2 3", available), ["0", "2", "3"])

    def test_pose_sequence_rejects_unsaved_pose(self):
        with self.assertRaisesRegex(ValueError, "没有保存"):
            parse_pose_sequence("1 4", {"1", "2"})

    def test_single_duration_is_reused_for_each_step(self):
        self.assertEqual(parse_transition_times("1.5", 3), [1.5, 1.5, 1.5])

    def test_per_step_durations_must_match_pose_count(self):
        self.assertEqual(parse_transition_times("1, 2, 3", 3), [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "一一对应"):
            parse_transition_times("1,2", 3)


if __name__ == "__main__":
    unittest.main()
