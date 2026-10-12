from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from visual_closed_loop import (
    BallDetection,
    BlackBallDetector,
    PixelServo,
    SyntheticCamera,
    detections_match,
    horizontal_direction_for_error,
    plan_base_yaw_raw,
    plan_wrist_flex_raw,
    read_image,
    rotate_frame,
    write_image,
)


class DetectionTests(unittest.TestCase):
    def test_detects_black_circle_and_ignores_gray_distractors(self) -> None:
        image = np.full((360, 520, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (190, 55), (180, 180, 180), -1)
        cv2.circle(image, (380, 235), 48, (5, 5, 5), -1)
        cv2.circle(image, (365, 220), 7, (70, 70, 70), -1)
        found = BlackBallDetector().detect(image)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertAlmostEqual(found.center[0], 380, delta=3)
        self.assertAlmostEqual(found.center[1], 235, delta=3)
        self.assertGreater(found.confidence, 0.70)

    def test_rejects_dark_rectangle(self) -> None:
        image = np.full((300, 400, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (40, 120), (340, 160), (0, 0, 0), -1)
        self.assertIsNone(BlackBallDetector().detect(image))

    def test_hough_fallback_handles_ball_joined_to_shadow(self) -> None:
        image = np.full((360, 520, 3), 245, dtype=np.uint8)
        cv2.ellipse(image, (340, 250), (115, 24), 0, 0, 360, (55, 55, 55), -1)
        cv2.circle(image, (285, 205), 62, (12, 12, 12), -1, cv2.LINE_AA)
        cv2.circle(image, (265, 182), 8, (80, 80, 80), -1, cv2.LINE_AA)
        found = BlackBallDetector(min_circularity=0.90).detect(image)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertAlmostEqual(found.center[0], 285, delta=8)
        self.assertAlmostEqual(found.center[1], 205, delta=8)

    def test_rejects_oversized_dark_circle_and_selects_ball(self) -> None:
        image = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.circle(image, (320, 180), 125, (12, 12, 12), -1, cv2.LINE_AA)
        cv2.circle(image, (110, 385), 60, (8, 8, 8), -1, cv2.LINE_AA)
        found = BlackBallDetector().detect(image)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertAlmostEqual(found.center[0], 110, delta=5)
        self.assertAlmostEqual(found.center[1], 385, delta=5)
        self.assertLess(found.radius, 100)

    def test_unicode_path_round_trip(self) -> None:
        image = np.full((40, 60, 3), 123, dtype=np.uint8)
        test_dir = Path(__file__).resolve().parent / ".test_output"
        test_dir.mkdir(exist_ok=True)
        path = test_dir / "视觉结果.png"
        try:
            write_image(path, image)
            restored = read_image(path)
        finally:
            path.unlink(missing_ok=True)
            try:
                test_dir.rmdir()
            except OSError:
                pass
        self.assertTrue(np.array_equal(restored, image))

    def test_camera_rotation_is_applied_before_detection(self) -> None:
        image = np.zeros((2, 3), dtype=np.uint8)
        image[0, 0] = 255
        rotated = rotate_frame(image, 180)
        self.assertEqual(int(rotated[-1, -1]), 255)


class ServoTests(unittest.TestCase):
    @staticmethod
    def ball(x: float, y: float) -> BallDetection:
        return BallDetection((x, y), 30, 2800, 0.9, 0.9)

    def test_chooses_dominant_axis(self) -> None:
        servo = PixelServo(stable_frames=1)
        right = servo.decide(self.ball(550, 250), (480, 640, 3))
        self.assertEqual(right.direction, "right")
        above = servo.decide(self.ball(320, 80), (480, 640, 3))
        self.assertEqual(above.direction, "up")

    def test_dominant_axis_uses_pixel_distance_not_frame_aspect(self) -> None:
        servo = PixelServo(stable_frames=1)
        decision = servo.decide(self.ball(83, 422), (480, 640, 3))
        self.assertEqual(decision.direction, "left")

    def test_deadband_stops(self) -> None:
        decision = PixelServo(stable_frames=1).decide(
            self.ball(330, 230), (480, 640, 3)
        )
        self.assertEqual(decision.status, "centered")
        self.assertIsNone(decision.direction)

    def test_center_requires_stable_frames(self) -> None:
        servo = PixelServo(stable_frames=3)
        self.assertEqual(servo.decide(self.ball(320, 240), (480, 640, 3)).status, "centering")
        self.assertEqual(servo.decide(self.ball(320, 240), (480, 640, 3)).status, "centering")
        self.assertEqual(servo.decide(self.ball(320, 240), (480, 640, 3)).status, "centered")

    def test_requires_stable_detection_and_loss_resets_it(self) -> None:
        servo = PixelServo(stable_frames=2)
        self.assertEqual(servo.decide(self.ball(500, 240), (480, 640, 3)).status, "stabilizing")
        self.assertEqual(servo.decide(None, (480, 640, 3)).status, "target_lost")
        self.assertEqual(servo.decide(self.ball(500, 240), (480, 640, 3)).status, "stabilizing")
        self.assertEqual(servo.decide(self.ball(500, 240), (480, 640, 3)).status, "move")

    def test_target_jump_restarts_stability_gate(self) -> None:
        servo = PixelServo(stable_frames=2)
        self.assertEqual(servo.decide(self.ball(100, 240), (480, 640, 3)).status, "stabilizing")
        self.assertEqual(servo.decide(self.ball(500, 240), (480, 640, 3)).status, "stabilizing")
        self.assertEqual(servo.decide(self.ball(500, 240), (480, 640, 3)).status, "move")

    def test_detection_match_checks_center_and_radius(self) -> None:
        self.assertTrue(detections_match(self.ball(100, 100), self.ball(120, 110)))
        self.assertFalse(detections_match(self.ball(100, 100), self.ball(220, 100)))
        small = self.ball(100, 100)
        large = BallDetection((100, 100), 60, 11300, 0.9, 0.9)
        self.assertFalse(detections_match(small, large))

    def test_axis_inversion_is_explicit(self) -> None:
        servo = PixelServo(stable_frames=1, invert_x=True, invert_y=True)
        self.assertEqual(servo.decide(self.ball(550, 240), (480, 640, 3)).direction, "left")
        self.assertEqual(servo.decide(self.ball(320, 400), (480, 640, 3)).direction, "up")

    def test_horizontal_fallback_respects_camera_inversion(self) -> None:
        self.assertEqual(horizontal_direction_for_error(-80), "left")
        self.assertEqual(horizontal_direction_for_error(80), "right")
        self.assertEqual(horizontal_direction_for_error(-80, invert_x=True), "right")


class ClosedLoopTests(unittest.TestCase):
    def test_synthetic_camera_converges(self) -> None:
        scene = SyntheticCamera()
        detector = BlackBallDetector()
        servo = PixelServo(stable_frames=1)
        decision = None
        for _ in range(30):
            frame = scene.frame()
            decision = servo.decide(detector.detect(frame), frame.shape)
            if decision.status == "centered":
                break
            assert decision.direction is not None
            scene.move(decision.direction, decision.step_m)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.status, "centered")


class BaseYawPlanningTests(unittest.TestCase):
    def test_horizontal_visual_motion_uses_id1_in_current_pose(self) -> None:
        from pathlib import Path
        import sys

        project = Path(__file__).resolve().parent
        ik_dir = project.parent / "SO101_SCS215_IK键盘直线控制"
        sys.path.insert(0, str(ik_dir))
        from ik_core import URDFKinematics
        from ik_keyboard_control import load_mappings

        mappings = load_mappings(ik_dir / "joint_mapping.json")
        kinematics = URDFKinematics(project.parent / "SO101" / "so101_new_calib.urdf")
        raw = (421, 454, 515, 256, 488)
        joints = tuple(
            mapping.raw_to_degrees(value)
            for mapping, value in zip(mappings, raw, strict=True)
        )
        left = plan_base_yaw_raw(
            kinematics, mappings[0], raw[0], joints, (92, 807), "left", 0.002
        )
        right = plan_base_yaw_raw(
            kinematics, mappings[0], raw[0], joints, (92, 807), "right", 0.002
        )
        self.assertLess(left, raw[0])
        self.assertGreater(right, raw[0])
        self.assertEqual(abs(left - raw[0]), 8)
        self.assertEqual(abs(right - raw[0]), 8)
        long_right = plan_base_yaw_raw(
            kinematics, mappings[0], raw[0], joints, (92, 807), "right", 0.012
        )
        self.assertGreater(abs(long_right - raw[0]), 8)
        self.assertLessEqual(abs(long_right - raw[0]), 24)

    def test_vertical_visual_motion_uses_id4_arc(self) -> None:
        from pathlib import Path
        import sys

        project = Path(__file__).resolve().parent
        ik_dir = project.parent / "SO101_SCS215_IK键盘直线控制"
        sys.path.insert(0, str(ik_dir))
        from ik_core import URDFKinematics
        from ik_keyboard_control import load_mappings

        mappings = load_mappings(ik_dir / "joint_mapping.json")
        kinematics = URDFKinematics(project.parent / "SO101" / "so101_new_calib.urdf")
        raw = (390, 454, 515, 256, 488)
        joints = tuple(
            mapping.raw_to_degrees(value)
            for mapping, value in zip(mappings, raw, strict=True)
        )
        down = plan_wrist_flex_raw(
            kinematics, mappings[3], raw[3], joints, (43, 656), "down", 0.002
        )
        up = plan_wrist_flex_raw(
            kinematics, mappings[3], raw[3], joints, (43, 656), "up", 0.002
        )
        self.assertGreater(down, raw[3])
        self.assertLess(up, raw[3])
        self.assertEqual(abs(down - raw[3]), 8)
        self.assertEqual(abs(up - raw[3]), 8)
        long_down = plan_wrist_flex_raw(
            kinematics, mappings[3], raw[3], joints, (43, 656), "down", 0.010
        )
        self.assertGreater(abs(long_down - raw[3]), 8)
        self.assertLessEqual(abs(long_down - raw[3]), 24)


if __name__ == "__main__":
    unittest.main()
