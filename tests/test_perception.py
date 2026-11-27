import json
import time

from jarvis.perception.geometry import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    WRIST,
    is_plausible_face,
    pointing_geometry,
    ray_to_border,
)
from jarvis.perception.tracker import FaceTracker, PointingTracker, select_primary_face
from jarvis.perception.servo import (
    FaceServoConfig,
    FaceServoPlanner,
    HttpPointingServoBridge,
    HttpServoBridge,
    PointingGazeConfig,
    PointingGazePlanner,
)
from jarvis.perception.naming import PointedObjectNamer
from jarvis.perception.types import (
    BoundingBox,
    FaceCandidate,
    PerceptionFrame,
    PointingRay,
    HandObservation,
    TrackedFace,
    UvPoint,
)


def make_face(cx: float, cy: float, confidence: float = 0.9) -> FaceCandidate:
    return FaceCandidate(
        bbox_px=BoundingBox(cx * 100 - 10, cy * 100 - 10, 20, 20),
        center_uv=(cx, cy),
        center_error=(2 * cx - 1, 2 * cy - 1),
        confidence=confidence,
    )


def test_primary_face_prefers_center_and_continuity() -> None:
    left = make_face(0.15, 0.5, 0.99)
    center = make_face(0.5, 0.5, 0.88)
    assert select_primary_face([left, center]).center_uv == center.center_uv
    assert select_primary_face([left, center], previous_center=(0.16, 0.5)).center_uv == left.center_uv


def test_face_tracker_smooths_and_holds_short_losses() -> None:
    tracker = FaceTracker(alpha=0.5, max_missed_frames=2, acquire_frames=1)
    first = tracker.update([make_face(0.4, 0.5)])
    second = tracker.update([make_face(0.6, 0.5)])
    assert first is not None and second is not None
    assert second.center_uv[0] == 0.5
    missed = tracker.update([])
    assert missed is not None and missed.missed_frames == 1
    tracker.update([])
    assert tracker.update([]) is None


def test_face_tracker_requires_two_consistent_detections() -> None:
    tracker = FaceTracker(acquire_frames=2)
    assert tracker.update([make_face(0.4, 0.5)]) is None
    assert tracker.update([make_face(0.42, 0.5)]) is not None


def test_face_geometry_rejects_implausible_keypoints() -> None:
    bbox = BoundingBox(10, 10, 100, 100)
    plausible = FaceCandidate(
        bbox_px=bbox,
        center_uv=(0.5, 0.5),
        center_error=(0.0, 0.0),
        confidence=0.8,
        keypoints_uv=(
            (0.35, 0.40),
            (0.65, 0.41),
            (0.50, 0.52),
            (0.50, 0.66),
            (0.28, 0.50),
            (0.72, 0.50),
        ),
    )
    assert is_plausible_face(plausible)
    invalid = FaceCandidate(
        bbox_px=bbox,
        center_uv=(0.5, 0.5),
        center_error=(0.0, 0.0),
        confidence=0.8,
        keypoints_uv=(
            (0.50, 0.40),
            (0.51, 0.41),
            (0.50, 0.52),
            (0.50, 0.66),
            (0.28, 0.50),
            (0.72, 0.50),
        ),
    )
    assert not is_plausible_face(invalid)


def test_pointing_geometry_detects_extended_index_and_border() -> None:
    points = [UvPoint(0.5, 0.5) for _ in range(21)]
    points[WRIST] = UvPoint(0.5, 0.82)
    points[INDEX_MCP] = UvPoint(0.45, 0.6)
    points[6] = UvPoint(0.44, 0.48)
    points[7] = UvPoint(0.43, 0.38)
    points[INDEX_TIP] = UvPoint(0.42, 0.25)
    for mcp, tip, x in ((MIDDLE_MCP, MIDDLE_TIP, 0.50), (RING_MCP, RING_TIP, 0.55), (PINKY_MCP, PINKY_TIP, 0.60)):
        points[mcp] = UvPoint(x, 0.62)
        points[mcp + 1] = UvPoint(x + 0.03, 0.61)
        points[mcp + 2] = UvPoint(x - 0.02, 0.65)
        points[tip] = UvPoint(x, 0.67)
    is_pointing, score, tip, direction, ray = pointing_geometry(points)
    assert is_pointing
    assert score > 0.6
    assert tip == (0.42, 0.25)
    assert direction is not None and direction[1] < 0
    assert ray is not None
    assert ray.end_uv[1] == 0.0


def test_ray_to_border_handles_all_directions() -> None:
    assert ray_to_border((0.5, 0.5), (1, 0)) == (1.0, 0.5)
    assert ray_to_border((0.5, 0.5), (-1, 0)) == (0.0, 0.5)
    assert ray_to_border((0.5, 0.5), (0, 1)) == (0.5, 1.0)
    assert ray_to_border((0.5, 0.5), (0, -1)) == (0.5, 0.0)


def make_frame(
    error: tuple[float, float] | None,
    *,
    timestamp: float = 1.0,
    missing: int = 0,
    pointing: bool = False,
) -> PerceptionFrame:
    face = None
    if error is not None:
        face = TrackedFace(
            bbox_px=BoundingBox(45, 45, 10, 10),
            center_uv=(0.5 + error[0] / 2, 0.5 + error[1] / 2),
            center_error=error,
            confidence=0.9,
            missed_frames=missing,
        )
    ray = (
        PointingRay((0.5, 0.5), (1.0, 0.0), (1.0, 0.5), 0.9)
        if pointing
        else None
    )
    return PerceptionFrame(
        timestamp_s=timestamp,
        width=100,
        height=100,
        primary_face=face,
        pointing=ray,
        attention="pointing" if pointing else "face" if face else "searching",
    )


def make_pointing_frame(
    direction: tuple[float, float],
    *,
    timestamp: float = 1.0,
    confidence: float = 0.9,
) -> PerceptionFrame:
    end = (
        max(0.0, min(1.0, 0.5 + direction[0])),
        max(0.0, min(1.0, 0.5 + direction[1])),
    )
    return PerceptionFrame(
        timestamp_s=timestamp,
        width=100,
        height=100,
        pointing=PointingRay((0.5, 0.5), direction, end, confidence),
        attention="pointing",
    )


def test_face_servo_shadow_has_deadband_and_step_limit() -> None:
    planner = FaceServoPlanner(
        FaceServoConfig(
            deadband=0.08,
            smoothing=1.0,
            pan_gain_deg=10.0,
            vertical_gain_deg=8.0,
            max_step_deg=2.0,
            pan_sign=1,
            vertical_sign=-1,
            vertical_joint="wrist_flex",
        )
    )
    centered = planner.update(make_frame((0.02, -0.02)))
    assert centered.tracking and not centered.move
    right = planner.update(make_frame((0.5, -0.5), timestamp=2.0))
    assert right.move
    assert 0 < right.pan_delta_deg <= 2.0
    assert 0 < right.vertical_delta_deg <= 2.0
    assert right.joint_deltas == {
        "shoulder_pan": right.pan_delta_deg,
        "wrist_flex": right.vertical_delta_deg,
    }


def test_face_servo_can_use_shoulder_lift_for_vertical_axis() -> None:
    planner = FaceServoPlanner(
        FaceServoConfig(
            deadband=0.0,
            smoothing=1.0,
            vertical_joint="shoulder_lift",
        )
    )
    result = planner.update(make_frame((0.0, -0.5)))
    assert "shoulder_lift" in result.joint_deltas
    assert "wrist_flex" not in result.joint_deltas


def test_vertical_minimum_effective_pulse() -> None:
    planner = FaceServoPlanner(
        FaceServoConfig(
            deadband=0.0,
            vertical_deadband=0.2,
            smoothing=1.0,
            vertical_gain_deg=1.0,
            vertical_max_step_deg=6.0,
            vertical_min_step_deg=6.0,
            vertical_joint="wrist_flex",
        )
    )
    result = planner.update(make_frame((0.0, 0.25)))
    assert result.vertical_delta_deg == 6.0
    assert result.joint_deltas["wrist_flex"] == 6.0
    stopped = planner.update(make_frame((0.0, 0.1), timestamp=2.0))
    assert not stopped.move


def test_face_servo_stops_when_face_lost_or_pointing() -> None:
    planner = FaceServoPlanner()
    lost = planner.update(make_frame(None))
    assert not lost.tracking and not lost.move
    assert lost.reason == "face_not_found"
    pointing = planner.update(make_frame((0.5, 0.0), pointing=True))
    assert not pointing.tracking
    assert pointing.reason == "pointing_has_priority"


def test_face_servo_searches_after_face_leaves_frame() -> None:
    planner = FaceServoPlanner(
        FaceServoConfig(
            pan_sign=-1,
            search_hold_s=0.1,
            search_out_s=0.4,
            search_back_s=0.4,
            search_reverse_pause_s=0.1,
            search_step_deg=3.0,
            search_interval_s=0.0,
        )
    )
    planner.update(make_frame((0.5, 0.0), timestamp=1.0))
    hold = planner.update(make_frame(None, timestamp=1.05))
    assert hold.reason == "face_search_hold"
    outward = planner.update(make_frame(None, timestamp=1.2))
    assert outward.reason == "search_last_direction"
    assert outward.pan_delta_deg == -3.0
    reverse_hold = planner.update(make_frame(None, timestamp=1.5))
    assert reverse_hold.reason == "face_search_reverse_hold"
    reverse = planner.update(make_frame(None, timestamp=1.6))
    assert reverse.reason == "search_return_sweep"
    assert reverse.pan_delta_deg == 3.0
    assert planner.update(make_frame(None, timestamp=2.1)).reason == "face_search_timeout"


def test_pointing_tracker_holds_a_brief_landmark_dropout() -> None:
    tracker = PointingTracker(alpha=1.0, release_grace_s=0.3)
    ray = PointingRay((0.5, 0.5), (1.0, 0.0), (1.0, 0.5), 0.9)
    hand = HandObservation(
        handedness="Right",
        confidence=0.9,
        landmarks_uv=(),
        is_pointing=True,
        pointing_score=0.9,
        index_tip_uv=(0.5, 0.5),
        index_direction_uv=(1.0, 0.0),
    )
    held_ray, held_hand = tracker.update(ray, hand, 1.0)
    assert held_ray is not None and held_hand is hand
    retained_ray, retained_hand = tracker.update(None, None, 1.2)
    assert retained_ray is not None
    assert retained_hand is hand
    released_ray, released_hand = tracker.update(None, None, 1.4)
    assert released_ray is None and released_hand is None


def test_http_bridge_defaults_to_horizontal_and_rate_limits(monkeypatch) -> None:
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        sent.append((request, timeout))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    bridge = HttpServoBridge(rate_hz=0.0001, max_step_deg=0.75)
    recommendation = FaceServoPlanner(
        FaceServoConfig(
            deadband=0.0,
            smoothing=1.0,
            pan_gain_deg=5.0,
            vertical_gain_deg=5.0,
            max_step_deg=2.0,
        )
    ).update(make_frame((0.8, -0.8)))
    first = bridge.maybe_send(recommendation)
    second = bridge.maybe_send(recommendation)
    assert first is not None and first["sent"]
    assert second is not None and second["reason"] == "horizontal_rate_limited"
    assert len(sent) == 1


def test_pointing_gaze_maps_direction_to_bounded_joint_turns() -> None:
    planner = PointingGazePlanner(
        PointingGazeConfig(
            smoothing=1.0,
            deadband=0.1,
            pan_gain_deg=12.0,
            vertical_gain_deg=8.0,
            max_step_deg=3.0,
            pan_sign=-1,
            vertical_sign=1,
            vertical_joint="wrist_flex",
        )
    )
    right_down = planner.update(make_pointing_frame((0.8, 0.6)))
    assert right_down.tracking and right_down.move
    assert right_down.reason == "move_pan_and_vertical"
    assert right_down.pan_delta_deg == -3.0
    assert right_down.vertical_delta_deg == 3.0
    assert right_down.joint_deltas == {
        "shoulder_pan": -3.0,
        "wrist_flex": 3.0,
    }

    left_up = planner.update(make_pointing_frame((-0.6, -0.8), timestamp=2.0))
    assert left_up.pan_delta_deg == 3.0
    assert left_up.vertical_delta_deg == -3.0


def test_pointing_gaze_stops_for_missing_or_uncertain_pointing() -> None:
    planner = PointingGazePlanner()
    missing = planner.update(make_frame(None))
    assert not missing.tracking and not missing.move
    assert missing.reason == "pointing_not_found"

    uncertain = planner.update(make_pointing_frame((1.0, 0.0), confidence=0.3))
    assert not uncertain.tracking and not uncertain.move
    assert uncertain.reason == "pointing_confidence_too_low"

    mostly_vertical = planner.update(make_pointing_frame((0.1, 0.995)))
    assert mostly_vertical.tracking and mostly_vertical.move
    assert mostly_vertical.reason == "move_vertical"
    assert mostly_vertical.pan_delta_deg == 0.0
    assert mostly_vertical.vertical_delta_deg == 3.0


def test_pointing_bridge_waits_for_stable_direction_and_limits_step(monkeypatch) -> None:
    sent = []
    clock = [10.0]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        sent.append((json.loads(request.data), timeout))
        return Response()

    monkeypatch.setattr("jarvis.perception.servo.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    planner = PointingGazePlanner(
        PointingGazeConfig(
            smoothing=1.0,
            deadband=0.1,
            pan_gain_deg=12.0,
            max_step_deg=3.0,
        )
    )
    recommendation = planner.update(make_pointing_frame((1.0, 0.0)))
    bridge = HttpPointingServoBridge(
        rate_hz=1.0,
        max_step_deg=2.0,
        settle_s=0.5,
    )

    settling = bridge.maybe_send(recommendation)
    assert settling == {"sent": False, "reason": "pointing_settling"}
    assert not sent

    clock[0] = 10.6
    first = bridge.maybe_send(recommendation)
    assert first is not None and first["sent"]
    assert first["deltas"] == {"shoulder_pan": -2.0}
    assert sent[0][0] == {"deltas": {"shoulder_pan": -2.0}}

    clock[0] = 10.7
    limited = bridge.maybe_send(recommendation)
    assert limited == {"sent": False, "reason": "horizontal_rate_limited"}
    assert len(sent) == 1


def test_pointing_bridge_latches_actuator_failure_until_pointing_released(
    monkeypatch,
) -> None:
    sent = []
    clock = [20.0]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"ok": false, "message": "no motion"}'

    def fake_urlopen(request, timeout):
        sent.append((request, timeout))
        return Response()

    monkeypatch.setattr("jarvis.perception.servo.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    planner = PointingGazePlanner(
        PointingGazeConfig(smoothing=1.0, deadband=0.1, pan_gain_deg=12.0)
    )
    recommendation = planner.update(make_pointing_frame((1.0, 0.0)))
    bridge = HttpPointingServoBridge(rate_hz=1.0, settle_s=0.0)

    settling = bridge.maybe_send(recommendation)
    assert settling == {"sent": False, "reason": "pointing_settling"}
    clock[0] = 20.1
    failed = bridge.maybe_send(recommendation)
    assert failed is not None and failed["reason"] == "actuator_no_response"
    clock[0] = 22.0
    latched = bridge.maybe_send(recommendation)
    assert latched is not None and latched["reason"] == "control_latched"
    assert len(sent) == 1

    released = planner.update(make_frame(None, timestamp=3.0))
    bridge.maybe_send(released)
    assert bridge._blocked_reason is None


def test_pointing_bridge_uses_minimum_effective_vertical_pulse(monkeypatch) -> None:
    sent = []
    clock = [30.0]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        sent.append((json.loads(request.data), timeout))
        return Response()

    monkeypatch.setattr("jarvis.perception.servo.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    planner = PointingGazePlanner(
        PointingGazeConfig(
            smoothing=1.0,
            deadband=0.18,
            pan_gain_deg=0.0,
            vertical_gain_deg=2.0,
            max_step_deg=6.0,
        )
    )
    recommendation = planner.update(make_pointing_frame((0.1, 0.995)))
    bridge = HttpPointingServoBridge(
        vertical_max_step_deg=6.0,
        vertical_min_step_deg=6.0,
        max_steps_per_gesture=1,
        enable_vertical=True,
        settle_s=0.0,
    )

    settling = bridge.maybe_send(recommendation)
    assert settling == {"sent": False, "reason": "pointing_settling"}
    clock[0] = 30.1
    result = bridge.maybe_send(recommendation)
    assert result is not None and result["sent"]
    assert result["deltas"] == {"wrist_flex": 6.0}
    assert result["gesture_limit_reached"]
    assert sent[0][0] == {"deltas": {"wrist_flex": 6.0}}

    clock[0] = 30.2
    limited = bridge.maybe_send(recommendation)
    assert limited == {"sent": False, "reason": "gesture_step_limit"}
    assert len(sent) == 1


def test_pointed_object_namer_records_vision_reply(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return '{"reply": "这是一个灰色杯子。"}'.encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    namer = PointedObjectNamer(delay_s=0.0, timeout_s=1.0)

    namer.schedule()
    assert namer.snapshot()["status"] == "scheduled"
    namer.maybe_start()
    for _ in range(20):
        if namer.snapshot()["status"] == "done":
            break
        time.sleep(0.01)
    assert namer.snapshot() == {
        "status": "done",
        "reply": "这是一个灰色杯子。",
        "error": None,
    }
    namer.close()

