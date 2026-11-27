from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .detectors import MediaPipeDetector
from .geometry import pointing_geometry
from .tracker import FaceTracker, PointingTracker
from .types import HandObservation, PerceptionFrame, PointingRay

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


class PerceptionPipeline:
    def __init__(
        self,
        models_dir=None,
        *,
        face_confidence: float = 0.5,
        hand_confidence: float = 0.5,
        mirror: bool = True,
        max_hands: int = 2,
        tracker_alpha: float = 0.38,
        max_missed_frames: int = 12,
        detector: MediaPipeDetector | None = None,
    ):
        self.mirror = mirror
        self.detector = detector or MediaPipeDetector(
            models_dir,
            face_confidence=face_confidence,
            hand_confidence=hand_confidence,
            max_hands=max_hands,
        )
        self.face_tracker = FaceTracker(tracker_alpha, max_missed_frames)
        self.pointing_tracker = PointingTracker()

    def process(self, frame_bgr: np.ndarray, timestamp_s: float) -> PerceptionFrame:
        if self.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        height, width = frame_bgr.shape[:2]
        faces, hands = self.detector.detect(frame_bgr, int(timestamp_s * 1000.0))
        primary_face = self.face_tracker.update(faces)

        raw_pointing, raw_pointing_hand = _best_pointing(hands)
        pointing, pointing_hand = self.pointing_tracker.update(
            raw_pointing,
            raw_pointing_hand,
            timestamp_s,
        )
        warnings: list[str] = []
        if not faces and pointing is None:
            warnings.append("no_face_or_pointing_hand")
        return PerceptionFrame(
            timestamp_s=timestamp_s,
            width=width,
            height=height,
            faces=faces,
            hands=hands,
            primary_face=primary_face,
            pointing=pointing,
            attention="pointing" if pointing else "face" if primary_face else "searching",
            warnings=tuple(warnings),
        )

    def annotate(self, frame_bgr: np.ndarray, result: PerceptionFrame) -> np.ndarray:
        output = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr.copy()
        height, width = output.shape[:2]

        for face in result.faces:
            box = face.bbox_px
            p1 = (int(box.left), int(box.top))
            p2 = (int(box.left + box.width), int(box.top + box.height))
            cv2.rectangle(output, p1, p2, (80, 220, 120), 2)
            cv2.putText(
                output,
                f"face {face.confidence:.2f}",
                (p1[0], max(18, p1[1] - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 220, 120),
                1,
                cv2.LINE_AA,
            )

        for hand in result.hands:
            points = [
                (int(point.x * width), int(point.y * height))
                for point in hand.landmarks_uv
            ]
            if len(points) != 21:
                continue
            color = (0, 190, 255) if hand.is_pointing else (180, 140, 255)
            for start, end in HAND_CONNECTIONS:
                cv2.line(output, points[start], points[end], color, 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(output, point, 3, color, -1, cv2.LINE_AA)

        if result.pointing:
            start = _uv_to_pixel(result.pointing.origin_uv, width, height)
            end = _uv_to_pixel(result.pointing.end_uv, width, height)
            cv2.arrowedLine(output, start, end, (40, 245, 170), 3, cv2.LINE_AA, tipLength=0.06)
            cv2.putText(
                output,
                f"pointing {result.pointing.confidence:.2f}",
                (start[0] + 8, start[1] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 245, 170),
                2,
                cv2.LINE_AA,
            )

        if result.primary_face:
            error = result.primary_face.center_error
            cv2.putText(
                output,
                f"attention={result.attention}  err=({error[0]:+.2f}, {error[1]:+.2f})",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
        return output

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "PerceptionPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _best_pointing(
    hands: Sequence[HandObservation],
) -> tuple[PointingRay | None, HandObservation | None]:
    best_hand: HandObservation | None = None
    best_score = -1.0
    for hand in hands:
        if (
            hand.is_pointing
            and hand.index_tip_uv is not None
            and hand.index_direction_uv is not None
            and hand.pointing_score > best_score
        ):
            best_hand = hand
            best_score = hand.pointing_score
    if best_hand is None:
        return None, None
    _, score, tip, direction, ray = pointing_geometry(best_hand.landmarks_uv)
    if not ray or tip is None or direction is None:
        return None, None
    return ray, best_hand


def _uv_to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return (int(point[0] * width), int(point[1] * height))

