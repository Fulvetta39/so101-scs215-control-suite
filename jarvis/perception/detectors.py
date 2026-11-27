from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .geometry import is_plausible_face, pointing_geometry
from .types import BoundingBox, FaceCandidate, HandObservation, UvPoint


class MediaPipeDetector:
    """MediaPipe 1.x Tasks implementation for face boxes and hand landmarks."""

    def __init__(
        self,
        models_dir: str | Path | None = None,
        *,
        face_confidence: float = 0.5,
        hand_confidence: float = 0.5,
        max_hands: int = 2,
    ):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Use .venv_perception or install "
                "the contents of requirements-perception.txt."
            ) from exc

        root = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        face_model = root / "blaze_face_short_range.tflite"
        hand_model = root / "hand_landmarker.task"
        if not face_model.is_file():
            raise FileNotFoundError(f"Face model not found: {face_model}")
        if not hand_model.is_file():
            raise FileNotFoundError(f"Hand model not found: {hand_model}")
        face_asset = _model_asset_path(face_model)
        hand_asset = _model_asset_path(hand_model)

        self._mp = mp
        self.face_confidence = face_confidence
        self._face = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=face_asset),
                running_mode=VisionTaskRunningMode.VIDEO,
                min_detection_confidence=face_confidence,
            )
        )
        self._hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_asset),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=hand_confidence,
                min_hand_presence_confidence=hand_confidence,
                min_tracking_confidence=hand_confidence,
            )
        )
        self._last_timestamp_ms = -1

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> tuple[tuple[FaceCandidate, ...], tuple[HandObservation, ...]]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape HxWx3")
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        faces: list[FaceCandidate] = []
        face_result = self._face.detect_for_video(image, timestamp_ms)
        for detection in face_result.detections:
            box = detection.bounding_box
            bbox = BoundingBox(
                left=float(box.origin_x),
                top=float(box.origin_y),
                width=float(box.width),
                height=float(box.height),
            )
            cx, cy = bbox.center
            center_uv = (cx / width, cy / height)
            confidence = (
                float(detection.categories[0].score)
                if detection.categories and detection.categories[0].score is not None
                else 0.0
            )
            keypoints = tuple(
                (float(keypoint.x), float(keypoint.y))
                for keypoint in (detection.keypoints or ())
                if keypoint.x is not None and keypoint.y is not None
            )
            face = FaceCandidate(
                bbox_px=bbox,
                center_uv=center_uv,
                center_error=(2.0 * center_uv[0] - 1.0, 2.0 * center_uv[1] - 1.0),
                confidence=confidence,
                keypoints_uv=keypoints,
            )
            if is_plausible_face(face, min_confidence=self.face_confidence):
                faces.append(face)

        hands: list[HandObservation] = []
        hand_result = self._hand.detect_for_video(image, timestamp_ms)
        for index, landmarks in enumerate(hand_result.hand_landmarks):
            points = tuple(
                UvPoint(
                    x=float(point.x),
                    y=float(point.y),
                    z=float(point.z or 0.0),
                    visibility=point.visibility,
                )
                for point in landmarks
            )
            world = hand_result.hand_world_landmarks[index] if hand_result.hand_world_landmarks else ()
            world_points = tuple(
                UvPoint(x=float(point.x), y=float(point.y), z=float(point.z or 0.0))
                for point in world
            )
            category = (
                hand_result.handedness[index][0]
                if hand_result.handedness and hand_result.handedness[index]
                else None
            )
            handedness = (
                category.category_name if category and category.category_name else "unknown"
            )
            confidence = (
                float(category.score)
                if category and category.score is not None
                else 0.0
            )
            is_pointing, score, tip, direction, _ = pointing_geometry(points)
            hands.append(
                HandObservation(
                    handedness=handedness,
                    confidence=confidence,
                    landmarks_uv=points,
                    world_landmarks=world_points,
                    is_pointing=is_pointing,
                    pointing_score=score,
                    index_tip_uv=tip,
                    index_direction_uv=direction,
                )
            )
        return tuple(faces), tuple(hands)

    def close(self) -> None:
        self._face.close()
        self._hand.close()

    def __enter__(self) -> "MediaPipeDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _model_asset_path(path: Path) -> str:
    """Use an ASCII relative path when models live below the current directory.

    MediaPipe's native model loader on Windows does not reliably handle Unicode
    absolute paths, so prefer a relative asset path in that case.
    """
    relative = os.path.relpath(path)
    if all(ord(character) < 128 for character in relative):
        return relative
    if all(ord(character) < 128 for character in str(path)):
        return str(path)
    cache = Path(tempfile.gettempdir()) / "jarvis-mediapipe-models"
    cache.mkdir(parents=True, exist_ok=True)
    copied = cache / path.name
    if not copied.is_file() or copied.stat().st_size != path.stat().st_size:
        shutil.copy2(path, copied)
    return str(copied)

