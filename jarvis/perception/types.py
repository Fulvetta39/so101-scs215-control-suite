from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: float
    top: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True, slots=True)
class UvPoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float | None = None


@dataclass(frozen=True, slots=True)
class FaceCandidate:
    bbox_px: BoundingBox
    center_uv: tuple[float, float]
    center_error: tuple[float, float]
    confidence: float
    keypoints_uv: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class TrackedFace:
    bbox_px: BoundingBox
    center_uv: tuple[float, float]
    center_error: tuple[float, float]
    confidence: float
    missed_frames: int = 0


@dataclass(frozen=True, slots=True)
class PointingRay:
    origin_uv: tuple[float, float]
    direction_uv: tuple[float, float]
    end_uv: tuple[float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class HandObservation:
    handedness: str
    confidence: float
    landmarks_uv: tuple[UvPoint, ...]
    world_landmarks: tuple[UvPoint, ...] = ()
    is_pointing: bool = False
    pointing_score: float = 0.0
    index_tip_uv: tuple[float, float] | None = None
    index_direction_uv: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    timestamp_s: float
    width: int
    height: int
    faces: tuple[FaceCandidate, ...] = ()
    hands: tuple[HandObservation, ...] = ()
    primary_face: TrackedFace | None = None
    pointing: PointingRay | None = None
    attention: str = "searching"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": _rounded(self.timestamp_s, 3),
            "width": self.width,
            "height": self.height,
            "attention": self.attention,
            "face_count": len(self.faces),
            "hand_count": len(self.hands),
            "primary_face": _face_payload(self.primary_face),
            "faces": [_face_payload(face) for face in self.faces],
            "hands": [_hand_payload(hand) for hand in self.hands],
            "pointing": _pointing_payload(self.pointing),
            "warnings": list(self.warnings),
        }


def _face_payload(face: FaceCandidate | TrackedFace | None) -> dict[str, Any] | None:
    if face is None:
        return None
    if isinstance(face, TrackedFace):
        return {
            "bbox_px": asdict(face.bbox_px),
            "center_uv": [_rounded(value) for value in face.center_uv],
            "center_error": [_rounded(value) for value in face.center_error],
            "confidence": _rounded(face.confidence),
            "missed_frames": face.missed_frames,
        }
    return {
        "bbox_px": asdict(face.bbox_px),
        "center_uv": [_rounded(value) for value in face.center_uv],
        "center_error": [_rounded(value) for value in face.center_error],
        "confidence": _rounded(face.confidence),
        "keypoints_uv": [
            [_rounded(px), _rounded(py)] for px, py in face.keypoints_uv
        ],
    }


def _hand_payload(hand: HandObservation) -> dict[str, Any]:
    return {
        "handedness": hand.handedness,
        "confidence": _rounded(hand.confidence),
        "is_pointing": hand.is_pointing,
        "pointing_score": _rounded(hand.pointing_score),
        "index_tip_uv": (
            [_rounded(value) for value in hand.index_tip_uv]
            if hand.index_tip_uv
            else None
        ),
        "index_direction_uv": (
            [_rounded(value) for value in hand.index_direction_uv]
            if hand.index_direction_uv
            else None
        ),
        "landmarks_uv": [
            [_rounded(point.x), _rounded(point.y)] for point in hand.landmarks_uv
        ],
        "world_landmarks_m": [
            [_rounded(point.x), _rounded(point.y), _rounded(point.z)]
            for point in hand.world_landmarks
        ],
    }


def _pointing_payload(ray: PointingRay | None) -> dict[str, Any] | None:
    if ray is None:
        return None
    return {
        "origin_uv": [_rounded(value) for value in ray.origin_uv],
        "direction_uv": [_rounded(value) for value in ray.direction_uv],
        "end_uv": [_rounded(value) for value in ray.end_uv],
        "confidence": _rounded(ray.confidence),
    }

