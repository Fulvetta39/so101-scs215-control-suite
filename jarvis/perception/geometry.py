from __future__ import annotations

import math
from collections.abc import Sequence

from .types import FaceCandidate, PointingRay, UvPoint

# MediaPipe hand landmark indices.
WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20


def _distance(a: UvPoint, b: UvPoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def ray_to_border(
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> tuple[float, float]:
    """Extend a normalized-image ray until it leaves the [0, 1] image."""
    ox, oy = origin
    dx, dy = _unit(*direction)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return (ox, oy)

    candidates: list[float] = []
    if dx > 0:
        candidates.append((1.0 - ox) / dx)
    elif dx < 0:
        candidates.append((0.0 - ox) / dx)
    if dy > 0:
        candidates.append((1.0 - oy) / dy)
    elif dy < 0:
        candidates.append((0.0 - oy) / dy)
    positive = [value for value in candidates if value >= 0.0]
    scale = min(positive) if positive else 0.0
    ex = max(0.0, min(1.0, ox + dx * scale))
    ey = max(0.0, min(1.0, oy + dy * scale))
    # Snap numerical residue so downstream thresholds see exact image borders.
    ex = 0.0 if abs(ex) < 1e-9 else 1.0 if abs(ex - 1.0) < 1e-9 else ex
    ey = 0.0 if abs(ey) < 1e-9 else 1.0 if abs(ey - 1.0) < 1e-9 else ey
    end = (ex, ey)
    return end


def is_plausible_face(
    face: FaceCandidate,
    *,
    min_confidence: float = 0.5,
) -> bool:
    """Reject small or geometrically inconsistent face-detector false positives."""
    if face.confidence < min_confidence:
        return False
    if face.bbox_px.width < 24 or face.bbox_px.height < 24:
        return False
    aspect = face.bbox_px.width / max(1.0, face.bbox_px.height)
    if not 0.35 <= aspect <= 2.2:
        return False
    if len(face.keypoints_uv) < 4:
        return False

    right_eye, left_eye, nose, mouth = face.keypoints_uv[:4]
    eye_distance = math.hypot(
        right_eye[0] - left_eye[0],
        right_eye[1] - left_eye[1],
    )
    if eye_distance < 0.025:
        return False
    eye_y = (right_eye[1] + left_eye[1]) / 2.0
    if abs(right_eye[1] - left_eye[1]) > 0.12:
        return False
    if not eye_y - 0.08 <= nose[1] <= eye_y + 0.18:
        return False
    if mouth[1] < eye_y + 0.015:
        return False
    return True


def pointing_geometry(
    landmarks: Sequence[UvPoint],
    *,
    pointing_threshold: float = 0.55,
) -> tuple[bool, float, tuple[float, float] | None, tuple[float, float] | None, PointingRay | None]:
    """Estimate whether the index finger is deliberately pointing.

    Returns:
        is_pointing, score, index_tip_uv, index_direction_uv, screen_ray
    """
    if len(landmarks) < 21:
        return False, 0.0, None, None, None

    palm_scale = max(
        1e-6,
        _distance(landmarks[WRIST], landmarks[MIDDLE_MCP]),
        _distance(landmarks[WRIST], landmarks[INDEX_MCP]),
    )

    def straightness(mcp: int, pip: int, dip: int, tip: int) -> float:
        path = (
            _distance(landmarks[mcp], landmarks[pip])
            + _distance(landmarks[pip], landmarks[dip])
            + _distance(landmarks[dip], landmarks[tip])
        )
        return _distance(landmarks[mcp], landmarks[tip]) / max(path, 1e-9)

    index_ext = _distance(landmarks[INDEX_MCP], landmarks[INDEX_TIP]) / palm_scale
    index_straight = straightness(INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    other_straight = (
        straightness(MIDDLE_MCP, MIDDLE_MCP + 1, MIDDLE_MCP + 2, MIDDLE_TIP),
        straightness(RING_MCP, RING_MCP + 1, RING_MCP + 2, RING_TIP),
        straightness(PINKY_MCP, PINKY_MCP + 1, PINKY_MCP + 2, PINKY_TIP),
    )
    folded_count = sum(value < 0.86 for value in other_straight)
    direction = _unit(
        landmarks[INDEX_TIP].x - landmarks[INDEX_MCP].x,
        landmarks[INDEX_TIP].y - landmarks[INDEX_MCP].y,
    )

    # A straight index finger and a folded middle finger is the primary cue.
    # At oblique camera angles the ring or pinky may look partially extended,
    # so requiring all three non-index fingers to be fully folded is too strict.
    index_score = min(1.0, max(0.0, (index_straight - 0.68) / 0.27))
    extension_score = min(1.0, max(0.0, (index_ext - 0.55) / 0.55))
    fold_score = folded_count / 3.0
    score = 0.45 * index_score + 0.25 * extension_score + 0.30 * fold_score
    is_pointing = (
        score >= pointing_threshold
        and index_straight >= 0.78
        and index_ext >= 0.80
        and folded_count >= 1
        and other_straight[0] < 0.90
        and direction != (0.0, 0.0)
    )
    tip = (landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y)
    ray = None
    if is_pointing:
        ray = PointingRay(
            origin_uv=tip,
            direction_uv=direction,
            end_uv=ray_to_border(tip, direction),
            confidence=score,
        )
    return is_pointing, score, tip, direction, ray

