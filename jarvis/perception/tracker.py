from __future__ import annotations

import math
from collections.abc import Sequence

from .geometry import ray_to_border
from .types import BoundingBox, FaceCandidate, HandObservation, PointingRay, TrackedFace


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def select_primary_face(
    faces: Sequence[FaceCandidate],
    previous_center: tuple[float, float] | None = None,
) -> FaceCandidate | None:
    """Pick the most plausible current conversation partner.

    Score favors confidence, faces near the image center, and continuity with the
    previously tracked face. This is intentionally deterministic and local.
    """
    if not faces:
        return None
    best: FaceCandidate | None = None
    best_score = float("-inf")
    for face in faces:
        cx, cy = face.center_uv
        center_penalty = math.hypot(cx - 0.5, cy - 0.5)
        continuity_bonus = (
            0.55 * max(0.0, 1.0 - 2.0 * _distance(face.center_uv, previous_center))
            if previous_center is not None
            else 0.0
        )
        score = face.confidence + continuity_bonus - 0.35 * center_penalty
        if score > best_score:
            best = face
            best_score = score
    return best


class FaceTracker:
    """Small exponential smoother with explicit short-loss hold."""

    def __init__(
        self,
        alpha: float = 0.38,
        max_missed_frames: int = 12,
        acquire_frames: int = 2,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if acquire_frames < 1:
            raise ValueError("acquire_frames must be at least 1")
        self.alpha = alpha
        self.max_missed_frames = max_missed_frames
        self.acquire_frames = acquire_frames
        self._tracked: TrackedFace | None = None
        self._missed = 0
        self._candidate_center: tuple[float, float] | None = None
        self._candidate_hits = 0

    @property
    def current(self) -> TrackedFace | None:
        return self._tracked

    def update(self, faces: Sequence[FaceCandidate]) -> TrackedFace | None:
        selected = select_primary_face(
            faces,
            self._tracked.center_uv if self._tracked else None,
        )
        if selected is None:
            self._candidate_center = None
            self._candidate_hits = 0
            return self._miss()

        if self._tracked is None:
            if (
                self._candidate_center is not None
                and _distance(selected.center_uv, self._candidate_center) <= 0.16
            ):
                self._candidate_hits += 1
            else:
                self._candidate_hits = 1
            self._candidate_center = selected.center_uv
            if self._candidate_hits < self.acquire_frames:
                return None
            self._tracked = TrackedFace(
                bbox_px=selected.bbox_px,
                center_uv=selected.center_uv,
                center_error=selected.center_error,
                confidence=selected.confidence,
            )
            self._missed = 0
            self._candidate_center = None
            self._candidate_hits = 0
            return self._tracked

        alpha = self.alpha
        old = self._tracked
        old_bbox = old.bbox_px
        new_bbox = selected.bbox_px
        bbox = BoundingBox(
            left=old_bbox.left + alpha * (new_bbox.left - old_bbox.left),
            top=old_bbox.top + alpha * (new_bbox.top - old_bbox.top),
            width=old_bbox.width + alpha * (new_bbox.width - old_bbox.width),
            height=old_bbox.height + alpha * (new_bbox.height - old_bbox.height),
        )
        center = (
            old.center_uv[0] + alpha * (selected.center_uv[0] - old.center_uv[0]),
            old.center_uv[1] + alpha * (selected.center_uv[1] - old.center_uv[1]),
        )
        self._tracked = TrackedFace(
            bbox_px=bbox,
            center_uv=center,
            center_error=(2.0 * center[0] - 1.0, 2.0 * center[1] - 1.0),
            confidence=selected.confidence,
        )
        self._missed = 0
        return self._tracked

    def _miss(self) -> TrackedFace | None:
        self._missed += 1
        if self._tracked is None or self._missed > self.max_missed_frames:
            self._tracked = None
            return None
        self._tracked = TrackedFace(
            bbox_px=self._tracked.bbox_px,
            center_uv=self._tracked.center_uv,
            center_error=self._tracked.center_error,
            confidence=self._tracked.confidence,
            missed_frames=self._missed,
        )
        return self._tracked

    def reset(self) -> None:
        self._tracked = None
        self._missed = 0
        self._candidate_center = None
        self._candidate_hits = 0


class PointingTracker:
    """Smooth pointing directions and bridge brief landmark dropouts."""

    def __init__(self, alpha: float = 0.38, release_grace_s: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if release_grace_s < 0:
            raise ValueError("release_grace_s must be non-negative")
        self.alpha = alpha
        self.release_grace_s = release_grace_s
        self._ray: PointingRay | None = None
        self._hand: HandObservation | None = None
        self._last_seen_s: float | None = None

    def update(
        self,
        ray: PointingRay | None,
        hand: HandObservation | None,
        timestamp_s: float,
    ) -> tuple[PointingRay | None, HandObservation | None]:
        if ray is not None and hand is not None:
            if self._ray is None:
                direction = ray.direction_uv
                origin = ray.origin_uv
            else:
                alpha = self.alpha
                direction = (
                    alpha * ray.direction_uv[0]
                    + (1.0 - alpha) * self._ray.direction_uv[0],
                    alpha * ray.direction_uv[1]
                    + (1.0 - alpha) * self._ray.direction_uv[1],
                )
                origin = (
                    alpha * ray.origin_uv[0]
                    + (1.0 - alpha) * self._ray.origin_uv[0],
                    alpha * ray.origin_uv[1]
                    + (1.0 - alpha) * self._ray.origin_uv[1],
                )
            length = math.hypot(*direction)
            if length >= 1e-9:
                direction = (direction[0] / length, direction[1] / length)
            else:
                direction = self._ray.direction_uv if self._ray else ray.direction_uv
            self._ray = PointingRay(
                origin_uv=origin,
                direction_uv=direction,
                end_uv=ray_to_border(origin, direction),
                confidence=(
                    ray.confidence
                    if self._ray is None
                    else alpha * ray.confidence + (1.0 - alpha) * self._ray.confidence
                ),
            )
            self._hand = hand
            self._last_seen_s = timestamp_s
            return self._ray, self._hand

        if (
            self._ray is not None
            and self._last_seen_s is not None
            and timestamp_s - self._last_seen_s <= self.release_grace_s
        ):
            return self._ray, self._hand
        self.reset()
        return None, None

    def reset(self) -> None:
        self._ray = None
        self._hand = None
        self._last_seen_s = None

