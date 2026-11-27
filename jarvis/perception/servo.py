from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from .types import PerceptionFrame


@dataclass(frozen=True, slots=True)
class FaceServoConfig:
    pan_gain_deg: float = 10.0
    vertical_gain_deg: float = 8.0
    max_step_deg: float = 2.0
    vertical_max_step_deg: float | None = None
    vertical_min_step_deg: float = 0.0
    deadband: float = 0.08
    vertical_deadband: float | None = None
    smoothing: float = 0.35
    pan_sign: int = -1
    vertical_sign: int = 1
    vertical_joint: str = "wrist_flex"
    min_face_confidence: float = 0.45
    lost_hold_s: float = 0.25
    search_hold_s: float = 0.35
    search_out_s: float = 4.0
    search_back_s: float = 4.0
    search_reverse_pause_s: float = 0.5
    search_step_deg: float = 3.0
    search_interval_s: float = 0.25
    pause_on_pointing: bool = True


@dataclass(frozen=True, slots=True)
class PointingGazeConfig:
    pan_gain_deg: float = 12.0
    vertical_gain_deg: float = 8.0
    max_step_deg: float = 3.0
    deadband: float = 0.18
    smoothing: float = 0.30
    pan_sign: int = -1
    vertical_sign: int = 1
    vertical_joint: str = "wrist_flex"
    min_confidence: float = 0.62


@dataclass(frozen=True, slots=True)
class ServoRecommendation:
    timestamp_s: float
    tracking: bool
    move: bool
    reason: str
    raw_error: tuple[float, float] | None
    filtered_error: tuple[float, float] | None
    pan_delta_deg: float
    vertical_delta_deg: float
    joint_deltas: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "timestamp_s": round(self.timestamp_s, 3),
            "tracking": self.tracking,
            "move": self.move,
            "reason": self.reason,
            "raw_error": (
                [round(self.raw_error[0], 4), round(self.raw_error[1], 4)]
                if self.raw_error
                else None
            ),
            "filtered_error": (
                [round(self.filtered_error[0], 4), round(self.filtered_error[1], 4)]
                if self.filtered_error
                else None
            ),
            "pan_delta_deg": round(self.pan_delta_deg, 3),
            "vertical_delta_deg": round(self.vertical_delta_deg, 3),
            "joint_deltas": {
                name: round(value, 3) for name, value in self.joint_deltas.items()
            },
        }


@dataclass(frozen=True, slots=True)
class PointingGazeRecommendation:
    timestamp_s: float
    tracking: bool
    move: bool
    reason: str
    confidence: float
    raw_direction: tuple[float, float] | None
    filtered_direction: tuple[float, float] | None
    pan_delta_deg: float
    vertical_delta_deg: float
    joint_deltas: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "timestamp_s": round(self.timestamp_s, 3),
            "tracking": self.tracking,
            "move": self.move,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "raw_direction": (
                [round(self.raw_direction[0], 4), round(self.raw_direction[1], 4)]
                if self.raw_direction
                else None
            ),
            "filtered_direction": (
                [
                    round(self.filtered_direction[0], 4),
                    round(self.filtered_direction[1], 4),
                ]
                if self.filtered_direction
                else None
            ),
            "pan_delta_deg": round(self.pan_delta_deg, 3),
            "vertical_delta_deg": round(self.vertical_delta_deg, 3),
            "joint_deltas": {
                name: round(value, 3) for name, value in self.joint_deltas.items()
            },
        }


class FaceServoPlanner:
    """Convert face position error into conservative shadow-mode corrections."""

    def __init__(self, config: FaceServoConfig | None = None):
        self.config = config or FaceServoConfig()
        if not 0.0 <= self.config.deadband < 1.0:
            raise ValueError("deadband must be in [0, 1)")
        if not 0.0 < self.config.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if self.config.pan_sign not in (-1, 1) or self.config.vertical_sign not in (-1, 1):
            raise ValueError("axis signs must be -1 or 1")
        vertical_max = (
            self.config.max_step_deg
            if self.config.vertical_max_step_deg is None
            else self.config.vertical_max_step_deg
        )
        if self.config.vertical_min_step_deg < 0 or self.config.vertical_min_step_deg > vertical_max:
            raise ValueError("vertical_min_step_deg must be between 0 and vertical maximum")
        if min(
            self.config.search_hold_s,
            self.config.search_out_s,
            self.config.search_back_s,
            self.config.search_reverse_pause_s,
            self.config.search_step_deg,
            self.config.search_interval_s,
        ) < 0:
            raise ValueError("face search limits must be non-negative")
        self._filtered: tuple[float, float] | None = None
        self._last_seen_s: float | None = None
        self._last_face_error: tuple[float, float] | None = None
        self._search_since_s: float | None = None
        self._last_search_command_s: float | None = None
        self._search_direction = 1

    def update(self, frame: PerceptionFrame) -> ServoRecommendation:
        face = frame.primary_face
        if self.config.pause_on_pointing and frame.pointing is not None:
            self._filtered = None
            self._reset_search()
            return self._stop(frame.timestamp_s, "pointing_has_priority")
        if face is None:
            self._filtered = None
            self._last_seen_s = None
            return self._search(frame.timestamp_s)
        if face.missed_frames:
            self._filtered = None
            self._last_face_error = face.center_error
            return self._search(frame.timestamp_s)
        if face.confidence < self.config.min_face_confidence:
            self._filtered = None
            self._reset_search()
            return self._stop(frame.timestamp_s, "face_confidence_too_low")

        raw = face.center_error
        self._last_face_error = raw
        self._reset_search()
        if self._filtered is None:
            self._filtered = raw
        else:
            alpha = self.config.smoothing
            self._filtered = (
                alpha * raw[0] + (1.0 - alpha) * self._filtered[0],
                alpha * raw[1] + (1.0 - alpha) * self._filtered[1],
            )
        self._last_seen_s = frame.timestamp_s

        x = _outside_deadband(self._filtered[0], self.config.deadband)
        vertical_deadband = (
            self.config.deadband
            if self.config.vertical_deadband is None
            else self.config.vertical_deadband
        )
        y = _outside_deadband(self._filtered[1], vertical_deadband)
        pan = _clamp(
            self.config.pan_sign * self.config.pan_gain_deg * x,
            self.config.max_step_deg,
        )
        vertical_max = (
            self.config.max_step_deg
            if self.config.vertical_max_step_deg is None
            else self.config.vertical_max_step_deg
        )
        vertical = _clamp(
            self.config.vertical_sign * self.config.vertical_gain_deg * y,
            vertical_max,
        )
        if y and abs(vertical) < self.config.vertical_min_step_deg:
            vertical = math.copysign(self.config.vertical_min_step_deg, vertical)
        move = abs(pan) > 1e-4 or abs(vertical) > 1e-4
        reason = "move_to_center" if move else "inside_deadband"
        joint_deltas: dict[str, float] = {}
        if pan:
            joint_deltas["shoulder_pan"] = pan
        if vertical:
            joint_deltas[self.config.vertical_joint] = vertical
        return ServoRecommendation(
            timestamp_s=frame.timestamp_s,
            tracking=True,
            move=move,
            reason=reason,
            raw_error=raw,
            filtered_error=self._filtered,
            pan_delta_deg=pan,
            vertical_delta_deg=vertical,
            joint_deltas=joint_deltas,
        )

    def _search(self, timestamp_s: float) -> ServoRecommendation:
        if self._last_face_error is None:
            return self._stop(timestamp_s, "face_not_found")
        if self._search_since_s is None:
            self._search_since_s = timestamp_s
            self._search_direction = 1 if self._last_face_error[0] >= 0 else -1
        elapsed = timestamp_s - self._search_since_s
        if elapsed < self.config.search_hold_s:
            return self._stop(timestamp_s, "face_search_hold")
        reverse_start = (
            self.config.search_out_s + self.config.search_reverse_pause_s
        )
        if elapsed > reverse_start + self.config.search_back_s:
            return self._stop(timestamp_s, "face_search_timeout")
        if (
            self._last_search_command_s is not None
            and timestamp_s - self._last_search_command_s < self.config.search_interval_s
        ):
            return self._stop(timestamp_s, "face_search_wait")

        direction = self._search_direction
        reason = "search_last_direction"
        if self.config.search_out_s <= elapsed < reverse_start:
            return self._stop(timestamp_s, "face_search_reverse_hold")
        if elapsed >= reverse_start:
            direction = -direction
            reason = "search_return_sweep"
        pan = self.config.pan_sign * direction * self.config.search_step_deg
        self._last_search_command_s = timestamp_s
        return ServoRecommendation(
            timestamp_s=timestamp_s,
            tracking=True,
            move=True,
            reason=reason,
            raw_error=self._last_face_error,
            filtered_error=None,
            pan_delta_deg=pan,
            vertical_delta_deg=0.0,
            joint_deltas={"shoulder_pan": pan},
        )

    def _reset_search(self) -> None:
        self._search_since_s = None
        self._last_search_command_s = None

    def _stop(self, timestamp_s: float, reason: str) -> ServoRecommendation:
        return ServoRecommendation(
            timestamp_s=timestamp_s,
            tracking=False,
            move=False,
            reason=reason,
            raw_error=None,
            filtered_error=None,
            pan_delta_deg=0.0,
            vertical_delta_deg=0.0,
            joint_deltas={},
        )


class PointingGazePlanner:
    """Map a pointing direction to a conservative shadow-mode camera turn."""

    def __init__(self, config: PointingGazeConfig | None = None):
        self.config = config or PointingGazeConfig()
        if not 0.0 <= self.config.deadband < 1.0:
            raise ValueError("deadband must be in [0, 1)")
        if not 0.0 < self.config.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if self.config.max_step_deg <= 0:
            raise ValueError("max_step_deg must be positive")
        if not 0.0 <= self.config.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.config.pan_sign not in (-1, 1) or self.config.vertical_sign not in (-1, 1):
            raise ValueError("axis signs must be -1 or 1")
        self._filtered: tuple[float, float] | None = None

    def update(self, frame: PerceptionFrame) -> PointingGazeRecommendation:
        ray = frame.pointing
        if ray is None:
            self._filtered = None
            return self._stop(frame.timestamp_s, "pointing_not_found", 0.0)
        if ray.confidence < self.config.min_confidence:
            self._filtered = None
            return self._stop(
                frame.timestamp_s,
                "pointing_confidence_too_low",
                ray.confidence,
            )

        raw = _unit(ray.direction_uv[0], ray.direction_uv[1])
        if raw == (0.0, 0.0):
            self._filtered = None
            return self._stop(frame.timestamp_s, "invalid_pointing_direction", ray.confidence)
        if self._filtered is None:
            self._filtered = raw
        else:
            alpha = self.config.smoothing
            self._filtered = _unit(
                alpha * raw[0] + (1.0 - alpha) * self._filtered[0],
                alpha * raw[1] + (1.0 - alpha) * self._filtered[1],
            )

        x = _outside_deadband(self._filtered[0], self.config.deadband)
        y = _outside_deadband(self._filtered[1], self.config.deadband)
        pan = _clamp(
            self.config.pan_sign * self.config.pan_gain_deg * x,
            self.config.max_step_deg,
        )
        vertical = _clamp(
            self.config.vertical_sign * self.config.vertical_gain_deg * y,
            self.config.max_step_deg,
        )
        move_pan = abs(pan) > 1e-4
        move_vertical = abs(vertical) > 1e-4
        joint_deltas: dict[str, float] = {}
        if move_pan:
            joint_deltas["shoulder_pan"] = pan
        if move_vertical:
            joint_deltas[self.config.vertical_joint] = vertical
        if move_pan and move_vertical:
            reason = "move_pan_and_vertical"
        elif move_pan:
            reason = "move_pan"
        elif move_vertical:
            reason = "move_vertical"
        else:
            reason = "inside_direction_deadband"
        return PointingGazeRecommendation(
            timestamp_s=frame.timestamp_s,
            tracking=True,
            move=move_pan or move_vertical,
            reason=reason,
            confidence=ray.confidence,
            raw_direction=raw,
            filtered_direction=self._filtered,
            pan_delta_deg=pan,
            vertical_delta_deg=vertical,
            joint_deltas=joint_deltas,
        )

    def _stop(
        self,
        timestamp_s: float,
        reason: str,
        confidence: float,
    ) -> PointingGazeRecommendation:
        return PointingGazeRecommendation(
            timestamp_s=timestamp_s,
            tracking=False,
            move=False,
            reason=reason,
            confidence=confidence,
            raw_direction=None,
            filtered_direction=None,
            pan_delta_deg=0.0,
            vertical_delta_deg=0.0,
            joint_deltas={},
        )


class HttpServoBridge:
    """Rate-limited shadow-to-hardware bridge using only the local safety API."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/api/servo",
        *,
        rate_hz: float = 2.5,
        vertical_rate_hz: float = 0.2,
        max_step_deg: float = 2.5,
        vertical_max_step_deg: float = 2.5,
        enable_vertical: bool = False,
        timeout_s: float = 0.75,
    ):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        if vertical_rate_hz <= 0:
            raise ValueError("vertical_rate_hz must be positive")
        self.endpoint = endpoint
        self.period_s = 1.0 / rate_hz
        self.vertical_period_s = 1.0 / vertical_rate_hz
        self.max_step_deg = max_step_deg
        self.vertical_max_step_deg = vertical_max_step_deg
        self.enable_vertical = enable_vertical
        self.timeout_s = timeout_s
        self._last_horizontal_s = float("-inf")
        self._last_vertical_s = float("-inf")
        self._last_any_s = float("-inf")
        self.last_error: str | None = None

    def maybe_send(self, recommendation: ServoRecommendation) -> dict | None:
        if not recommendation.tracking or not recommendation.move:
            return None
        now = time.monotonic()
        deltas: dict[str, float] = {}
        reasons: list[str] = []
        if now - self._last_horizontal_s >= self.period_s:
            pan = _limit(recommendation.pan_delta_deg, self.max_step_deg)
            if abs(pan) > 0.02:
                deltas["shoulder_pan"] = pan
        else:
            reasons.append("horizontal_rate_limited")
        if (
            self.enable_vertical
            and now - self._last_vertical_s >= self.vertical_period_s
            and now - self._last_any_s >= 0.15
        ):
            vertical = _limit(
                recommendation.vertical_delta_deg,
                self.vertical_max_step_deg,
            )
            joint = next(
                (
                    name
                    for name in recommendation.joint_deltas
                    if name != "shoulder_pan"
                ),
                "wrist_flex",
            )
            if abs(vertical) > 0.02:
                deltas[joint] = vertical
        elif self.enable_vertical:
            reasons.append("vertical_rate_limited")
        if not deltas:
            return {
                "sent": False,
                "reason": ",".join(reasons) if reasons else "below_step_threshold",
            }

        body = json.dumps({"deltas": deltas}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.reason}"
            self.last_error = error
            return {"sent": False, "reason": "server_rejected", "error": error}
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            return {"sent": False, "reason": "transport_error", "error": str(exc)}
        finally:
            # Cooldown starts after the request finishes. Starting it before a
            # slow servo action would let the next request violate server rate
            # limiting and produce an avoidable burst of HTTP 429 responses.
            completed = time.monotonic()
            self._last_any_s = completed
            if "shoulder_pan" in deltas:
                self._last_horizontal_s = completed
            if any(joint != "shoulder_pan" for joint in deltas):
                self._last_vertical_s = completed
        if payload.get("ok") is False:
            self.last_error = str(payload.get("message", "actuator did not respond"))
            return {
                "sent": False,
                "reason": "actuator_no_response",
                "error": self.last_error,
                "response": payload,
            }
        self.last_error = None
        return {"sent": True, "deltas": deltas, "response": payload}


class HttpPointingServoBridge:
    """Hardware bridge for stable, short pointing-direction corrections."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/api/servo",
        *,
        rate_hz: float = 1.0,
        vertical_rate_hz: float = 0.2,
        max_step_deg: float = 2.0,
        vertical_max_step_deg: float = 2.0,
        vertical_min_step_deg: float = 0.0,
        max_steps_per_gesture: int = 3,
        settle_s: float = 0.6,
        enable_vertical: bool = False,
        timeout_s: float = 0.75,
    ):
        if rate_hz <= 0 or vertical_rate_hz <= 0:
            raise ValueError("pointing servo rates must be positive")
        if max_step_deg <= 0 or vertical_max_step_deg <= 0:
            raise ValueError("pointing servo step limits must be positive")
        if not 0.0 <= vertical_min_step_deg <= vertical_max_step_deg:
            raise ValueError(
                "vertical_min_step_deg must be between 0 and vertical maximum"
            )
        if max_steps_per_gesture < 0:
            raise ValueError("max_steps_per_gesture cannot be negative")
        if settle_s < 0:
            raise ValueError("settle_s cannot be negative")
        self.endpoint = endpoint
        self.period_s = 1.0 / rate_hz
        self.vertical_period_s = 1.0 / vertical_rate_hz
        self.max_step_deg = max_step_deg
        self.vertical_max_step_deg = vertical_max_step_deg
        self.vertical_min_step_deg = vertical_min_step_deg
        self.max_steps_per_gesture = max_steps_per_gesture
        self.settle_s = settle_s
        self.enable_vertical = enable_vertical
        self.timeout_s = timeout_s
        self.last_error: str | None = None
        self._stable_signs: tuple[int, int] | None = None
        self._stable_since_s: float | None = None
        self._last_horizontal_s = float("-inf")
        self._last_vertical_s = float("-inf")
        self._last_any_s = float("-inf")
        self._blocked_reason: str | None = None
        self._steps_sent = 0

    def maybe_send(self, recommendation: PointingGazeRecommendation) -> dict | None:
        if not recommendation.tracking or not recommendation.move:
            self._reset_stability()
            return {
                "sent": False,
                "reason": recommendation.reason,
            }
        if self._blocked_reason is not None:
            return {
                "sent": False,
                "reason": "control_latched",
                "error": self._blocked_reason,
            }
        if (
            self.max_steps_per_gesture > 0
            and self._steps_sent >= self.max_steps_per_gesture
        ):
            return {
                "sent": False,
                "reason": "gesture_step_limit",
            }

        signs = (
            _sign(recommendation.pan_delta_deg),
            _sign(recommendation.vertical_delta_deg) if self.enable_vertical else 0,
        )
        now = time.monotonic()
        if signs != self._stable_signs or self._stable_since_s is None:
            self._stable_signs = signs
            self._stable_since_s = now
            return {"sent": False, "reason": "pointing_settling"}
        if now - self._stable_since_s < self.settle_s:
            return {"sent": False, "reason": "pointing_settling"}

        deltas: dict[str, float] = {}
        reasons: list[str] = []
        if signs[0] and now - self._last_horizontal_s >= self.period_s:
            pan = _limit(recommendation.pan_delta_deg, self.max_step_deg)
            if abs(pan) > 0.02:
                deltas["shoulder_pan"] = pan
        elif signs[0]:
            reasons.append("horizontal_rate_limited")

        if (
            self.enable_vertical
            and signs[1]
            and now - self._last_vertical_s >= self.vertical_period_s
            and now - self._last_any_s >= 0.15
        ):
            vertical = _limit(
                recommendation.vertical_delta_deg,
                self.vertical_max_step_deg,
            )
            joint = next(
                (
                    name
                    for name in recommendation.joint_deltas
                    if name != "shoulder_pan"
                ),
                "wrist_flex",
            )
            if abs(vertical) > 0.02:
                if abs(vertical) < self.vertical_min_step_deg:
                    vertical = math.copysign(
                        self.vertical_min_step_deg,
                        vertical,
                    )
                deltas[joint] = vertical
        elif self.enable_vertical and signs[1]:
            reasons.append("vertical_rate_limited")

        if not deltas:
            return {
                "sent": False,
                "reason": ",".join(reasons) if reasons else "below_step_threshold",
            }

        body = json.dumps({"deltas": deltas}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.reason}"
            self.last_error = error
            self._blocked_reason = error
            return {"sent": False, "reason": "server_rejected", "error": error}
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            self._blocked_reason = str(exc)
            return {"sent": False, "reason": "transport_error", "error": str(exc)}
        finally:
            completed = time.monotonic()
            self._last_any_s = completed
            if "shoulder_pan" in deltas:
                self._last_horizontal_s = completed
            if any(joint != "shoulder_pan" for joint in deltas):
                self._last_vertical_s = completed
        if payload.get("ok") is False:
            self.last_error = str(payload.get("message", "actuator did not respond"))
            self._blocked_reason = self.last_error
            return {
                "sent": False,
                "reason": "actuator_no_response",
                "error": self.last_error,
                "response": payload,
            }
        self.last_error = None
        self._steps_sent += 1
        return {
            "sent": True,
            "deltas": deltas,
            "response": payload,
            "gesture_step": self._steps_sent,
            "gesture_limit_reached": (
                self.max_steps_per_gesture > 0
                and self._steps_sent >= self.max_steps_per_gesture
            ),
        }

    def _reset_stability(self) -> None:
        self._stable_signs = None
        self._stable_since_s = None
        self._blocked_reason = None
        self._steps_sent = 0


def annotate_servo(
    image: np.ndarray,
    recommendation: ServoRecommendation,
    *,
    pan_sign: int = -1,
    vertical_sign: int = 1,
) -> np.ndarray:
    """Draw a small diagnostic vector without moving the robot."""
    output = image.copy()
    h, w = output.shape[:2]
    center = (w // 2, h // 2)
    color = (60, 230, 170) if recommendation.move else (160, 170, 180)
    dx = int(80 * pan_sign * (recommendation.pan_delta_deg / 2.0))
    dy = int(80 * vertical_sign * (recommendation.vertical_delta_deg / 2.0))
    if dx or dy:
        cv2.arrowedLine(
            output,
            center,
            (center[0] + dx, center[1] + dy),
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.25,
        )
    cv2.putText(
        output,
        (
            f"SHADOW pan={recommendation.pan_delta_deg:+.2f} "
            f"wrist={recommendation.vertical_delta_deg:+.2f} "
            f"({recommendation.reason})"
        ),
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def annotate_pointing_gaze(
    image: np.ndarray,
    recommendation: PointingGazeRecommendation,
    *,
    pan_sign: int = -1,
    vertical_sign: int = 1,
    mode_label: str = "SHADOW",
) -> np.ndarray:
    """Draw the proposed pointing turn without changing the robot."""
    output = image.copy()
    h, w = output.shape[:2]
    center = (w // 2, h // 2)
    color = (255, 190, 40) if recommendation.move else (160, 170, 180)
    dx = int(35 * pan_sign * recommendation.pan_delta_deg)
    dy = int(35 * vertical_sign * recommendation.vertical_delta_deg)
    if dx or dy:
        cv2.arrowedLine(
            output,
            center,
            (center[0] + dx, center[1] + dy),
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.25,
        )
    cv2.putText(
        output,
        (
            f"POINT {mode_label} pan={recommendation.pan_delta_deg:+.2f} "
            f"wrist={recommendation.vertical_delta_deg:+.2f} "
            f"({recommendation.reason})"
        ),
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def _outside_deadband(value: float, deadband: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadband:
        return 0.0
    return (1.0 if value > 0 else -1.0) * (magnitude - deadband) / (1.0 - deadband)


def _clamp(value: float, maximum: float) -> float:
    return max(-maximum, min(maximum, value))


def _limit(value: float, maximum: float) -> float:
    return max(-maximum, min(maximum, value))


def _sign(value: float) -> int:
    if value > 1e-4:
        return 1
    if value < -1e-4:
        return -1
    return 0


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (0.0, 0.0)
    return (dx / length, dy / length)

