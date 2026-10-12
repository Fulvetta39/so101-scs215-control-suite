"""Black-ball visual servoing for the SO-101 / SCS215 arm.

The default command is an entirely offline simulation.  Camera preview never
moves the arm.  Real movement needs both ``run --real`` and an interactive
confirmation after a stable target has been seen by the camera.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
IK_DIR = ROOT_DIR / "SO101_SCS215_IK键盘直线控制"
DEFAULT_URDF = ROOT_DIR / "SO101" / "so101_new_calib.urdf"
DEFAULT_MAPPING = IK_DIR / "joint_mapping.json"


@dataclass(frozen=True)
class BallDetection:
    center: tuple[float, float]
    radius: float
    area: float
    circularity: float
    confidence: float


@dataclass(frozen=True)
class ServoDecision:
    status: str
    error_px: tuple[float, float]
    direction: str | None
    step_m: float


class MotionBudgetReached(RuntimeError):
    """The next planned joint microstep would exceed the motion budget."""


def horizontal_direction_for_error(error_x: float, invert_x: bool = False) -> str:
    positive = error_x > 0
    if invert_x:
        positive = not positive
    return "right" if positive else "left"


def detections_match(
    first: BallDetection,
    second: BallDetection,
    *,
    max_center_delta_px: float = 55.0,
    max_radius_change_fraction: float = 0.35,
) -> bool:
    center_delta = math.hypot(
        second.center[0] - first.center[0],
        second.center[1] - first.center[1],
    )
    radius_scale = max(first.radius, second.radius, 1.0)
    radius_change = abs(second.radius - first.radius) / radius_scale
    return (
        center_delta <= max_center_delta_px
        and radius_change <= max_radius_change_fraction
    )


class BlackBallDetector:
    """Find a large, dark, approximately circular contour."""

    def __init__(
        self,
        *,
        threshold: int = 125,
        min_area_fraction: float = 0.0005,
        max_area_fraction: float = 0.40,
        min_circularity: float = 0.55,
        max_radius_fraction: float = 0.24,
    ) -> None:
        if not 0 <= threshold <= 255:
            raise ValueError("黑色阈值必须在 0~255")
        self.threshold = int(threshold)
        self.min_area_fraction = float(min_area_fraction)
        self.max_area_fraction = float(max_area_fraction)
        self.min_circularity = float(min_circularity)
        self.max_radius_fraction = float(max_radius_fraction)

    def detect(self, frame: np.ndarray) -> BallDetection | None:
        if frame is None or frame.ndim not in (2, 3) or frame.size == 0:
            raise ValueError("输入图像为空或格式不正确")
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        image_area = float(height * width)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        mask = cv2.threshold(
            blurred, self.threshold, 255, cv2.THRESH_BINARY_INV
        )[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: BallDetection | None = None
        best_score = -math.inf
        for contour in contours:
            area = float(cv2.contourArea(contour))
            fraction = area / image_area
            if not self.min_area_fraction <= fraction <= self.max_area_fraction:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1:
                continue
            aspect = box_w / max(1.0, float(box_h))
            if not 0.72 <= aspect <= 1.38:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < 7.0 or radius > min(height, width) * self.max_radius_fraction:
                continue
            fill = area / max(1.0, math.pi * radius * radius)
            if fill < 0.58:
                continue
            candidate_mask = np.zeros_like(gray)
            cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=-1)
            mean_gray = float(cv2.mean(gray, mask=candidate_mask)[0])
            darkness = max(0.0, min(1.0, 1.0 - mean_gray / 160.0))
            confidence = max(
                0.0,
                min(1.0, 0.48 * circularity + 0.30 * fill + 0.22 * darkness),
            )
            score = confidence + min(0.35, fraction * 3.0)
            if score > best_score:
                best_score = score
                best = BallDetection(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    area=area,
                    circularity=circularity,
                    confidence=confidence,
                )
        return best if best is not None else self._detect_hough(gray)

    def _detect_hough(self, gray: np.ndarray) -> BallDetection | None:
        """Fallback for a sphere whose cast shadow joins its threshold contour."""
        height, width = gray.shape[:2]
        smoothed = cv2.GaussianBlur(gray, (9, 9), 1.5)
        circles = cv2.HoughCircles(
            smoothed,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(40.0, min(height, width) / 6.0),
            param1=90,
            param2=35,
            minRadius=max(8, int(min(height, width) * 0.025)),
            maxRadius=max(12, int(min(height, width) * self.max_radius_fraction)),
        )
        if circles is None:
            return None
        best: BallDetection | None = None
        best_score = -math.inf
        for cx, cy, radius in circles[0]:
            if radius < 7 or not (0 <= cx < width and 0 <= cy < height):
                continue
            inner = np.zeros_like(gray)
            cv2.circle(
                inner,
                (int(round(cx)), int(round(cy))),
                max(2, int(round(radius * 0.72))),
                255,
                -1,
            )
            inner_mean = float(cv2.mean(gray, mask=inner)[0])
            if inner_mean > 115.0:
                continue
            outer = np.zeros_like(gray)
            cv2.circle(
                outer,
                (int(round(cx)), int(round(cy))),
                max(3, int(round(radius * 1.15))),
                255,
                -1,
            )
            cv2.circle(
                outer,
                (int(round(cx)), int(round(cy))),
                max(2, int(round(radius * 0.96))),
                0,
                -1,
            )
            ring_mean = float(cv2.mean(gray, mask=outer)[0])
            contrast = ring_mean - inner_mean
            if contrast < 28.0:
                continue
            darkness = max(0.0, min(1.0, 1.0 - inner_mean / 150.0))
            contrast_score = max(0.0, min(1.0, contrast / 140.0))
            confidence = 0.55 * darkness + 0.45 * contrast_score
            score = confidence + min(0.25, radius / max(height, width))
            if score > best_score:
                best_score = score
                best = BallDetection(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    area=float(math.pi * radius * radius),
                    circularity=1.0,
                    confidence=float(confidence),
                )
        return best


class PixelServo:
    """Turn image error into one conservative Cartesian move at a time."""

    def __init__(
        self,
        *,
        deadband_px: float = 28.0,
        min_step_m: float = 0.001,
        max_step_m: float = 0.004,
        gain_m: float = 0.012,
        stable_frames: int = 3,
        invert_x: bool = False,
        invert_y: bool = False,
    ) -> None:
        if deadband_px < 0 or not 0 < min_step_m <= max_step_m:
            raise ValueError("死区和移动步长配置不正确")
        if stable_frames < 1:
            raise ValueError("稳定帧数必须至少为 1")
        self.deadband_px = float(deadband_px)
        self.min_step_m = float(min_step_m)
        self.max_step_m = float(max_step_m)
        self.gain_m = float(gain_m)
        self.stable_frames = int(stable_frames)
        self.invert_x = bool(invert_x)
        self.invert_y = bool(invert_y)
        self._seen = 0
        self._centered_seen = 0
        self._last_detection: BallDetection | None = None

    def decide(
        self,
        detection: BallDetection | None,
        frame_shape: Sequence[int],
    ) -> ServoDecision:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if detection is None:
            self._seen = 0
            self._centered_seen = 0
            self._last_detection = None
            return ServoDecision("target_lost", (0.0, 0.0), None, 0.0)
        if self._last_detection is not None and not detections_match(
            self._last_detection, detection
        ):
            self._seen = 1
            self._centered_seen = 0
        else:
            self._seen += 1
        self._last_detection = detection
        error_x = detection.center[0] - width / 2.0
        error_y = detection.center[1] - height / 2.0
        if abs(error_x) <= self.deadband_px and abs(error_y) <= self.deadband_px:
            self._centered_seen += 1
            if self._centered_seen < self.stable_frames:
                return ServoDecision("centering", (error_x, error_y), None, 0.0)
            return ServoDecision("centered", (error_x, error_y), None, 0.0)
        self._centered_seen = 0
        if self._seen < self.stable_frames:
            return ServoDecision("stabilizing", (error_x, error_y), None, 0.0)

        x_norm = error_x / max(1.0, width / 2.0)
        y_norm = error_y / max(1.0, height / 2.0)
        # Camera pixels have the same physical pitch on X/Y.  Compare pixel
        # errors directly; normalizing by half-width/height over-prioritized
        # the shorter image dimension and could repeatedly select an
        # unreachable vertical move while the larger horizontal error remained.
        if abs(error_x) >= abs(error_y):
            direction = horizontal_direction_for_error(error_x, self.invert_x)
            magnitude = abs(x_norm)
        else:
            positive = y_norm > 0
            if self.invert_y:
                positive = not positive
            direction = "down" if positive else "up"
            magnitude = abs(y_norm)
        step = max(self.min_step_m, min(self.max_step_m, self.gain_m * magnitude))
        return ServoDecision("move", (error_x, error_y), direction, step)


class SyntheticCamera:
    """Simple image/actuator model used to verify the loop without hardware."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        target: tuple[float, float] | None = None,
        radius: int = 44,
        pixels_per_meter: float = 11_000.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.target = np.asarray(target or (width * 0.80, height * 0.22), dtype=float)
        self.radius = int(radius)
        self.pixels_per_meter = float(pixels_per_meter)

    def frame(self) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 242, dtype=np.uint8)
        cv2.rectangle(image, (24, 24), (155, 68), (205, 205, 205), -1)
        cv2.line(image, (35, self.height - 45), (195, self.height - 45), (175, 175, 175), 9)
        center = tuple(int(round(value)) for value in self.target)
        cv2.circle(image, center, self.radius, (8, 8, 8), -1, cv2.LINE_AA)
        cv2.circle(
            image,
            (center[0] - self.radius // 3, center[1] - self.radius // 3),
            max(3, self.radius // 8),
            (75, 75, 75),
            -1,
            cv2.LINE_AA,
        )
        return image

    def move(self, direction: str, step_m: float) -> None:
        shift = self.pixels_per_meter * float(step_m)
        # The image moves opposite to a camera mounted on the moving end effector.
        delta = {
            "left": (shift, 0.0),
            "right": (-shift, 0.0),
            "up": (0.0, shift),
            "down": (0.0, -shift),
        }[direction]
        self.target += np.asarray(delta)


def annotate(
    frame: np.ndarray,
    detection: BallDetection | None,
    decision: ServoDecision | None = None,
) -> np.ndarray:
    result = frame.copy()
    height, width = result.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(result, center, (0, 180, 0), cv2.MARKER_CROSS, 34, 2)
    cv2.rectangle(
        result,
        (int(center[0] - 28), int(center[1] - 28)),
        (int(center[0] + 28), int(center[1] + 28)),
        (0, 180, 0),
        1,
    )
    if detection is not None:
        ball_center = tuple(int(round(value)) for value in detection.center)
        cv2.circle(result, ball_center, int(round(detection.radius)), (0, 0, 255), 2)
        cv2.circle(result, ball_center, 3, (0, 0, 255), -1)
        cv2.line(result, center, ball_center, (255, 120, 0), 2)
        cv2.putText(
            result,
            f"ball conf={detection.confidence:.2f}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 220),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            result,
            "ball not found - no motion",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 220),
            2,
            cv2.LINE_AA,
        )
    if decision is not None:
        command = decision.direction or "stop"
        cv2.putText(
            result,
            f"{decision.status}: {command} {decision.step_m * 1000:.1f} mm",
            (12, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (40, 90, 210),
            2,
            cv2.LINE_AA,
        )
    return result


def read_image(path: Path) -> np.ndarray:
    """Read through NumPy so Windows paths containing Chinese stay supported."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(f"无法读取图像：{path}") from exc
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"无法解码图像：{path}")
    return frame


def write_image(path: Path, frame: np.ndarray) -> None:
    """Write through NumPy so Windows paths containing Chinese stay supported."""
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, frame)
    if not ok:
        raise RuntimeError(f"无法编码图像：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise RuntimeError(f"无法写入图像：{path}") from exc


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate camera input before both detection and control decisions."""
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if degrees not in rotations:
        raise ValueError("画面旋转只支持 0/90/180/270 度")
    code = rotations[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def run_demo(args: argparse.Namespace) -> int:
    detector = detector_from_args(args)
    servo = servo_from_args(args, stable_frames=1)
    scene = SyntheticCamera()
    first = scene.frame()
    trace: list[tuple[str, float]] = []
    final_detection: BallDetection | None = None
    final_decision: ServoDecision | None = None
    for _ in range(args.max_moves + 1):
        frame = scene.frame()
        final_detection = detector.detect(frame)
        final_decision = servo.decide(final_detection, frame.shape)
        if final_decision.status == "centered":
            break
        if final_decision.direction is None:
            continue
        trace.append((final_decision.direction, final_decision.step_m))
        scene.move(final_decision.direction, final_decision.step_m)
    final = annotate(scene.frame(), final_detection, final_decision)
    first_annotated = annotate(first, detector.detect(first), None)
    montage = np.hstack((first_annotated, final))
    output = Path(args.output).resolve()
    write_image(output, montage)
    centered = final_decision is not None and final_decision.status == "centered"
    print(f"离线闭环：移动 {len(trace)} 次，结果={'已对准' if centered else '未对准'}")
    print(f"演示对比图：{output}")
    return 0 if centered else 2


def run_image(args: argparse.Namespace) -> int:
    source = Path(args.path).resolve()
    frame = read_image(source)
    detector = detector_from_args(args)
    detection = detector.detect(frame)
    decision = servo_from_args(args, stable_frames=1).decide(detection, frame.shape)
    output = Path(args.output).resolve()
    write_image(output, annotate(frame, detection, decision))
    if detection is None:
        print("未检测到可信的黑色圆球，已保持停止。")
        print(f"标注图：{output}")
        return 2
    print(
        f"黑球中心=({detection.center[0]:.1f}, {detection.center[1]:.1f}) px，"
        f"半径={detection.radius:.1f} px，置信度={detection.confidence:.2f}"
    )
    print(f"当前决策：{decision.direction or '停止'}")
    print(f"标注图：{output}")
    return 0


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开摄像头 {index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def settle_and_drain_camera(
    capture: cv2.VideoCapture,
    settle_s: float,
    flush_frames: int,
) -> None:
    """Wait for arm/camera settling and discard frames queued before the move."""
    if settle_s > 0:
        time.sleep(settle_s)
    for _ in range(max(0, int(flush_frames))):
        if not capture.grab():
            break


def plan_base_yaw_raw(
    kinematics,
    mapping,
    current_raw: int,
    current_joints_deg: Sequence[float],
    safe_range: tuple[int, int],
    direction: str,
    distance_m: float,
    *,
    min_raw_step: int = 8,
    max_raw_step: int = 24,
) -> int:
    """Plan a small ID1-only move that produces the requested base-Y motion."""
    if direction not in {"left", "right"}:
        raise ValueError("底座水平微调只支持 left/right")
    if distance_m <= 0:
        raise ValueError("底座水平微调距离必须大于 0")
    current_xyz = np.asarray(kinematics.position(current_joints_deg), dtype=float)
    wanted_sign = 1.0 if direction == "left" else -1.0
    candidates: list[tuple[float, int]] = []
    for raw_delta in (-1, 1):
        candidate_raw = current_raw + raw_delta
        if not safe_range[0] <= candidate_raw <= safe_range[1]:
            continue
        candidate_joints = list(float(value) for value in current_joints_deg)
        candidate_joints[0] = mapping.raw_to_degrees(candidate_raw)
        candidate_xyz = np.asarray(kinematics.position(candidate_joints), dtype=float)
        projected = wanted_sign * float(candidate_xyz[1] - current_xyz[1])
        candidates.append((projected, raw_delta))
    if not candidates:
        raise RuntimeError("ID1 已在安全范围边界，无法继续水平微调")
    per_raw_m, raw_sign = max(candidates)
    if per_raw_m <= 1e-7:
        raise RuntimeError("当前姿态下 ID1 无法产生所需水平方向")
    raw_steps = max(
        1,
        int(min_raw_step),
        min(int(max_raw_step), int(math.ceil(distance_m / per_raw_m))),
    )
    return max(
        safe_range[0],
        min(safe_range[1], current_raw + raw_sign * raw_steps),
    )


def plan_wrist_flex_raw(
    kinematics,
    mapping,
    current_raw: int,
    current_joints_deg: Sequence[float],
    safe_range: tuple[int, int],
    direction: str,
    distance_m: float,
    *,
    min_raw_step: int = 8,
    max_raw_step: int = 24,
) -> int:
    """Plan an ID4-only arc that moves the endpoint up or down.

    This intentionally allows simultaneous X/Y motion: the vision task asks
    the endpoint to approach the target, not to preserve a Cartesian plane.
    """
    if direction not in {"up", "down"}:
        raise ValueError("腕部垂直微调只支持 up/down")
    if distance_m <= 0:
        raise ValueError("腕部垂直微调距离必须大于 0")
    current_xyz = np.asarray(kinematics.position(current_joints_deg), dtype=float)
    wanted_sign = 1.0 if direction == "up" else -1.0
    candidates: list[tuple[float, int]] = []
    for raw_delta in (-1, 1):
        candidate_raw = current_raw + raw_delta
        if not safe_range[0] <= candidate_raw <= safe_range[1]:
            continue
        candidate_joints = list(float(value) for value in current_joints_deg)
        candidate_joints[3] = mapping.raw_to_degrees(candidate_raw)
        candidate_xyz = np.asarray(kinematics.position(candidate_joints), dtype=float)
        projected = wanted_sign * float(candidate_xyz[2] - current_xyz[2])
        candidates.append((projected, raw_delta))
    if not candidates:
        raise RuntimeError("ID4 已在安全范围边界，无法继续垂直微调")
    per_raw_m, raw_sign = max(candidates)
    if per_raw_m <= 1e-7:
        raise RuntimeError("当前姿态下 ID4 无法产生所需垂直方向")
    raw_steps = max(
        1,
        int(min_raw_step),
        min(int(max_raw_step), int(math.ceil(distance_m / per_raw_m))),
    )
    return max(
        safe_range[0],
        min(safe_range[1], current_raw + raw_sign * raw_steps),
    )


def validate_camera_target(
    capture: cv2.VideoCapture,
    detector: BlackBallDetector,
    required_frames: int = 5,
    rotate_degrees: int = 0,
) -> tuple[np.ndarray, BallDetection]:
    stable = 0
    last_frame: np.ndarray | None = None
    last_detection: BallDetection | None = None
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if not ok:
            stable = 0
            continue
        frame = rotate_frame(frame, rotate_degrees)
        detection = detector.detect(frame)
        if detection is None:
            stable = 0
        elif last_detection is not None and detections_match(last_detection, detection):
            stable += 1
        else:
            stable = 1
        last_frame, last_detection = frame, detection
        if stable >= required_frames and last_detection is not None:
            return last_frame, last_detection
    raise RuntimeError("摄像头未在 6 秒内稳定检测到黑球；机械臂未启动")


def run_camera(args: argparse.Namespace) -> int:
    detector = detector_from_args(args)
    servo = servo_from_args(args)
    capture = open_camera(args.camera, args.width, args.height)
    print("摄像头预览模式：不会连接或移动机械臂。按 Esc 退出。")
    headless = bool(args.headless)
    warned_headless = False
    frames = 0
    detections = 0
    last_frame: np.ndarray | None = None
    last_detection: BallDetection | None = None
    last_decision: ServoDecision | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("摄像头取帧失败")
            frame = rotate_frame(frame, args.rotate)
            detection = detector.detect(frame)
            decision = servo.decide(detection, frame.shape)
            frames += 1
            detections += int(detection is not None)
            last_frame, last_detection, last_decision = frame, detection, decision
            if not headless:
                try:
                    cv2.imshow(
                        "SO-101 black-ball preview (NO ARM MOTION)",
                        annotate(frame, detection, decision),
                    )
                    if cv2.waitKey(1) & 0xFF == 27:
                        return 0
                except cv2.error:
                    headless = True
                    warned_headless = True
                    print(
                        "当前 OpenCV 不支持窗口显示，已自动切换为 30 帧保存检查。"
                    )
            if headless and frames >= args.check_frames:
                assert last_frame is not None
                output = Path(args.output).resolve()
                write_image(output, annotate(last_frame, last_detection, last_decision))
                print(f"摄像头取帧 {frames} 帧，黑球检出 {detections} 帧。")
                if last_detection is not None:
                    print(
                        f"最后一帧黑球中心={last_detection.center}，"
                        f"置信度={last_detection.confidence:.2f}，"
                        f"决策={last_decision.direction or '停止'}。"
                    )
                else:
                    print("最后一帧未检测到黑球，不会生成移动指令。")
                print(f"摄像头检查图：{output}")
                return 0 if detections >= max(1, frames // 2) else 2
    finally:
        capture.release()
        if not warned_headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


@contextmanager
def real_arm_executor(args: argparse.Namespace) -> Iterator[Callable[[str, float, float], float]]:
    if not args.real:
        raise RuntimeError("真机运动必须明确增加 --real")
    if str(IK_DIR) not in sys.path:
        sys.path.insert(0, str(IK_DIR))
    from ik_core import CartesianController, URDFKinematics
    from ik_keyboard_control import (
        SCS215ArmDriver,
        SCS215Bus,
        find_port,
        load_mappings,
        print_states,
        validate_models,
    )

    port = find_port(args.port)
    kinematics = URDFKinematics(args.urdf)
    mappings = load_mappings(Path(args.mapping))
    with SCS215Bus(port) as bus:
        states = bus.snapshot()
        print_states(states)
        validate_models(states)
        driver = SCS215ArmDriver(bus, states, mappings, kinematics.joint_limits_deg)
        print("软件安全范围：" + "，".join(
            f"ID{i + 1} {low}~{high}"
            for i, (low, high) in enumerate(driver.safe_raw_ranges)
        ))
        answer = input(
            "摄像头已稳定看到黑球。请托住机械臂、确认周围无障碍，"
            "输入 MOVE 才会开启扭矩："
        ).strip()
        if answer != "MOVE":
            raise RuntimeError("未收到 MOVE 确认，机械臂保持卸力")
        driver.hold_then_enable()
        controller = CartesianController(
            kinematics,
            driver,
            joint_limits_deg=driver.effective_joint_limits_deg,
            cartesian_resolution_m=args.resolution_mm / 1000.0,
            tolerance_m=args.tolerance_mm / 1000.0,
            max_joint_step_deg=args.max_joint_step,
            command_interval_s=args.interval,
        )
        try:
            controller.initialize()
            controller.cartesian_reference_xyz = kinematics.position(controller.joints_deg)

            def execute(direction: str, step_m: float, remaining_m: float) -> float:
                if direction in {"left", "right"}:
                    before_raw = driver.bus.positions()
                    before_joints, _ = driver.get_joint_state()
                    before_xyz = np.asarray(kinematics.position(before_joints), dtype=float)
                    target_raw = plan_base_yaw_raw(
                        kinematics,
                        driver.mappings[0],
                        before_raw[0],
                        before_joints,
                        driver.safe_raw_ranges[0],
                        direction,
                        step_m,
                    )
                    target_joints = list(before_joints)
                    target_joints[0] = driver.mappings[0].raw_to_degrees(target_raw)
                    planned_xyz = np.asarray(kinematics.position(target_joints), dtype=float)
                    planned_distance_m = float(np.linalg.norm(planned_xyz - before_xyz))
                    if max(step_m, planned_distance_m + 0.001) > remaining_m:
                        raise MotionBudgetReached(
                            f"剩余运动额度 {remaining_m * 100:.2f} cm，"
                            f"下一次 ID1 微调预计需要 {planned_distance_m * 100:.2f} cm；"
                            "已在动作前停止"
                        )
                    driver.send_joint_target(target_joints, driver.gripper_raw)
                    _, _, actual_raw = driver.verify_last_target()
                    actual_joints, _ = driver.get_joint_state()
                    actual_xyz = np.asarray(kinematics.position(actual_joints), dtype=float)
                    wanted_sign = 1.0 if direction == "left" else -1.0
                    achieved_m = wanted_sign * float(actual_xyz[1] - before_xyz[1])
                    actual_distance_m = float(np.linalg.norm(actual_xyz - before_xyz))
                    controller.joints_deg = actual_joints
                    controller.cartesian_reference_xyz = tuple(float(value) for value in actual_xyz)
                    print(
                        f"ID1 水平微调：{before_raw[0]} -> {target_raw}，"
                        f"回读 {actual_raw[0]}，实际方向位移 "
                        f"{achieved_m * 1000:+.1f} mm。",
                        flush=True,
                    )
                    if achieved_m <= 0.0002:
                        raise RuntimeError(
                            f"ID1 微调后未产生有效{direction}位移，已停止"
                        )
                    return actual_distance_m
                if direction in {"up", "down"}:
                    before_raw = driver.bus.positions()
                    before_joints, _ = driver.get_joint_state()
                    before_xyz = np.asarray(kinematics.position(before_joints), dtype=float)
                    target_raw = plan_wrist_flex_raw(
                        kinematics,
                        driver.mappings[3],
                        before_raw[3],
                        before_joints,
                        driver.safe_raw_ranges[3],
                        direction,
                        step_m,
                    )
                    target_joints = list(before_joints)
                    target_joints[3] = driver.mappings[3].raw_to_degrees(target_raw)
                    planned_xyz = np.asarray(kinematics.position(target_joints), dtype=float)
                    planned_distance_m = float(np.linalg.norm(planned_xyz - before_xyz))
                    if max(step_m, planned_distance_m + 0.001) > remaining_m:
                        raise MotionBudgetReached(
                            f"剩余运动额度 {remaining_m * 100:.2f} cm，"
                            f"下一次 ID4 微调预计需要 {planned_distance_m * 100:.2f} cm；"
                            "已在动作前停止"
                        )
                    driver.send_joint_target(target_joints, driver.gripper_raw)
                    _, _, actual_raw = driver.verify_last_target()
                    actual_joints, _ = driver.get_joint_state()
                    actual_xyz = np.asarray(kinematics.position(actual_joints), dtype=float)
                    wanted_sign = 1.0 if direction == "up" else -1.0
                    achieved_m = wanted_sign * float(actual_xyz[2] - before_xyz[2])
                    actual_distance_m = float(np.linalg.norm(actual_xyz - before_xyz))
                    controller.joints_deg = actual_joints
                    controller.cartesian_reference_xyz = tuple(float(value) for value in actual_xyz)
                    print(
                        f"ID4 垂直微调：{before_raw[3]} -> {target_raw}，"
                        f"回读 {actual_raw[3]}，实际方向位移 "
                        f"{achieved_m * 1000:+.1f} mm。",
                        flush=True,
                    )
                    if achieved_m <= 0.0002:
                        raise RuntimeError(
                            f"ID4 微调后未产生有效{direction}位移，已停止"
                        )
                    return actual_distance_m
                raise ValueError(f"不支持的视觉移动方向：{direction}")

            print("真机闭环已开始；按 Esc 立即结束并卸力。")
            yield execute
        finally:
            driver.shutdown()
            print("六个舵机已卸力。")


def run_real(args: argparse.Namespace) -> int:
    if not args.real:
        raise RuntimeError("真机运动必须明确增加 --real")
    detector = detector_from_args(args)
    servo = servo_from_args(args)
    capture = open_camera(args.camera, args.width, args.height)
    moves = 0
    total_motion = 0.0
    previous_error: float | None = None
    worsening = 0
    display_available = not args.headless
    warned_headless = False
    loop_started = 0.0
    frames_seen = 0
    detections_seen = 0
    try:
        _, detection = validate_camera_target(
            capture, detector, rotate_degrees=args.rotate
        )
        print(
            f"摄像头黑球预检通过：中心 {detection.center}，"
            f"置信度 {detection.confidence:.2f}"
        )
        with real_arm_executor(args) as execute:
            # The operator confirmation can take arbitrarily long; idle time
            # starts only after torque-on has completed and the loop begins.
            loop_started = time.monotonic()
            while moves < args.max_moves and total_motion < args.max_total_cm / 100.0:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("摄像头取帧失败，已停止")
                frame = rotate_frame(frame, args.rotate)
                detection = detector.detect(frame)
                decision = servo.decide(detection, frame.shape)
                frames_seen += 1
                detections_seen += int(detection is not None)
                if display_available:
                    try:
                        cv2.imshow(
                            "SO-101 visual closed loop - Esc to stop",
                            annotate(frame, detection, decision),
                        )
                        if cv2.waitKey(1) & 0xFF == 27:
                            return 0
                    except cv2.error:
                        display_available = False
                        warned_headless = True
                        print(
                            "OpenCV 窗口不可用；闭环仍受移动次数、"
                            "累计距离和误差反向保护，可按 Ctrl+C 停止。"
                        )
                if decision.status == "centered":
                    print("黑球已进入画面中心死区，闭环完成。")
                    return 0
                if decision.direction is None:
                    if time.monotonic() - loop_started >= args.idle_timeout_s:
                        raise RuntimeError(
                            f"闭环 {args.idle_timeout_s:.1f} 秒内无可执行决策："
                            f"取帧 {frames_seen}，检出 {detections_seen}；已卸力"
                        )
                    continue
                error_norm = math.hypot(*decision.error_px)
                if previous_error is not None and error_norm > previous_error + 8.0:
                    worsening += 1
                else:
                    worsening = 0
                if worsening >= 2:
                    raise RuntimeError(
                        "连续两次移动后像素误差反而增大，已卸力；"
                        "请检查摄像头方向并调整 --invert-x/--invert-y"
                    )
                motion_direction = decision.direction
                print(
                    f"闭环第 {moves + 1} 步：像素误差 "
                    f"({decision.error_px[0]:+.1f}, {decision.error_px[1]:+.1f})，"
                    f"向{motion_direction}移动 {decision.step_m * 1000:.1f} mm。",
                    flush=True,
                )
                remaining_m = args.max_total_cm / 100.0 - total_motion
                try:
                    actual_motion_m = execute(
                        motion_direction,
                        decision.step_m,
                        remaining_m,
                    )
                except MotionBudgetReached as exc:
                    print(str(exc), flush=True)
                    return 0
                previous_error = error_norm
                moves += 1
                # Use the larger of requested and measured endpoint movement so
                # the physical safety budget is never underestimated.
                total_motion += max(decision.step_m, actual_motion_m)
                print(
                    f"累计实际安全计数 {total_motion * 100:.2f} cm / "
                    f"{args.max_total_cm:.2f} cm。",
                    flush=True,
                )
                loop_started = time.monotonic()
                frames_seen = 0
                detections_seen = 0
                settle_and_drain_camera(
                    capture,
                    args.settle_ms / 1000.0,
                    args.flush_frames,
                )
        raise RuntimeError("已达移动次数或累计距离上限，未继续追踪")
    finally:
        capture.release()
        if not warned_headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def detector_from_args(args: argparse.Namespace) -> BlackBallDetector:
    return BlackBallDetector(
        threshold=args.threshold,
        min_area_fraction=args.min_area_fraction,
        min_circularity=args.min_circularity,
        max_radius_fraction=args.max_ball_radius_fraction,
    )


def servo_from_args(args: argparse.Namespace, *, stable_frames: int | None = None) -> PixelServo:
    return PixelServo(
        deadband_px=args.deadband,
        min_step_m=args.min_step_mm / 1000.0,
        max_step_m=args.max_step_mm / 1000.0,
        gain_m=args.gain_mm / 1000.0,
        stable_frames=args.stable_frames if stable_frames is None else stable_frames,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold", type=int, default=125, help="黑色灰度阈值")
    parser.add_argument("--min-area-fraction", type=float, default=0.0005)
    parser.add_argument("--min-circularity", type=float, default=0.55)
    parser.add_argument(
        "--max-ball-radius-fraction",
        type=float,
        default=0.24,
        help="黑球最大半径占画面短边的比例",
    )
    parser.add_argument("--deadband", type=float, default=28.0, help="中心死区像素")
    parser.add_argument("--min-step-mm", type=float, default=1.0)
    parser.add_argument("--max-step-mm", type=float, default=4.0)
    parser.add_argument("--gain-mm", type=float, default=12.0)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--invert-x", action="store_true", help="反转左右映射")
    parser.add_argument("--invert-y", action="store_true", help="反转上下映射")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101 黑球视觉闭环控制（默认仅离线模拟）"
    )
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="无摄像头、无机械臂的离线闭环")
    add_common(demo)
    demo.add_argument("--max-moves", type=int, default=30)
    demo.add_argument("--output", default=str(PROJECT_DIR / "demo_result.png"))
    demo.set_defaults(func=run_demo)

    image = sub.add_parser("image", help="检测一张静态图像，不移动机械臂")
    add_common(image)
    image.add_argument("path")
    image.add_argument("--output", default=str(PROJECT_DIR / "detected_ball.png"))
    image.set_defaults(func=run_image)

    camera = sub.add_parser("camera", help="摄像头检测预览，不移动机械臂")
    add_common(camera)
    camera.add_argument("--camera", type=int, default=1)
    camera.add_argument("--width", type=int, default=640)
    camera.add_argument("--height", type=int, default=480)
    camera.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=180)
    camera.add_argument("--headless", action="store_true", help="不打开窗口，保存多帧检查结果")
    camera.add_argument("--check-frames", type=int, default=30)
    camera.add_argument("--output", default=str(PROJECT_DIR / "camera_preview.png"))
    camera.set_defaults(func=run_camera)

    real = sub.add_parser("run", help="摄像头 + 真机闭环（需 --real）")
    add_common(real)
    real.add_argument("--camera", type=int, default=1)
    real.add_argument("--width", type=int, default=640)
    real.add_argument("--height", type=int, default=480)
    real.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=180)
    real.add_argument("--headless", action="store_true", help="不打开 OpenCV 窗口")
    real.add_argument("--port")
    real.add_argument("--real", action="store_true")
    real.add_argument("--urdf", default=str(DEFAULT_URDF))
    real.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    real.add_argument("--resolution-mm", type=float, default=2.0)
    real.add_argument("--tolerance-mm", type=float, default=0.8)
    real.add_argument("--max-joint-step", type=float, default=12.0)
    real.add_argument("--interval", type=float, default=0.05)
    real.add_argument("--settle-ms", type=int, default=180)
    real.add_argument("--flush-frames", type=int, default=5)
    real.add_argument("--idle-timeout-s", type=float, default=8.0)
    real.add_argument("--max-moves", type=int, default=40)
    real.add_argument("--max-total-cm", type=float, default=8.0)
    real.set_defaults(func=run_real)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "min_area_fraction": args.min_area_fraction,
        "min_circularity": args.min_circularity,
        "max_ball_radius_fraction": args.max_ball_radius_fraction,
        "min_step_mm": args.min_step_mm,
        "max_step_mm": args.max_step_mm,
        "gain_mm": args.gain_mm,
        "stable_frames": args.stable_frames,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError("以下参数必须大于 0：" + ", ".join(bad))
    if args.min_step_mm > args.max_step_mm:
        raise ValueError("--min-step-mm 不能大于 --max-step-mm")
    if args.max_ball_radius_fraction >= 0.5:
        raise ValueError("--max-ball-radius-fraction 必须小于 0.5")
    if getattr(args, "max_total_cm", 1.0) <= 0 or getattr(args, "max_moves", 1) <= 0:
        raise ValueError("移动上限必须大于 0")
    if getattr(args, "flush_frames", 0) < 0:
        raise ValueError("--flush-frames 不能小于 0")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        args = parser.parse_args(["demo"] if argv is None else ["demo", *argv])
    try:
        validate_args(args)
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n已由用户中止。")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
