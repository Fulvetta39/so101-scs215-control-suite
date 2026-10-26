"""Guarded ArUco approach servo for the SO-101 arm.

The image is corrected before detection, the lecture's 40/60 percent rule
is available for image-region reporting, while calibrated ArUco PnP and the
hand-eye transform define a 3-D gripper target in front of the marker plane.
All hardware motion is sent through the existing checked Cartesian IK driver.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
IK_DIR = ROOT_DIR / "SO101_SCS215_IK键盘直线控制"
if str(IK_DIR) not in sys.path:
    sys.path.insert(0, str(IK_DIR))

from ik_core import (  # noqa: E402
    CartesianController,
    DIRECTION_JOINT_MOTION_WEIGHTS,
    DIRECTIONS,
    DryRunDriver,
    URDFKinematics,
)
from ik_keyboard_control import (  # noqa: E402
    SCS215ArmDriver,
    SCS215Bus,
    find_port,
    load_mappings,
    print_states,
    validate_models,
)
from hand_eye_capture import open_camera, read_fresh_frame  # noqa: E402
from aruco_visual_servo import (  # noqa: E402
    DEFAULT_CAMERA_RESULT,
    DEFAULT_HAND_EYE_RESULT,
    _write_image,
    detect_aruco,
    homogeneous,
    invert_transform,
    load_camera_calibration,
    rotate_frame,
    select_marker,
    simple_servo_plan,
)


DEFAULT_MAPPING = IK_DIR / "joint_mapping.json"
DEFAULT_URDF = ROOT_DIR / "SO101" / "so101_new_calib.urdf"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "simple_servo"
CAMERA_DELTAS = {
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "up": (0.0, -1.0, 0.0),
    "down": (0.0, 1.0, 0.0),
    "forward": (0.0, 0.0, 1.0),
}
CENTER_OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
VISUAL_JOINT_MOTION_WEIGHTS = {
    **DIRECTION_JOINT_MOTION_WEIGHTS,
    # Near the target ID3 carries substantial gravity load.  Prefer the wrist
    # for upward image correction instead of repeatedly asking ID3 to reverse.
    "up": (1.0, 4.0, 8.0, 0.3, 1.0),
}
# A position-only IK has two practical ways to extend toward the wall target.
# The unweighted local solution increases ID3 until the arm folds back on
# itself.  Favor shoulder extension and make elbow flex expensive so the
# closed loop stays on the long-reach configuration selected at startup.
APPROACH_JOINT_MOTION_WEIGHTS = (1.0, 0.3, 8.0, 1.0, 1.0)
DEFAULT_ELBOW_BRANCH_MARGIN_RAW = 80


@dataclass(frozen=True)
class Observation:
    frame: np.ndarray
    marker_id: int
    center: tuple[float, float]
    corners: np.ndarray
    direction: str
    region_error: float
    area_px2: float
    target_to_camera: np.ndarray
    camera_depth_m: float
    gripper_plane_clearance_m: float


class MarkerLostError(ValueError):
    def __init__(self, message: str, frame: np.ndarray) -> None:
        super().__init__(message)
        self.frame = frame


def load_camera_to_gripper(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform = np.asarray(payload["T_gripper_camera"], dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("手眼标定矩阵必须是有限的 4x4 矩阵")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("手眼标定矩阵最后一行无效")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("手眼标定旋转矩阵不是正交矩阵")
    return transform


def estimate_target_to_camera(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_size_m: float,
) -> np.ndarray:
    """Estimate the square marker frame in OpenCV camera coordinates."""
    if marker_size_m <= 0:
        raise ValueError("ArUco 边长必须为正数")
    half = marker_size_m / 2.0
    # Required order for SOLVEPNP_IPPE_SQUARE and ArUco's TL, TR, BR, BL order.
    object_points = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        np.asarray(corners, dtype=np.float64).reshape(4, 1, 2),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok or float(tvec.reshape(3)[2]) <= 0:
        raise RuntimeError("ArUco PnP 位姿估计失败")
    rotation, _ = cv2.Rodrigues(rvec)
    return homogeneous(rotation, tvec)


def gripper_clearance_to_marker_plane(
    target_to_camera: np.ndarray, gripper_to_camera: np.ndarray
) -> float:
    """Return the perpendicular distance from gripper-frame origin to marker plane."""
    gripper_origin_camera = invert_transform(gripper_to_camera) @ np.asarray(
        [0.0, 0.0, 0.0, 1.0]
    )
    gripper_origin_target = invert_transform(target_to_camera) @ gripper_origin_camera
    return abs(float(gripper_origin_target[2]))


def approach_target_in_base(
    base_to_gripper: np.ndarray,
    gripper_to_camera: np.ndarray,
    target_to_camera: np.ndarray,
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return desired gripper point, marker origin, and current plane clearance."""
    if clearance_m <= 0:
        raise ValueError("安全间隙必须为正数")
    base_to_target = base_to_gripper @ gripper_to_camera @ target_to_camera
    gripper_point = base_to_gripper[:3, 3]
    marker_point = base_to_target[:3, 3]
    normal = base_to_target[:3, 2]
    side = 1.0 if float(np.dot(gripper_point - marker_point, normal)) >= 0 else -1.0
    desired = marker_point + side * clearance_m * normal
    plane_clearance = abs(float(np.dot(gripper_point - marker_point, normal)))
    return desired, marker_point, plane_clearance


def region_error(center: Sequence[float], image_size: Sequence[int]) -> float:
    width, height = (float(value) for value in image_size)
    x, y = float(center[0]) / width, float(center[1]) / height
    dx = max(0.0, 0.4 - x, x - 0.6)
    dy = max(0.0, 0.4 - y, y - 0.6)
    return math.hypot(dx, dy)


def next_adaptive_step_m(
    base_step_m: float,
    current_step_m: float,
    made_progress: bool,
    max_step_m: float = 0.03,
) -> float:
    """Increase a quantized no-motion command, then reset after useful motion."""
    if base_step_m <= 0 or current_step_m <= 0 or max_step_m < base_step_m:
        raise ValueError("自适应步长参数无效")
    if made_progress:
        return float(base_step_m)
    return min(float(max_step_m), max(current_step_m * 1.5, current_step_m + base_step_m))


def limit_approach_step_by_plane(
    requested_m: float, plane_clearance_m: float, target_clearance_m: float
) -> float:
    """Limit the final approach by the remaining normal distance to the plane."""
    plane_margin_m = max(0.0, plane_clearance_m - target_clearance_m)
    return min(requested_m, plane_margin_m)


def center_axis(direction: str) -> str:
    if direction in {"left", "right"}:
        return "horizontal"
    if direction in {"up", "down"}:
        return "vertical"
    raise ValueError("居中方向必须是 left/right/up/down")


def center_command_direction(direction: str, reversed_axis: bool) -> str:
    center_axis(direction)
    return CENTER_OPPOSITE[direction] if reversed_axis else direction


def reverse_raw_step_target(
    start_raw: Sequence[int],
    bad_raw: Sequence[int],
    multiple: int,
) -> tuple[int, ...]:
    """Continue opposite a measured bad joint step, keeping the gripper fixed."""
    if len(start_raw) != 6 or len(bad_raw) != 6 or multiple < 1:
        raise ValueError("关节反向恢复参数无效")
    target = [
        int(round(int(bad_raw[index]) - multiple * (int(bad_raw[index]) - int(start_raw[index]))))
        for index in range(5)
    ]
    target.append(int(bad_raw[5]))
    return tuple(target)


def center_raw_recovery_target(
    actual_raw: Sequence[int], direction: str, vertical_step_raw: int = 30
) -> tuple[int, ...]:
    """Use the empirically stable pan/wrist joints after coupled IK moves badly."""
    if (
        len(actual_raw) != 6
        or direction not in CENTER_OPPOSITE
        or not 1 <= vertical_step_raw <= 30
    ):
        raise ValueError("直接居中恢复参数无效")
    target = [int(value) for value in actual_raw]
    if direction == "right":
        target[0] += 8
    elif direction == "left":
        target[0] -= 8
    elif direction == "up":
        target[3] -= vertical_step_raw
    else:
        target[3] += vertical_step_raw
    return tuple(target)


def parse_raw_pose(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("启动姿态必须是六个逗号分隔整数") from exc
    if len(result) != 6:
        raise argparse.ArgumentTypeError("启动姿态必须包含 ID1~ID6 六个值")
    return result


def validate_branch_candidate(
    reference_raw: Sequence[int],
    candidate_raw: Sequence[int],
    elbow_margin_raw: int = DEFAULT_ELBOW_BRANCH_MARGIN_RAW,
) -> None:
    """Reject the observed folded-elbow branch while leaving the target visual."""
    if (
        len(reference_raw) != 6
        or len(candidate_raw) not in {5, 6}
        or elbow_margin_raw < 1
    ):
        raise ValueError("构型分支保护参数无效")
    elbow_limit = int(reference_raw[2]) + int(elbow_margin_raw)
    if int(candidate_raw[2]) > elbow_limit:
        raise RuntimeError(
            "规划将进入折叠构型："
            f"ID3={int(candidate_raw[2])}，启动分支上限={elbow_limit}；已拒绝运动"
        )


def move_to_startup_raw(
    driver: SCS215ArmDriver,
    mappings: Sequence,
    target: Sequence[int],
    transition_s: float = 3.0,
) -> None:
    """Move from a gravity-sagged pose to a known visible pose without unloading."""
    if len(target) != 6 or transition_s <= 0:
        raise ValueError("启动姿态或过渡时间无效")
    start = driver.bus.positions()
    if abs(int(target[0]) - int(start[0])) > 511:
        raise RuntimeError("启动姿态的 ID1 路径会跨越 0/1022 接缝")
    for index, (value, safe_range) in enumerate(
        zip(target[:5], driver.safe_raw_ranges, strict=True), start=1
    ):
        if not safe_range[0] <= int(value) <= safe_range[1]:
            raise ValueError(
                f"启动姿态 ID{index}={value} 超出安全范围 {safe_range[0]}~{safe_range[1]}"
            )
    gripper = driver.states[5]
    if not gripper.minimum <= int(target[5]) <= gripper.maximum:
        raise ValueError("启动姿态 ID6 超出 EEPROM 限位")
    largest = max(abs(int(b) - int(a)) for a, b in zip(start, target, strict=True))
    steps = max(1, math.ceil(transition_s / 0.05), math.ceil(largest / 4.0))
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        phase = index / steps
        eased = phase * phase * (3.0 - 2.0 * phase)
        raw = tuple(
            int(round(a + (b - a) * eased))
            for a, b in zip(start, target, strict=True)
        )
        joints = tuple(
            mapping.raw_to_degrees(value)
            for mapping, value in zip(mappings, raw[:5], strict=True)
        )
        driver.send_joint_target(joints, float(raw[5]))
        deadline += transition_s / steps
        time.sleep(max(0.0, deadline - time.monotonic()))
    verify_last_target_with_retry(driver)


def camera_delta_to_base(
    base_to_gripper: np.ndarray,
    gripper_to_camera: np.ndarray,
    direction: str,
    step_m: float,
) -> np.ndarray:
    if direction not in CAMERA_DELTAS or step_m <= 0:
        raise ValueError("相机移动方向未知或步长无效")
    camera_delta = np.asarray(CAMERA_DELTAS[direction], dtype=float) * step_m
    return base_to_gripper[:3, :3] @ gripper_to_camera[:3, :3] @ camera_delta


def dominant_base_direction(delta_base: Sequence[float]) -> str:
    delta = np.asarray(delta_base, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("基座坐标增量必须是有限三维向量")
    return max(DIRECTIONS, key=lambda name: float(np.dot(delta, DIRECTIONS[name])))


def verify_last_target_with_retry(
    driver: SCS215ArmDriver, attempts: int = 3
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Retry only transient serial framing/time-out failures during motion."""
    transient_text = ("无回复", "回复头不完整", "回复长度不足", "回复校验失败")
    for attempt in range(1, attempts + 1):
        try:
            return driver.verify_last_target()
        except RuntimeError as exc:
            if attempt >= attempts or not any(text in str(exc) for text in transient_text):
                raise
            print(f"目标回读遇到瞬时串口错误（{attempt}/{attempts}）：{exc}；正在重试。")
            time.sleep(0.25)
    raise AssertionError("unreachable")


def annotate(observation: Observation) -> np.ndarray:
    image = observation.frame.copy()
    height, width = image.shape[:2]
    cv2.aruco.drawDetectedMarkers(
        image,
        [observation.corners.reshape(1, 4, 2).astype(np.float32)],
        np.asarray([[observation.marker_id]], dtype=np.int32),
    )
    for fraction in (0.4, 0.6):
        cv2.line(image, (int(width * fraction), 0), (int(width * fraction), height), (0, 180, 255), 2)
        cv2.line(image, (0, int(height * fraction)), (width, int(height * fraction)), (0, 180, 255), 2)
    center = tuple(int(round(value)) for value in observation.center)
    cv2.circle(image, center, 9, (0, 0, 255), -1)
    cv2.putText(
        image,
        (
            f"ID {observation.marker_id}  {observation.direction}  "
            f"plane={observation.gripper_plane_clearance_m * 1000:.0f}mm"
        ),
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (40, 255, 40),
        3,
        cv2.LINE_AA,
    )
    return image


def observe(
    capture: cv2.VideoCapture,
    rotate: int,
    dictionary: str,
    marker_id: int | None,
    step_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_size_m: float,
    hand_eye: np.ndarray,
    expected_center: Sequence[float] | None = None,
) -> Observation:
    frame = None
    marker = None
    fallback = None
    fallback_frame = None
    last_error: ValueError | None = None
    for _ in range(4):
        frame = read_fresh_frame(capture, rotate)
        markers = detect_aruco(frame, dictionary)
        try:
            marker = select_marker(markers, marker_id)
            break
        except ValueError as exc:
            last_error = exc
        if expected_center is not None and len(markers) == 1:
            candidate = markers[0]
            jump = float(np.linalg.norm(np.asarray(candidate.center) - expected_center))
            if jump <= 400.0:
                fallback = candidate
                fallback_frame = frame
    if marker is None and fallback is not None:
        marker = fallback
        frame = fallback_frame
        print(
            f"目标 ID 本帧误解码为 {marker.marker_id}；"
            "因其是唯一候选且位置连续，继续使用其角点。"
        )
    if marker is None or frame is None:
        if frame is None:
            raise RuntimeError("摄像头读取失败")
        message = str(last_error) if last_error is not None else "未检测到 ArUco 码"
        raise MarkerLostError(message, frame)
    height, width = frame.shape[:2]
    plan = simple_servo_plan(marker.center, (width, height), step_m)
    target_to_camera = estimate_target_to_camera(
        marker.corners, camera_matrix, distortion, marker_size_m
    )
    plane_clearance = gripper_clearance_to_marker_plane(target_to_camera, hand_eye)
    direction = str(plan["direction"])
    return Observation(
        frame=frame,
        marker_id=marker_id if marker_id is not None else marker.marker_id,
        center=marker.center,
        corners=marker.corners,
        direction=direction,
        region_error=region_error(marker.center, (width, height)),
        area_px2=abs(float(cv2.contourArea(marker.corners.astype(np.float32)))),
        target_to_camera=target_to_camera,
        camera_depth_m=float(target_to_camera[2, 3]),
        gripper_plane_clearance_m=plane_clearance,
    )


def make_controller(
    kinematics: URDFKinematics,
    driver,
    limits: Sequence[tuple[float, float]],
    args: argparse.Namespace,
) -> CartesianController:
    return CartesianController(
        kinematics,
        driver,
        joint_limits_deg=limits,
        cartesian_resolution_m=args.resolution_mm / 1000.0,
        tolerance_m=args.tolerance_mm / 1000.0,
        max_joint_step_deg=args.max_joint_step,
        command_interval_s=args.interval,
    )


def plan_step(
    kinematics: URDFKinematics,
    joints_deg: Sequence[float],
    gripper_raw: float,
    limits: Sequence[tuple[float, float]],
    hand_eye: np.ndarray,
    direction: str,
    step_m: float,
    args: argparse.Namespace,
):
    base_to_gripper = kinematics.forward(joints_deg)
    delta_base = camera_delta_to_base(base_to_gripper, hand_eye, direction, step_m)
    target = base_to_gripper[:3, 3] + delta_base
    dominant = dominant_base_direction(delta_base)
    dry_driver = DryRunDriver(joints_deg, gripper_raw)
    dry_controller = make_controller(kinematics, dry_driver, limits, args)
    dry_controller.initialize()
    result = dry_controller.move_to_xyz(
        target,
        joint_motion_weights=VISUAL_JOINT_MOTION_WEIGHTS.get(dominant),
    )
    return delta_base, target, dominant, result, dry_driver.joints_deg


def plan_approach_step(
    kinematics: URDFKinematics,
    joints_deg: Sequence[float],
    gripper_raw: float,
    limits: Sequence[tuple[float, float]],
    hand_eye: np.ndarray,
    observation: Observation,
    clearance_m: float,
    max_step_m: float,
    args: argparse.Namespace,
):
    base_to_gripper = kinematics.forward(joints_deg)
    desired, marker_base, plane_clearance = approach_target_in_base(
        base_to_gripper,
        hand_eye,
        observation.target_to_camera,
        clearance_m,
    )
    error = desired - base_to_gripper[:3, 3]
    remaining = float(np.linalg.norm(error))
    if remaining <= 0:
        delta_base = np.zeros(3, dtype=float)
        target = base_to_gripper[:3, 3].copy()
    else:
        delta_base = error / remaining * min(max_step_m, remaining)
        target = base_to_gripper[:3, 3] + delta_base
    dominant = dominant_base_direction(delta_base if remaining > 0 else [1.0, 0.0, 0.0])
    dry_driver = DryRunDriver(joints_deg, gripper_raw)
    dry_controller = make_controller(kinematics, dry_driver, limits, args)
    dry_controller.initialize()
    result = dry_controller.move_to_xyz(
        target,
        joint_motion_weights=APPROACH_JOINT_MOTION_WEIGHTS,
    )
    return (
        delta_base,
        target,
        dominant,
        result,
        dry_driver.joints_deg,
        desired,
        marker_base,
        remaining,
        plane_clearance,
    )


def plan_hybrid_step(
    kinematics: URDFKinematics,
    joints_deg: Sequence[float],
    gripper_raw: float,
    limits: Sequence[tuple[float, float]],
    hand_eye: np.ndarray,
    observation: Observation,
    clearance_m: float,
    max_step_m: float,
    args: argparse.Namespace,
):
    """Keep the marker in view, then approach its calibrated 3-D standoff point."""
    base_to_gripper = kinematics.forward(joints_deg)
    desired, marker_base, plane_clearance = approach_target_in_base(
        base_to_gripper, hand_eye, observation.target_to_camera, clearance_m
    )
    remaining = float(np.linalg.norm(desired - base_to_gripper[:3, 3]))
    if observation.direction == "forward":
        planned = plan_approach_step(
            kinematics,
            joints_deg,
            gripper_raw,
            limits,
            hand_eye,
            observation,
            clearance_m,
            max_step_m,
            args,
        )
        return ("approach", *planned)

    delta, target, dominant, result, dry_joints = plan_step(
        kinematics,
        joints_deg,
        gripper_raw,
        limits,
        hand_eye,
        observation.direction,
        max_step_m,
        args,
    )
    return (
        "center",
        delta,
        target,
        dominant,
        result,
        dry_joints,
        desired,
        marker_base,
        remaining,
        plane_clearance,
    )


def state_payload(observation: Observation) -> dict:
    return {
        "marker_id": observation.marker_id,
        "center_px": [float(value) for value in observation.center],
        "direction": observation.direction,
        "region_error": observation.region_error,
        "area_px2": observation.area_px2,
        "camera_depth_m": observation.camera_depth_m,
        "gripper_plane_clearance_m": observation.gripper_plane_clearance_m,
    }


def write_run_log(output_dir: Path, log: dict) -> None:
    (output_dir / "run_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def execute(args: argparse.Namespace) -> int:
    if args.check == args.real:
        raise ValueError("必须且只能选择 --check 或 --real")
    if not 0.2 <= args.step_cm <= 1.0:
        raise ValueError("简单版步长必须在 0.2~1.0 cm；默认使用 1.0 cm")
    if not 1 <= args.max_moves <= 60:
        raise ValueError("最大动作数必须在 1~60 之间")
    if not 20.0 <= args.emergency_max_total_cm <= 50.0:
        raise ValueError("故障兜底总行程必须在 20~50 cm 之间")
    if not 5.0 <= args.marker_size_mm <= 100.0:
        raise ValueError("ArUco 边长必须在 5~100 mm 之间")
    if not 5.0 <= args.clearance_mm <= 50.0:
        raise ValueError("末端安全间隙必须在 5~50 mm 之间")
    if not 1.0 <= args.goal_tolerance_mm <= 5.0:
        raise ValueError("目标容差必须在 1~5 mm 之间")
    if not 20 <= args.elbow_branch_margin_raw <= 150:
        raise ValueError("肘部分支余量必须在 20~150 raw 之间")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hand_eye = load_camera_to_gripper(Path(args.hand_eye))
    camera_matrix, distortion, _ = load_camera_calibration(Path(args.camera_calibration))
    kinematics = URDFKinematics(args.urdf)
    mappings = load_mappings(Path(args.mapping))
    port = find_port(args.port)
    capture = open_camera(args.camera, args.width, args.height)
    log = {
        "schema": 1,
        "mode": "real" if args.real else "check",
        "camera_rotate_deg": args.rotate,
        "step_cm": args.step_cm,
        "marker_size_mm": args.marker_size_mm,
        "clearance_mm": args.clearance_mm,
        "elbow_branch_margin_raw": args.elbow_branch_margin_raw,
        "emergency_max_total_cm": args.emergency_max_total_cm,
        "moves": [],
    }
    try:
        with SCS215Bus(port) as bus:
            states = bus.snapshot()
            print_states(states)
            validate_models(states)
            driver = SCS215ArmDriver(bus, states, mappings, kinematics.joint_limits_deg)
            driver.validate_current()
            if args.startup_raw is not None:
                if not args.real:
                    raise ValueError("--startup-raw 只允许与 --real 同时使用")
                driver.hold_then_enable()
                try:
                    move_to_startup_raw(driver, mappings, args.startup_raw)
                    time.sleep(args.settle_sec)
                    print(f"已在保持扭矩下恢复启动姿态：{args.startup_raw}")
                except Exception:
                    driver.shutdown()
                    raise
            # --startup-raw may already have enabled torque; any read, planning or
            # preflight failure below must not leave the arm torqued on exit.
            try:
                branch_reference_raw = driver.bus.positions()
                log["branch_reference_raw"] = list(branch_reference_raw)
                print(
                    "自主接近构型保护已启用："
                    f"以当前初始姿态为分支参考，ID3 不得超过 "
                    f"{branch_reference_raw[2] + args.elbow_branch_margin_raw}。"
                )
                first = observe(
                    capture,
                    args.rotate,
                    args.dictionary,
                    args.id,
                    args.step_cm / 100.0,
                    camera_matrix,
                    distortion,
                    args.marker_size_mm / 1000.0,
                    hand_eye,
                )
                _write_image(output_dir / "observation_000.png", annotate(first))
                print(
                    f"ArUco ID {first.marker_id}: center=({first.center[0]:.1f}, {first.center[1]:.1f}), "
                    f"region={first.direction}, camera_depth={first.camera_depth_m * 1000:.1f} mm, "
                    f"plane_clearance={first.gripper_plane_clearance_m * 1000:.1f} mm"
                )
                joints, gripper_raw = driver.get_joint_state()
                (
                    motion_mode,
                    delta,
                    target,
                    dominant,
                    dry_result,
                    dry_joints,
                    desired,
                    marker_base,
                    remaining,
                    plane_clearance,
                ) = plan_hybrid_step(
                    kinematics,
                    joints,
                    gripper_raw,
                    driver.effective_joint_limits_deg,
                    hand_eye,
                    first,
                    args.clearance_mm / 1000.0,
                    args.step_cm / 100.0,
                    args,
                )
                dry_raw = tuple(
                    mapping.degrees_to_raw(angle)
                    for mapping, angle in zip(mappings, dry_joints, strict=True)
                )
                validate_branch_candidate(
                    branch_reference_raw,
                    dry_raw,
                    args.elbow_branch_margin_raw,
                )
                print(
                    f"首步离线规划：{motion_mode}"
                    + f", base_delta_mm={[round(value * 1000, 2) for value in delta]}, "
                    + f"dominant={dominant}, reachable={dry_result.executed_m * 1000:.1f}/"
                    + f"{dry_result.requested_m * 1000:.1f} mm, remaining={remaining * 1000:.1f} mm, "
                    + f"plane={plane_clearance * 1000:.1f} mm, final_raw={dry_raw}"
                )
                if remaining > args.goal_tolerance_mm / 1000.0 and dry_result.executed_m < args.min_partial_mm / 1000.0:
                    raise RuntimeError("首步安全可达距离太短，拒绝启动")
                if args.check:
                    log["preflight"] = {
                        "observation": state_payload(first),
                        "motion_mode": motion_mode,
                        "base_delta_m": delta.tolist(),
                        "target_base_m": target.tolist(),
                        "desired_gripper_base_m": desired.tolist(),
                        "marker_origin_base_m": marker_base.tolist(),
                        "remaining_to_standoff_m": remaining,
                        "plane_clearance_m": plane_clearance,
                        "dominant_base_direction": dominant,
                        "planned_executed_m": dry_result.executed_m,
                        "planned_full_success": dry_result.full_success,
                        "planned_final_raw": list(dry_raw),
                    }
                    write_run_log(output_dir, log)
                    print("只读预检完成：未开启扭矩、未发送运动指令。")
                    return 0
            except BaseException:
                driver.shutdown()
                raise

            if not driver.enabled:
                driver.hold_then_enable()
            print("保持目标已回读，六个舵机扭矩已开启。")
            controller = make_controller(kinematics, driver, driver.effective_joint_limits_deg, args)
            controller.initialize()
            observation = first
            total_m = 0.0
            stalled_moves = 0
            base_requested_m = args.step_cm / 100.0
            base_requested_by_mode = {
                "center": base_requested_m,
                # This arm's loaded shoulder/wrist deadband absorbs 5-10 mm
                # Cartesian requests.  Fifteen millimetres remains effective,
                # while 1 mm waypoints at 25 Hz reduce supply-current peaks.
                "approach": max(base_requested_m, 0.015),
            }
            adaptive_requested_m = {
                mode: requested for mode, requested in base_requested_by_mode.items()
            }
            reversed_center_axes = {"horizontal": False, "vertical": False}
            direct_center_axes = {"horizontal": False, "vertical": False}
            try:
                for move_index in range(1, args.max_moves + 1):
                    expected_motion_mode = (
                        "approach" if observation.direction == "forward" else "center"
                    )
                    requested_m = adaptive_requested_m[expected_motion_mode]
                    if expected_motion_mode == "approach":
                        target_clearance_m = args.clearance_mm / 1000.0
                        goal_tolerance_m = args.goal_tolerance_mm / 1000.0
                        if (
                            observation.gripper_plane_clearance_m
                            <= target_clearance_m + goal_tolerance_m
                        ):
                            print(
                                f"已到目标前约 {args.clearance_mm:.1f} mm 终止平面，停止。"
                            )
                            break
                        requested_m = limit_approach_step_by_plane(
                            requested_m,
                            observation.gripper_plane_clearance_m,
                            target_clearance_m,
                        )
                    if total_m + requested_m > args.emergency_max_total_cm / 100.0 + 1e-9:
                        print("已到故障兜底总行程上限，停止。")
                        break
                    current_joints, current_gripper = driver.get_joint_state()
                    controller.joints_deg = current_joints
                    controller.gripper_raw = current_gripper
                    if observation.direction != "forward":
                        axis = center_axis(observation.direction)
                        if direct_center_axes[axis]:
                            start_raw = driver.bus.positions()
                            vertical_step_raw = 30
                            if axis == "vertical":
                                height = observation.frame.shape[0]
                                if observation.direction == "up":
                                    pixel_error = max(
                                        0.0, height * 0.42 - observation.center[1]
                                    )
                                else:
                                    pixel_error = max(
                                        0.0, observation.center[1] - height * 0.58
                                    )
                                vertical_step_raw = max(
                                    12, min(30, int(math.ceil(pixel_error / 6.0)))
                                )
                            recovery_target = center_raw_recovery_target(
                                start_raw,
                                observation.direction,
                                vertical_step_raw=vertical_step_raw,
                            )
                            validate_branch_candidate(
                                branch_reference_raw,
                                recovery_target,
                                args.elbow_branch_margin_raw,
                            )
                            start_xyz = np.asarray(kinematics.position(current_joints))
                            base_to_gripper = kinematics.forward(current_joints)
                            desired, marker_base, plane_clearance = approach_target_in_base(
                                base_to_gripper,
                                hand_eye,
                                observation.target_to_camera,
                                args.clearance_mm / 1000.0,
                            )
                            remaining = float(np.linalg.norm(desired - start_xyz))
                            move_to_startup_raw(
                                driver,
                                mappings,
                                recovery_target,
                                transition_s=0.6,
                            )
                            time.sleep(min(args.settle_sec, 0.5))
                            actual_raw = driver.bus.positions()
                            actual_joints, actual_gripper = driver.get_joint_state()
                            controller.joints_deg = actual_joints
                            controller.gripper_raw = actual_gripper
                            actual_xyz = np.asarray(kinematics.position(actual_joints))
                            actual_progress = float(np.linalg.norm(actual_xyz - start_xyz))
                            total_m += actual_progress
                            after = observe(
                                capture,
                                args.rotate,
                                args.dictionary,
                                first.marker_id,
                                requested_m,
                                camera_matrix,
                                distortion,
                                args.marker_size_mm / 1000.0,
                                hand_eye,
                                expected_center=observation.center,
                            )
                            after_desired, _, after_plane_clearance = approach_target_in_base(
                                kinematics.forward(actual_joints),
                                hand_eye,
                                after.target_to_camera,
                                args.clearance_mm / 1000.0,
                            )
                            after_remaining = float(
                                np.linalg.norm(after_desired - actual_xyz)
                            )
                            _write_image(
                                output_dir / f"observation_{move_index:03d}.png",
                                annotate(after),
                            )
                            controlled_joint = 0 if axis == "horizontal" else 3
                            raw_motion = abs(
                                int(actual_raw[controlled_joint])
                                - int(start_raw[controlled_joint])
                            )
                            if observation.direction == "up":
                                visual_improvement = (
                                    after.center[1] - observation.center[1]
                                ) / observation.frame.shape[0]
                            elif observation.direction == "down":
                                visual_improvement = (
                                    observation.center[1] - after.center[1]
                                ) / observation.frame.shape[0]
                            elif observation.direction == "right":
                                visual_improvement = (
                                    observation.center[0] - after.center[0]
                                ) / observation.frame.shape[1]
                            else:
                                visual_improvement = (
                                    after.center[0] - observation.center[0]
                                ) / observation.frame.shape[1]
                            log["moves"].append(
                                {
                                    "index": move_index,
                                    "motion_mode": "center_raw",
                                    "requested_step_m": requested_m,
                                    "commanded_center_direction": observation.direction,
                                    "before": state_payload(observation),
                                    "after": state_payload(after),
                                    "base_delta_requested_m": [0.0, 0.0, 0.0],
                                    "dominant_base_direction": f"raw_{axis}",
                                    "planned_full_success": True,
                                    "planned_executed_m": actual_progress,
                                    "actual_progress_m": actual_progress,
                                    "actual_delta_base_m": (actual_xyz - start_xyz).tolist(),
                                    "desired_gripper_base_m": desired.tolist(),
                                    "marker_origin_base_m": marker_base.tolist(),
                                    "remaining_before_m": remaining,
                                    "remaining_after_m": after_remaining,
                                    "plane_clearance_before_m": plane_clearance,
                                    "plane_clearance_after_m": after_plane_clearance,
                                    "actual_raw": list(actual_raw),
                                    "visual_recovery": {
                                        "method": "persistent_stable_single_joint",
                                        "target_raw": list(recovery_target),
                                        "controlled_joint_raw_motion": raw_motion,
                                    },
                                }
                            )
                            write_run_log(output_dir, log)
                            print(
                                f"步 {move_index}: center_raw/{axis}, raw_motion={raw_motion}, "
                                f"region {observation.region_error:.3f}->{after.region_error:.3f}, "
                                f"remaining {remaining * 1000:.1f}->{after_remaining * 1000:.1f} mm"
                            )
                            if after.direction == "forward" or (
                                after.direction != "forward"
                                and center_axis(after.direction) != axis
                            ):
                                direct_center_axes[axis] = False
                            elif raw_motion < 2 and visual_improvement <= 0:
                                raise RuntimeError(
                                    f"ID{controlled_joint + 1} 单关节居中目标未产生有效回读运动"
                                )
                            elif visual_improvement < -0.01:
                                raise RuntimeError("单关节居中使图像误差增大，安全停止")
                            observation = after
                            continue
                    planning_observation = observation
                    commanded_center_direction = observation.direction
                    if observation.direction != "forward":
                        axis = center_axis(observation.direction)
                        commanded_center_direction = center_command_direction(
                            observation.direction, reversed_center_axes[axis]
                        )
                        planning_observation = replace(
                            observation, direction=commanded_center_direction
                        )
                    (
                        motion_mode,
                        delta,
                        target,
                        dominant,
                        dry_result,
                        planned_joints,
                        desired,
                        marker_base,
                        remaining,
                        plane_clearance,
                    ) = plan_hybrid_step(
                        kinematics,
                        current_joints,
                        current_gripper,
                        driver.effective_joint_limits_deg,
                        hand_eye,
                        planning_observation,
                        args.clearance_mm / 1000.0,
                        requested_m,
                        args,
                    )
                    planned_raw = tuple(
                        mapping.degrees_to_raw(angle)
                        for mapping, angle in zip(mappings, planned_joints, strict=True)
                    )
                    validate_branch_candidate(
                        branch_reference_raw,
                        planned_raw,
                        args.elbow_branch_margin_raw,
                    )
                    if motion_mode == "approach" and remaining <= args.goal_tolerance_mm / 1000.0:
                        print(
                            f"已到目标前安全间隙：剩余位置误差 {remaining * 1000:.1f} mm，停止。"
                        )
                        break
                    minimum_planned_m = min(
                        args.min_partial_mm / 1000.0, requested_m * 0.5
                    )
                    if dry_result.executed_m < minimum_planned_m:
                        adaptive_requested_m[motion_mode] = next_adaptive_step_m(
                            base_requested_by_mode[motion_mode], requested_m, False
                        )
                        if observation.direction != "forward":
                            axis = center_axis(observation.direction)
                            reversed_center_axes[axis] = not reversed_center_axes[axis]
                            adaptive_requested_m["center"] = min(base_requested_m, 0.002)
                            print(
                                "当前居中方向安全可达距离过短；已换向，"
                                "下一次使用 2 mm 微步。"
                            )
                        else:
                            print(
                                "当前接近步落入可达/量化死区；"
                                f"下一次自适应为 "
                                f"{adaptive_requested_m['approach'] * 1000:.0f} mm。"
                            )
                        continue
                    # Camera-frame moves generally combine several base axes.  The
                    # legacy one-axis load-bias table can therefore point away from
                    # this calibrated vector; send the nominal IK path unchanged.
                    start_raw = driver.bus.positions()
                    motion_weights = (
                        APPROACH_JOINT_MOTION_WEIGHTS
                        if motion_mode == "approach"
                        else VISUAL_JOINT_MOTION_WEIGHTS.get(dominant)
                    )
                    result = controller.move_to_xyz(
                        target,
                        stream_targets=True,
                        joint_motion_weights=motion_weights,
                    )
                    if driver.last_targets is None:
                        raise RuntimeError("IK 未生成任何可发送路点")
                    # Give the servos a short head start before querying six devices.
                    # Some SCS215 chains briefly miss the first read while motion begins.
                    time.sleep(0.15)
                    _, _, actual_raw = verify_last_target_with_retry(driver)
                    validate_branch_candidate(
                        branch_reference_raw,
                        actual_raw,
                        args.elbow_branch_margin_raw,
                    )
                    commanded_raw = driver.last_targets
                    if commanded_raw is None:
                        raise RuntimeError("运动后缺少最终舵机目标")
                    raw_motion_max = max(
                        abs(int(after_raw) - int(before_raw))
                        for before_raw, after_raw in zip(
                            start_raw[:5], actual_raw[:5], strict=True
                        )
                    )
                    actual_joints, actual_gripper = driver.get_joint_state()
                    controller.joints_deg = actual_joints
                    controller.gripper_raw = actual_gripper
                    actual_xyz = np.asarray(kinematics.position(actual_joints))
                    start_xyz = np.asarray(result.start_xyz)
                    intended = delta / float(np.linalg.norm(delta))
                    actual_progress = max(0.0, float(np.dot(actual_xyz - start_xyz, intended)))
                    time.sleep(args.settle_sec)
                    after = observe(
                        capture,
                        args.rotate,
                        args.dictionary,
                        first.marker_id,
                        requested_m,
                        camera_matrix,
                        distortion,
                        args.marker_size_mm / 1000.0,
                        hand_eye,
                        expected_center=observation.center,
                    )
                    recovery_info = None
                    initial_visual_improvement = observation.region_error - after.region_error
                    if motion_mode == "center" and initial_visual_improvement < -0.01:
                        raw_delta = tuple(
                            int(actual_raw[index]) - int(start_raw[index])
                            for index in range(5)
                        )
                        if any(raw_delta):
                            _write_image(
                                output_dir / f"bad_center_{move_index:03d}.png",
                                annotate(after),
                            )
                            recovery_info = {
                                "trigger_center_px": [float(value) for value in after.center],
                                "trigger_region_error": after.region_error,
                                "start_raw": list(start_raw),
                                "bad_actual_raw": list(actual_raw),
                                "bad_raw_delta": list(raw_delta),
                                "samples": [],
                            }
                            print(
                                "居中响应反向；避开受载关节，使用稳定的单关节图像补偿。"
                            )
                            recovery_target = center_raw_recovery_target(
                                actual_raw, after.direction
                            )
                            validate_branch_candidate(
                                branch_reference_raw,
                                recovery_target,
                                args.elbow_branch_margin_raw,
                            )
                            move_to_startup_raw(
                                driver,
                                mappings,
                                recovery_target,
                                transition_s=0.5,
                            )
                            time.sleep(min(args.settle_sec, 0.5))
                            after = observe(
                                capture,
                                args.rotate,
                                args.dictionary,
                                first.marker_id,
                                requested_m,
                                camera_matrix,
                                distortion,
                                args.marker_size_mm / 1000.0,
                                hand_eye,
                                expected_center=observation.center,
                            )
                            _write_image(
                                output_dir / f"recovery_{move_index:03d}.png",
                                annotate(after),
                            )
                            recovery_info["samples"].append(
                                {
                                    "method": "stable_single_joint",
                                    "target_raw": list(recovery_target),
                                    "observation": state_payload(after),
                                }
                            )
                            print(
                                f"  单关节补偿后: center=({after.center[0]:.0f}, {after.center[1]:.0f}), "
                                f"region={after.direction}；下一循环按新主误差轴规划。"
                            )
                            axis = center_axis(observation.direction)
                            direct_center_axes[axis] = after.direction != "forward"
                            actual_raw = driver.bus.positions()
                            actual_joints, actual_gripper = driver.get_joint_state()
                            controller.joints_deg = actual_joints
                            controller.gripper_raw = actual_gripper
                            actual_xyz = np.asarray(kinematics.position(actual_joints))
                            actual_progress = max(
                                0.0,
                                float(np.dot(actual_xyz - start_xyz, intended)),
                            )
                            recovery_info["final_raw"] = list(actual_raw)
                            recovery_info["final_observation"] = state_payload(after)
                    total_m += actual_progress
                    after_desired, _, after_plane_clearance = approach_target_in_base(
                        kinematics.forward(actual_joints),
                        hand_eye,
                        after.target_to_camera,
                        args.clearance_mm / 1000.0,
                    )
                    after_remaining = float(np.linalg.norm(after_desired - actual_xyz))
                    _write_image(output_dir / f"observation_{move_index:03d}.png", annotate(after))
                    move_record = {
                        "index": move_index,
                        "motion_mode": motion_mode,
                        "requested_step_m": requested_m,
                        "commanded_center_direction": (
                            commanded_center_direction if motion_mode == "center" else None
                        ),
                        "before": state_payload(observation),
                        "after": state_payload(after),
                        "base_delta_requested_m": delta.tolist(),
                        "dominant_base_direction": dominant,
                        "planned_full_success": result.full_success,
                        "planned_executed_m": result.executed_m,
                        "actual_progress_m": actual_progress,
                        "start_raw": list(start_raw),
                        "commanded_raw": list(commanded_raw),
                        "actual_raw_motion_max": raw_motion_max,
                        "actual_delta_base_m": (actual_xyz - start_xyz).tolist(),
                        "desired_gripper_base_m": desired.tolist(),
                        "marker_origin_base_m": marker_base.tolist(),
                        "remaining_before_m": remaining,
                        "remaining_after_m": after_remaining,
                        "plane_clearance_before_m": plane_clearance,
                        "plane_clearance_after_m": after_plane_clearance,
                        "actual_raw": list(actual_raw),
                        "visual_recovery": recovery_info,
                    }
                    log["moves"].append(move_record)
                    write_run_log(output_dir, log)
                    print(
                        f"步 {move_index}: {motion_mode}, actual={actual_progress * 1000:.1f} mm, "
                        f"raw_motion={raw_motion_max}, "
                        f"remaining {remaining * 1000:.1f}->{after_remaining * 1000:.1f} mm, "
                        f"plane {plane_clearance * 1000:.1f}->{after_plane_clearance * 1000:.1f} mm"
                    )
                    minimum_progress = min(
                        result.executed_m * 0.30, args.min_partial_mm / 1000.0
                    )
                    direction_reversed_this_step = False
                    if motion_mode == "center":
                        visual_improvement = observation.region_error - after.region_error
                        visual_threshold = 0.005
                        if visual_improvement < -0.01:
                            axis = center_axis(observation.direction)
                            if recovery_info is None:
                                reversed_center_axes[axis] = not reversed_center_axes[axis]
                                direction_reversed_this_step = True
                                print(
                                    f"{axis} 居中响应反向；后续自动切换控制方向。"
                                )
                            else:
                                print(
                                    f"{axis} 居中仍未完成；下一步继续单关节闭环补偿。"
                                )
                    else:
                        visual_improvement = remaining - after_remaining
                        visual_threshold = args.min_visual_improvement_mm / 1000.0
                    visual_success = (
                        after.region_error <= 1e-9
                        if motion_mode == "center"
                        else (
                            raw_motion_max >= 2
                            and visual_improvement >= visual_threshold
                        )
                    )
                    if (
                        actual_progress < minimum_progress
                        and not visual_success
                    ):
                        stalled_moves += 1
                        if direction_reversed_this_step:
                            adaptive_requested_m["center"] = min(base_requested_m, 0.002)
                            print("方向刚刚反转；下一步改用 2 mm 微步试探。")
                        elif motion_mode == "center":
                            adaptive_requested_m["center"] = base_requested_m
                            print(
                                f"本步居中响应较小（连续 {stalled_moves} 次）；"
                                "下一步仍保持 10 mm，避免放大姿态耦合。"
                            )
                        else:
                            adaptive_requested_m[motion_mode] = next_adaptive_step_m(
                                base_requested_by_mode[motion_mode], requested_m, False
                            )
                            print(
                                f"本步落入量化死区（连续 {stalled_moves} 次）；"
                                f"下一步 {motion_mode} 自适应为 "
                                f"{adaptive_requested_m[motion_mode] * 1000:.0f} mm。"
                            )
                    else:
                        stalled_moves = 0
                        adaptive_requested_m[motion_mode] = next_adaptive_step_m(
                            base_requested_by_mode[motion_mode], requested_m, True
                        )
                    if (
                        motion_mode == "approach"
                        and (
                            after_remaining <= args.goal_tolerance_mm / 1000.0
                            or after_plane_clearance
                            <= (args.clearance_mm + args.goal_tolerance_mm) / 1000.0
                        )
                    ):
                        print(
                            f"已到目标前约 {args.clearance_mm:.1f} mm 安全间隙，停止。"
                        )
                        observation = after
                        break
                    observation = after
            finally:
                driver.shutdown()
                print("六个舵机已卸力。")
    except Exception as exc:
        if isinstance(exc, MarkerLostError):
            _write_image(output_dir / "target_lost.png", exc.frame)
        log["termination"] = {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        capture.release()
        log.setdefault("termination", {"status": "completed"})
        write_run_log(output_dir, log)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SO-101 简单版 ArUco 视觉伺服")
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="只读相机/机械臂检查与首步规划")
    mode.add_argument("--real", action="store_true", help="启用受限真机闭环")
    result.add_argument("--port", help="CH343 串口；省略时自动识别唯一设备")
    result.add_argument("--camera", type=int, default=0)
    result.add_argument("--width", type=int, default=1920)
    result.add_argument("--height", type=int, default=1080)
    result.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    result.add_argument("--dictionary", default="4X4_50")
    result.add_argument("--id", type=int)
    result.add_argument("--step-cm", type=float, default=1.0)
    result.add_argument("--marker-size-mm", type=float, default=25.0)
    result.add_argument("--clearance-mm", type=float, default=10.0)
    result.add_argument("--goal-tolerance-mm", type=float, default=3.0)
    result.add_argument("--max-moves", type=int, default=60)
    result.add_argument(
        "--elbow-branch-margin-raw",
        type=int,
        default=DEFAULT_ELBOW_BRANCH_MARGIN_RAW,
        help="相对启动姿态允许 ID3 增大的最大 raw；用于拒绝折叠 IK 分支",
    )
    result.add_argument(
        "--emergency-max-total-cm",
        type=float,
        default=40.0,
        help="仅用于 PnP/状态异常时的故障兜底，不是正常停止条件",
    )
    result.add_argument("--min-partial-mm", type=float, default=3.0)
    result.add_argument("--min-visual-improvement-mm", type=float, default=1.0)
    result.add_argument("--settle-sec", type=float, default=0.35)
    result.add_argument(
        "--startup-raw",
        type=parse_raw_pose,
        help="真机启动时在保持扭矩下先平滑移动到六舵机原始位置，例如 414,457,305,524,483,150",
    )
    result.add_argument("--resolution-mm", type=float, default=1.0)
    result.add_argument("--tolerance-mm", type=float, default=0.8)
    result.add_argument("--max-joint-step", type=float, default=12.0)
    result.add_argument("--interval", type=float, default=1.0 / 25.0)
    result.add_argument("--urdf", default=str(DEFAULT_URDF))
    result.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    result.add_argument("--camera-calibration", default=str(DEFAULT_CAMERA_RESULT))
    result.add_argument("--hand-eye", default=str(DEFAULT_HAND_EYE_RESULT))
    result.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return result


def main() -> int:
    try:
        return execute(parser().parse_args())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
