"""Camera calibration, hand-eye calibration, and ArUco/IBVS helpers.

Every command in this file is read-only with respect to the robot.  It never
opens the servo bus or sends a joint command.  Camera images and recorded joint
angles are converted into calibration files or motion *plans* for later review.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
IK_DIR = ROOT_DIR / "SO101_SCS215_IK键盘直线控制"
DEFAULT_URDF = ROOT_DIR / "SO101" / "so101_new_calib.urdf"
DEFAULT_CAMERA_RESULT = PROJECT_DIR / "data" / "camera_calibration.json"
DEFAULT_HAND_EYE_RESULT = PROJECT_DIR / "data" / "hand_eye_calibration.json"


@dataclass(frozen=True)
class ChessboardObservation:
    path: Path
    corners: np.ndarray
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ArucoObservation:
    marker_id: int
    corners: np.ndarray

    @property
    def center(self) -> tuple[float, float]:
        point = np.mean(self.corners, axis=0)
        return float(point[0]), float(point[1])


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"无法编码图像：{path}")
    encoded.tofile(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return frame
    codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if degrees not in codes:
        raise ValueError("图像旋转角度只能是 0、90、180 或 270")
    return cv2.rotate(frame, codes[degrees])


def _matrix(values: Sequence[Sequence[float]], shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"矩阵应为 {shape} 且元素有限，实际为 {result.shape}")
    return result


def chessboard_object_points(
    columns: int, rows: int, square_size_m: float
) -> np.ndarray:
    """Return internal-corner coordinates in row-major order."""
    if columns < 2 or rows < 2 or square_size_m <= 0:
        raise ValueError("棋盘格内角点数量至少为 2x2，方格边长必须为正数")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * float(square_size_m)
    )
    return points


def find_chessboard(
    image: np.ndarray, columns: int, rows: int
) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (columns, rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )
    else:
        found, corners = False, None
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FILTER_QUADS
            ),
        )
        if found:
            cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
            )
    if not found:
        return None
    return np.asarray(corners, dtype=np.float32).reshape(-1, 2)


def canonicalize_chessboard_corners(corners: np.ndarray) -> np.ndarray:
    """Resolve the 180-degree ordering ambiguity of a symmetric checkerboard.

    During one hand-eye session the board remains roughly upright in the image,
    so the image-space upper-left extreme is a stable physical origin.  OpenCV's
    SB detector may otherwise return the exact reverse order on some frames.
    """
    ordered = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if float(ordered[0].sum()) > float(ordered[-1].sum()):
        ordered = ordered[::-1].copy()
    return ordered


def collect_chessboard_observations(
    paths: Iterable[Path], columns: int, rows: int
) -> tuple[list[ChessboardObservation], list[str]]:
    observations: list[ChessboardObservation] = []
    rejected: list[str] = []
    expected_size: tuple[int, int] | None = None
    for path in sorted(paths):
        image = _read_image(path)
        image_size = (image.shape[1], image.shape[0])
        if expected_size is None:
            expected_size = image_size
        if image_size != expected_size:
            rejected.append(f"{path}: 分辨率 {image_size} 与 {expected_size} 不一致")
            continue
        corners = find_chessboard(image, columns, rows)
        if corners is None:
            rejected.append(f"{path}: 未找到完整棋盘格")
            continue
        observations.append(ChessboardObservation(path, corners, image_size))
    return observations, rejected


def calibrate_camera(
    observations: Sequence[ChessboardObservation],
    columns: int,
    rows: int,
    square_size_m: float,
) -> dict:
    if len(observations) < 3:
        raise ValueError("至少需要 3 张有效图片；课堂作业建议拍摄并使用 15 张以上")
    image_size = observations[0].image_size
    if any(item.image_size != image_size for item in observations):
        raise ValueError("所有标定图片必须使用相同分辨率")
    object_template = chessboard_object_points(columns, rows, square_size_m)
    object_points = [object_template.copy() for _ in observations]
    image_points = [item.corners.reshape(-1, 1, 2) for item in observations]
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view_errors: list[float] = []
    for object_points_i, observed, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs, strict=True
    ):
        projected, _ = cv2.projectPoints(
            object_points_i, rvec, tvec, camera_matrix, distortion
        )
        error = cv2.norm(observed, projected, cv2.NORM_L2) / math.sqrt(
            len(projected)
        )
        per_view_errors.append(float(error))
    return {
        "schema": 1,
        "model": "opencv_pinhole",
        "image_size_px": list(image_size),
        "checkerboard_internal_corners": [columns, rows],
        "square_size_m": float(square_size_m),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "rms_reprojection_error_px": float(rms),
        "per_view_reprojection_error_px": per_view_errors,
        "used_images": [str(item.path) for item in observations],
    }


def load_camera_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera_matrix = _matrix(payload["camera_matrix"], (3, 3))
    distortion = np.asarray(
        payload["distortion_coefficients"], dtype=np.float64
    ).reshape(-1, 1)
    if distortion.size < 4 or not np.all(np.isfinite(distortion)):
        raise ValueError("畸变系数至少需要 4 个有限值")
    return camera_matrix, distortion, payload


def homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _matrix(rotation, (3, 3))
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(translation)):
        raise ValueError("平移向量必须为有限值")
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = _matrix(transform, (4, 4))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve_hand_eye(
    gripper_to_base: Sequence[np.ndarray],
    target_to_camera: Sequence[np.ndarray],
    method: str = "tsai",
) -> np.ndarray:
    if len(gripper_to_base) != len(target_to_camera) or len(gripper_to_base) < 3:
        raise ValueError("手眼标定需要至少 3 组一一对应的机械臂/图像姿态")
    if method not in HAND_EYE_METHODS:
        raise ValueError(f"未知手眼标定方法：{method}")
    robot = [_matrix(item, (4, 4)) for item in gripper_to_base]
    target = [_matrix(item, (4, 4)) for item in target_to_camera]
    rotation, translation = cv2.calibrateHandEye(
        [item[:3, :3] for item in robot],
        [item[:3, 3] for item in robot],
        [item[:3, :3] for item in target],
        [item[:3, 3] for item in target],
        method=HAND_EYE_METHODS[method],
    )
    return homogeneous(rotation, translation)


def _load_kinematics():
    if str(IK_DIR) not in sys.path:
        sys.path.insert(0, str(IK_DIR))
    from ik_core import URDFKinematics

    return URDFKinematics(DEFAULT_URDF)


def calibrate_hand_eye_from_manifest(
    manifest_path: Path,
    camera_path: Path,
    columns: int,
    rows: int,
    square_size_m: float,
    method: str,
) -> dict:
    camera_matrix, distortion, _ = load_camera_calibration(camera_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if len(samples) < 3:
        raise ValueError("清单至少需要 3 组样本；课堂作业建议采集 12 组以上")
    kinematics = _load_kinematics()
    object_points = chessboard_object_points(columns, rows, square_size_m)
    gripper_to_base: list[np.ndarray] = []
    target_to_camera: list[np.ndarray] = []
    used: list[str] = []
    rejected: list[str] = []
    for index, sample in enumerate(samples, 1):
        image_path = (manifest_path.parent / sample["image"]).resolve()
        joints = sample.get("joints_deg", [])
        if len(joints) != 5:
            rejected.append(f"样本 {index}: joints_deg 必须恰好包含 5 个关节角")
            continue
        image = _read_image(image_path)
        corners = find_chessboard(image, columns, rows)
        if corners is None:
            rejected.append(f"样本 {index}: {image_path} 未找到完整棋盘格")
            continue
        corners = canonicalize_chessboard_corners(corners)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            corners.reshape(-1, 1, 2),
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            rejected.append(f"样本 {index}: solvePnP 失败")
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        gripper_to_base.append(kinematics.forward(joints))
        target_to_camera.append(homogeneous(rotation, tvec))
        used.append(str(image_path))
    if len(used) < 3:
        raise ValueError("有效手眼标定样本少于 3 组")
    camera_to_gripper = solve_hand_eye(
        gripper_to_base, target_to_camera, method=method
    )
    base_to_target = [
        robot @ camera_to_gripper @ target
        for robot, target in zip(gripper_to_base, target_to_camera, strict=True)
    ]
    translations = np.asarray([item[:3, 3] for item in base_to_target])
    center = np.mean(translations, axis=0)
    per_axis_max = np.max(np.abs(translations - center), axis=0)
    return {
        "schema": 1,
        "method": method,
        "convention": "T_gripper_camera: camera coordinates to end-effector coordinates",
        "translation_unit": "m",
        "T_gripper_camera": camera_to_gripper.tolist(),
        "translation_gripper_camera_m": camera_to_gripper[:3, 3].tolist(),
        "rotation_gripper_camera": camera_to_gripper[:3, :3].tolist(),
        "board_origin_base_mean_m": center.tolist(),
        "board_origin_base_max_deviation_per_axis_m": per_axis_max.tolist(),
        "used_samples": used,
        "rejected_samples": rejected,
    }


def get_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 缺少 aruco 模块，请安装 opencv-contrib-python")
    attribute = f"DICT_{name.upper()}"
    if not hasattr(cv2.aruco, attribute):
        raise ValueError(f"未知 ArUco 字典：{name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, attribute))


def detect_aruco(
    image: np.ndarray, dictionary_name: str = "4X4_50"
) -> list[ArucoObservation]:
    detector = cv2.aruco.ArucoDetector(
        get_aruco_dictionary(dictionary_name), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return []
    return [
        ArucoObservation(int(marker_id), np.asarray(points).reshape(4, 2))
        for points, marker_id in zip(corners, ids.reshape(-1), strict=True)
    ]


def select_marker(
    observations: Sequence[ArucoObservation], marker_id: int | None
) -> ArucoObservation:
    candidates = [
        item for item in observations if marker_id is None or item.marker_id == marker_id
    ]
    if not candidates:
        requested = "任意 ID" if marker_id is None else f"ID {marker_id}"
        raise ValueError(f"图像中未检测到 {requested} 的 ArUco 码")
    if marker_id is None and len(candidates) > 1:
        raise ValueError("图像中有多个 ArUco 码，请使用 --id 指定目标")
    return candidates[0]


def simple_servo_plan(
    center: tuple[float, float], image_size: tuple[int, int], step_m: float = 0.01
) -> dict:
    width, height = image_size
    x, y = center
    if not (0 <= x < width and 0 <= y < height) or step_m <= 0:
        raise ValueError("中心点必须位于图像内，步长必须为正数")
    nx, ny = x / width, y / height
    # Correct the axis that is furthest outside the 40--60% target box.
    # A fixed horizontal-first priority can drive a low marker behind the
    # gripper before the more important vertical error has been corrected.
    outside = {
        "left": max(0.0, 0.4 - nx),
        "right": max(0.0, nx - 0.6),
        "up": max(0.0, 0.4 - ny),
        "down": max(0.0, ny - 0.6),
    }
    direction = max(outside, key=outside.get)
    if outside[direction] == 0.0:
        direction = "forward"
    return {
        "mode": "simple",
        "direction": direction,
        "step_m": float(step_m),
        "marker_center_px": [float(x), float(y)],
        "image_size_px": [int(width), int(height)],
        "execution": "plan_only_no_robot_command",
    }


def target_square_corners(
    image_size: tuple[int, int], half_size_fraction: float = 0.10
) -> np.ndarray:
    width, height = image_size
    if not 0 < half_size_fraction < 0.5:
        raise ValueError("目标半边长比例必须在 0 和 0.5 之间")
    half = min(width, height) * half_size_fraction
    cx, cy = width / 2.0, height / 2.0
    return np.asarray(
        [[cx - half, cy - half], [cx + half, cy - half],
         [cx + half, cy + half], [cx - half, cy + half]],
        dtype=np.float64,
    )


def ibvs_interaction_matrix(
    corners_px: np.ndarray,
    camera_matrix: np.ndarray,
    depths_m: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Build the stacked 8x6 point-feature interaction matrix from the notes."""
    corners_px = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    camera_matrix = _matrix(camera_matrix, (3, 3))
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError("相机焦距必须为正数")
    depths = np.asarray(depths_m, dtype=np.float64)
    if depths.ndim == 0:
        depths = np.full(4, float(depths), dtype=np.float64)
    depths = depths.reshape(-1)
    if depths.shape != (4,) or np.any(depths <= 0) or not np.all(np.isfinite(depths)):
        raise ValueError("四个角点深度必须是正的有限值")
    normalized = np.column_stack(
        ((corners_px[:, 0] - cx) / fx, (corners_px[:, 1] - cy) / fy)
    )
    rows: list[list[float]] = []
    for (x, y), depth in zip(normalized, depths, strict=True):
        rows.extend(
            [
                [-1 / depth, 0, x / depth, x * y, -(1 + x * x), y],
                [0, -1 / depth, y / depth, 1 + y * y, -x * y, -x],
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def ibvs_camera_velocity(
    corners_px: np.ndarray,
    target_px: np.ndarray,
    camera_matrix: np.ndarray,
    depth_m: float | Sequence[float] | np.ndarray,
    gain: float = 0.5,
    damping: float = 1e-4,
) -> tuple[np.ndarray, float]:
    corners_px = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    target_px = np.asarray(target_px, dtype=np.float64).reshape(4, 2)
    camera_matrix = _matrix(camera_matrix, (3, 3))
    if gain <= 0 or damping < 0:
        raise ValueError("控制增益必须为正数，阻尼不能为负数")
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError("相机焦距必须为正数")
    normalized = np.column_stack(
        ((corners_px[:, 0] - cx) / fx, (corners_px[:, 1] - cy) / fy)
    )
    target_normalized = np.column_stack(
        ((target_px[:, 0] - cx) / fx, (target_px[:, 1] - cy) / fy)
    )
    interaction = ibvs_interaction_matrix(corners_px, camera_matrix, depth_m)
    error = (normalized - target_normalized).reshape(-1)
    normal = interaction.T @ interaction + float(damping) ** 2 * np.eye(6)
    velocity = -float(gain) * np.linalg.solve(normal, interaction.T @ error)
    return velocity, float(np.linalg.norm(error))


def project_camera_points(points_camera_m: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera_m, dtype=np.float64).reshape(4, 3)
    camera_matrix = _matrix(camera_matrix, (3, 3))
    if np.any(points[:, 2] <= 0) or not np.all(np.isfinite(points)):
        raise ValueError("仿真角点必须位于相机前方")
    normalized = points[:, :2] / points[:, 2:3]
    return np.column_stack(
        (
            camera_matrix[0, 0] * normalized[:, 0] + camera_matrix[0, 2],
            camera_matrix[1, 1] * normalized[:, 1] + camera_matrix[1, 2],
        )
    )


def integrate_camera_twist(
    points_camera_m: np.ndarray, camera_twist: np.ndarray, dt_s: float
) -> np.ndarray:
    """Move a simulated camera while keeping the square fixed in the world."""
    points = np.asarray(points_camera_m, dtype=np.float64).reshape(4, 3)
    twist = np.asarray(camera_twist, dtype=np.float64).reshape(6)
    if dt_s <= 0 or not np.all(np.isfinite(twist)):
        raise ValueError("仿真时间步必须为正数，速度必须为有限值")
    rotation_delta, _ = cv2.Rodrigues(twist[3:] * float(dt_s))
    translated = points - twist[:3] * float(dt_s)
    return (rotation_delta.T @ translated.T).T


def _limit_vector_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if maximum <= 0:
        raise ValueError("速度上限必须为正数")
    return vector if norm <= maximum else vector * (maximum / norm)


def simulate_ibvs(
    camera_matrix: np.ndarray,
    image_size: tuple[int, int],
    marker_size_m: float = 0.025,
    target_half_fraction: float = 0.10,
    gain: float = 0.8,
    damping: float = 1e-4,
    dt_s: float = 0.05,
    max_steps: int = 400,
    tolerance_px: float = 1.0,
    max_linear_m_s: float = 0.12,
    max_angular_rad_s: float = 0.8,
    initial_translation_xy_m: tuple[float, float] = (0.045, -0.035),
    initial_depth_scale: float = 1.7,
    initial_rotation_vector_rad: Sequence[float] = (0.30, -0.20, 0.35),
) -> dict:
    """Run a closed-loop four-corner IBVS simulation without camera or robot I/O."""
    camera_matrix = _matrix(camera_matrix, (3, 3))
    width, height = image_size
    if width <= 0 or height <= 0 or marker_size_m <= 0:
        raise ValueError("图像尺寸和 ArUco 边长必须为正数")
    if max_steps < 1 or tolerance_px <= 0 or initial_depth_scale <= 1.0:
        raise ValueError("仿真步数、容差必须为正数，初始深度比例必须大于 1")
    target_px = target_square_corners(image_size, target_half_fraction)
    target_side_px = float(np.linalg.norm(target_px[1] - target_px[0]))
    target_depth_m = (
        math.sqrt(camera_matrix[0, 0] * camera_matrix[1, 1])
        * marker_size_m
        / target_side_px
    )
    half = marker_size_m / 2.0
    object_points = np.asarray(
        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]],
        dtype=np.float64,
    )
    rotation, _ = cv2.Rodrigues(
        np.asarray(initial_rotation_vector_rad, dtype=np.float64).reshape(3)
    )
    translation = np.asarray(
        [
            float(initial_translation_xy_m[0]),
            float(initial_translation_xy_m[1]),
            target_depth_m * float(initial_depth_scale),
        ]
    )
    points = (rotation @ object_points.T).T + translation
    history: list[dict] = []
    converged = False
    for step in range(max_steps + 1):
        corners_px = project_camera_points(points, camera_matrix)
        pixel_errors = corners_px - target_px
        rms_error_px = float(np.sqrt(np.mean(pixel_errors**2)))
        record = {
            "step": step,
            "corners_px": corners_px.tolist(),
            "depths_m": points[:, 2].tolist(),
            "rms_error_px": rms_error_px,
        }
        history.append(record)
        if rms_error_px <= tolerance_px:
            converged = True
            break
        velocity, normalized_error = ibvs_camera_velocity(
            corners_px,
            target_px,
            camera_matrix,
            points[:, 2],
            gain=gain,
            damping=damping,
        )
        velocity[:3] = _limit_vector_norm(velocity[:3], max_linear_m_s)
        velocity[3:] = _limit_vector_norm(velocity[3:], max_angular_rad_s)
        record["normalized_error_norm"] = normalized_error
        record["camera_twist"] = velocity.tolist()
        points = integrate_camera_twist(points, velocity, dt_s)
    return {
        "schema": 1,
        "mode": "four_corner_ibvs_simulation",
        "simulation_only": True,
        "robot_io": "not_opened",
        "camera_io": "not_opened",
        "converged": converged,
        "iterations": len(history) - 1,
        "initial_rms_error_px": history[0]["rms_error_px"],
        "final_rms_error_px": history[-1]["rms_error_px"],
        "target_depth_m": target_depth_m,
        "target_corners_px": target_px.tolist(),
        "history": history,
    }


def render_ibvs_simulation(result: dict, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    target = np.asarray(result["target_corners_px"], dtype=np.float64)
    history = result["history"]

    def panel(record: dict, title: str) -> np.ndarray:
        frame = np.full((height, width, 3), 248, dtype=np.uint8)
        corners = np.asarray(record["corners_px"], dtype=np.float64)
        center = (width // 2, height // 2)
        cv2.line(frame, (center[0] - 25, center[1]), (center[0] + 25, center[1]), (180, 180, 180), 2)
        cv2.line(frame, (center[0], center[1] - 25), (center[0], center[1] + 25), (180, 180, 180), 2)
        cv2.polylines(frame, [np.rint(target).astype(np.int32)], True, (80, 170, 40), 8, cv2.LINE_AA)
        cv2.polylines(frame, [np.rint(corners).astype(np.int32)], True, (0, 120, 235), 8, cv2.LINE_AA)
        for point in corners:
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 12, (0, 80, 220), -1, cv2.LINE_AA)
        cv2.putText(frame, title, (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(frame, f"RMS error: {record['rms_error_px']:.2f} px", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (40, 40, 40), 2, cv2.LINE_AA)
        return cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

    selected = [history[0], history[len(history) // 2], history[-1]]
    top = np.hstack(
        [panel(selected[0], "Initial"), panel(selected[1], "Midpoint"), panel(selected[2], "Final")]
    )
    graph = np.full((320, 1920, 3), 255, dtype=np.uint8)
    errors = np.asarray([item["rms_error_px"] for item in history], dtype=np.float64)
    left, right, top_y, bottom = 90, 1870, 45, 265
    cv2.rectangle(graph, (left, top_y), (right, bottom), (150, 150, 150), 2)
    if len(errors) > 1:
        xs = np.linspace(left, right, len(errors))
        scale = max(float(errors.max()), 1.0)
        ys = bottom - errors / scale * (bottom - top_y)
        curve = np.column_stack((xs, ys)).astype(np.int32)
        cv2.polylines(graph, [curve], False, (200, 90, 30), 4, cv2.LINE_AA)
    cv2.putText(graph, "Four-corner IBVS convergence", (left, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(graph, f"0", (left - 10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.putText(graph, f"{len(history) - 1} steps", (right - 120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
    return np.vstack((top, graph))


def _image_paths(directory: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [path for path in directory.iterdir() if path.suffix.lower() in suffixes]


def scan_camera_indices(max_index: int = 4) -> list[dict]:
    if max_index < 0:
        raise ValueError("最大摄像头索引不能为负数")
    results: list[dict] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        try:
            opened = capture.isOpened()
            ok, frame = capture.read() if opened else (False, None)
            results.append(
                {
                    "index": index,
                    "opened": bool(opened),
                    "frame_received": bool(ok),
                    "image_size_px": (
                        [int(frame.shape[1]), int(frame.shape[0])] if ok else None
                    ),
                }
            )
        finally:
            capture.release()
    return results


def command_camera_scan(args: argparse.Namespace) -> int:
    results = scan_camera_indices(args.max_index)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if any(item["frame_received"] for item in results) else 2


def command_camera_capture(args: argparse.Namespace) -> int:
    """Interactively save raw calibration frames only when the board is visible."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头索引 {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    saved = len(_image_paths(output_dir))
    last_saved_at = 0.0
    window_name = "Camera calibration - SPACE save, Q quit"
    try:
        while saved < args.target_count:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("摄像头已打开，但无法读取画面")
            frame = rotate_frame(frame, args.rotate)
            corners = find_chessboard(frame, args.columns, args.rows)
            preview = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(
                    preview,
                    (args.columns, args.rows),
                    corners.reshape(-1, 1, 2),
                    True,
                )
            status = (
                f"board OK | saved {saved}/{args.target_count} | SPACE save"
                if corners is not None
                else f"board not found | saved {saved}/{args.target_count}"
            )
            color = (40, 210, 40) if corners is not None else (40, 40, 230)
            cv2.putText(
                preview,
                status,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                if corners is None:
                    print("未保存：棋盘格未完整检出。")
                    continue
                now = time.monotonic()
                if now - last_saved_at < args.min_interval_sec:
                    print("未保存：与上一张间隔过短，请明显改变棋盘格姿态。")
                    continue
                saved += 1
                path = output_dir / f"camera_{saved:03d}.png"
                _write_image(path, frame)
                last_saved_at = now
                print(f"已保存 {saved}/{args.target_count}: {path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"采集结束：目录中共有 {len(_image_paths(output_dir))} 张图片。")
    return 0 if saved >= args.target_count else 2


def command_check(_: argparse.Namespace) -> int:
    checks = {
        "opencv_version": cv2.__version__,
        "aruco_available": hasattr(cv2, "aruco"),
        "calibrate_hand_eye_available": hasattr(cv2, "calibrateHandEye"),
        "urdf_exists": DEFAULT_URDF.is_file(),
        "robot_io": "not_opened",
    }
    if not all(
        checks[key]
        for key in ("aruco_available", "calibrate_hand_eye_available", "urdf_exists")
    ):
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 2
    kinematics = _load_kinematics()
    checks["fk_zero_translation_m"] = kinematics.forward([0] * 5)[:3, 3].tolist()
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0


def command_camera_calibrate(args: argparse.Namespace) -> int:
    observations, rejected = collect_chessboard_observations(
        _image_paths(Path(args.images)), args.columns, args.rows
    )
    result = calibrate_camera(
        observations, args.columns, args.rows, args.square_mm / 1000.0
    )
    result["rejected_images"] = rejected
    output = Path(args.output)
    _write_json(output, result)
    print(f"相机标定已保存：{output}")
    print(
        f"有效图片 {len(observations)} 张，RMS 重投影误差 "
        f"{result['rms_reprojection_error_px']:.4f} px"
    )
    if result["rms_reprojection_error_px"] > 1.5:
        print("警告：重投影误差明显高于约 1.0 px，请检查角点、打印比例和图片覆盖范围。")
        return 2
    return 0


def command_hand_eye(args: argparse.Namespace) -> int:
    result = calibrate_hand_eye_from_manifest(
        Path(args.manifest),
        Path(args.camera),
        args.columns,
        args.rows,
        args.square_mm / 1000.0,
        args.method,
    )
    output = Path(args.output)
    _write_json(output, result)
    print(f"手眼标定已保存：{output}")
    print(
        "标定板原点在基座系的各轴最大离均差："
        + ", ".join(
            f"{value * 100:.2f} cm"
            for value in result["board_origin_base_max_deviation_per_axis_m"]
        )
    )
    return 0


def _load_marker(args: argparse.Namespace) -> tuple[np.ndarray, ArucoObservation]:
    image = _read_image(Path(args.path))
    marker = select_marker(detect_aruco(image, args.dictionary), args.id)
    return image, marker


def command_aruco_image(args: argparse.Namespace) -> int:
    image, marker = _load_marker(args)
    annotated = image.copy()
    cv2.aruco.drawDetectedMarkers(
        annotated, [marker.corners.reshape(1, 4, 2)], np.asarray([[marker.marker_id]])
    )
    output = Path(args.output)
    _write_image(output, annotated)
    print(f"检测到 ArUco ID {marker.marker_id}，中心 {marker.center}，结果：{output}")
    return 0


def command_plan_simple(args: argparse.Namespace) -> int:
    image, marker = _load_marker(args)
    plan = simple_servo_plan(
        marker.center, (image.shape[1], image.shape[0]), args.step_cm / 100.0
    )
    plan["marker_id"] = marker.marker_id
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def command_plan_ibvs(args: argparse.Namespace) -> int:
    image, marker = _load_marker(args)
    camera_matrix, _, _ = load_camera_calibration(Path(args.camera))
    target = target_square_corners(
        (image.shape[1], image.shape[0]), args.target_half_fraction
    )
    velocity, error = ibvs_camera_velocity(
        marker.corners, target, camera_matrix, args.depth_m, args.gain
    )
    plan = {
        "mode": "ibvs",
        "marker_id": marker.marker_id,
        "corner_error_norm_normalized": error,
        "camera_twist": {
            "linear_m_per_s": velocity[:3].tolist(),
            "angular_rad_per_s": velocity[3:].tolist(),
        },
        "assumed_depth_m": args.depth_m,
        "target_corners_px": target.tolist(),
        "execution": "plan_only_no_robot_command",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def command_simulate_ibvs(args: argparse.Namespace) -> int:
    camera_matrix, _, calibration = load_camera_calibration(Path(args.camera))
    image_size = tuple(int(value) for value in calibration["image_size_px"])
    rotation_rad = tuple(math.radians(value) for value in args.initial_rvec_deg)
    result = simulate_ibvs(
        camera_matrix,
        image_size,
        marker_size_m=args.marker_size_mm / 1000.0,
        target_half_fraction=args.target_half_fraction,
        gain=args.gain,
        damping=args.damping,
        dt_s=args.dt,
        max_steps=args.max_steps,
        tolerance_px=args.tolerance_px,
        max_linear_m_s=args.max_linear_m_s,
        max_angular_rad_s=args.max_angular_rad_s,
        initial_translation_xy_m=(
            args.initial_x_mm / 1000.0,
            args.initial_y_mm / 1000.0,
        ),
        initial_depth_scale=args.initial_depth_scale,
        initial_rotation_vector_rad=rotation_rad,
    )
    result["camera_calibration"] = str(Path(args.camera))
    result["configuration"] = {
        "marker_size_mm": args.marker_size_mm,
        "gain": args.gain,
        "damping": args.damping,
        "dt_s": args.dt,
        "tolerance_px": args.tolerance_px,
        "max_linear_m_s": args.max_linear_m_s,
        "max_angular_rad_s": args.max_angular_rad_s,
        "initial_translation_xy_mm": [args.initial_x_mm, args.initial_y_mm],
        "initial_depth_scale": args.initial_depth_scale,
        "initial_rotation_vector_deg": list(args.initial_rvec_deg),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ibvs_simulation.json"
    image_path = output_dir / "ibvs_simulation.png"
    _write_json(json_path, result)
    _write_image(image_path, render_ibvs_simulation(result, image_size))
    print("四角点 IBVS 纯仿真完成；未打开摄像头、串口或机械臂。")
    print(
        f"误差：{result['initial_rms_error_px']:.2f} -> "
        f"{result['final_rms_error_px']:.2f} px，迭代 {result['iterations']} 步。"
    )
    print(f"数据：{json_path}")
    print(f"可视化：{image_path}")
    return 0 if result["converged"] else 2


def _add_board_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--columns", type=int, default=8, help="棋盘格横向内角点数")
    parser.add_argument("--rows", type=int, default=6, help="棋盘格纵向内角点数")
    parser.add_argument("--square-mm", type=float, default=25.0)


def _add_aruco_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path")
    parser.add_argument("--id", type=int)
    parser.add_argument("--dictionary", default="4X4_50")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ArUco 视觉伺服、标定与 IBVS 工具"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="无相机、无机械臂依赖检查")
    check.set_defaults(func=command_check)

    camera_scan = sub.add_parser("camera-scan", help="只读枚举可取帧的摄像头索引")
    camera_scan.add_argument("--max-index", type=int, default=4)
    camera_scan.set_defaults(func=command_camera_scan)

    camera_capture = sub.add_parser(
        "camera-capture", help="交互采集棋盘格照片，不连接机械臂"
    )
    camera_capture.add_argument("--camera", type=int, default=0)
    camera_capture.add_argument("--width", type=int, default=1280)
    camera_capture.add_argument("--height", type=int, default=720)
    camera_capture.add_argument(
        "--rotate", type=int, choices=(0, 90, 180, 270), default=180
    )
    camera_capture.add_argument("--target-count", type=int, default=20)
    camera_capture.add_argument("--min-interval-sec", type=float, default=1.0)
    camera_capture.add_argument(
        "--output-dir", default=str(PROJECT_DIR / "data" / "camera")
    )
    _add_board_arguments(camera_capture)
    camera_capture.set_defaults(func=command_camera_capture)

    camera = sub.add_parser("camera-calibrate", help="从棋盘格图片计算相机内参")
    camera.add_argument("--images", required=True)
    camera.add_argument("--output", default=str(DEFAULT_CAMERA_RESULT))
    _add_board_arguments(camera)
    camera.set_defaults(func=command_camera_calibrate)

    hand_eye = sub.add_parser("hand-eye", help="从图片与关节角清单计算手眼矩阵")
    hand_eye.add_argument("--manifest", required=True)
    hand_eye.add_argument("--camera", default=str(DEFAULT_CAMERA_RESULT))
    hand_eye.add_argument("--output", default=str(DEFAULT_HAND_EYE_RESULT))
    hand_eye.add_argument("--method", choices=tuple(HAND_EYE_METHODS), default="tsai")
    _add_board_arguments(hand_eye)
    hand_eye.set_defaults(func=command_hand_eye)

    detect = sub.add_parser("aruco-image", help="检测静态图片中的 ArUco 码")
    _add_aruco_arguments(detect)
    detect.add_argument("--output", default=str(PROJECT_DIR / "data" / "aruco_detected.png"))
    detect.set_defaults(func=command_aruco_image)

    simple = sub.add_parser("plan-simple", help="输出基于图像分区的简单版动作计划")
    _add_aruco_arguments(simple)
    simple.add_argument("--step-cm", type=float, default=1.0)
    simple.set_defaults(func=command_plan_simple)

    ibvs = sub.add_parser("plan-ibvs", help="输出四角点 IBVS 相机速度计划")
    _add_aruco_arguments(ibvs)
    ibvs.add_argument("--camera", default=str(DEFAULT_CAMERA_RESULT))
    ibvs.add_argument("--depth-m", type=float, required=True)
    ibvs.add_argument("--gain", type=float, default=0.5)
    ibvs.add_argument("--target-half-fraction", type=float, default=0.10)
    ibvs.set_defaults(func=command_plan_ibvs)

    simulation = sub.add_parser(
        "simulate-ibvs", help="纯仿真四角点 IBVS，不打开摄像头或机械臂"
    )
    simulation.add_argument("--camera", default=str(DEFAULT_CAMERA_RESULT))
    simulation.add_argument("--marker-size-mm", type=float, default=25.0)
    simulation.add_argument("--gain", type=float, default=0.8)
    simulation.add_argument("--damping", type=float, default=1e-4)
    simulation.add_argument("--dt", type=float, default=0.05)
    simulation.add_argument("--max-steps", type=int, default=400)
    simulation.add_argument("--tolerance-px", type=float, default=1.0)
    simulation.add_argument("--max-linear-m-s", type=float, default=0.12)
    simulation.add_argument("--max-angular-rad-s", type=float, default=0.8)
    simulation.add_argument("--target-half-fraction", type=float, default=0.10)
    simulation.add_argument("--initial-x-mm", type=float, default=45.0)
    simulation.add_argument("--initial-y-mm", type=float, default=-35.0)
    simulation.add_argument("--initial-depth-scale", type=float, default=1.7)
    simulation.add_argument(
        "--initial-rvec-deg", type=float, nargs=3, default=(18.0, -12.0, 20.0)
    )
    simulation.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "data" / "advanced_ibvs_simulation"),
    )
    simulation.set_defaults(func=command_simulate_ibvs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
