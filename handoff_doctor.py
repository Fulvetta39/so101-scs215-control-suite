from __future__ import annotations

import json
import tomllib
from pathlib import Path

import cv2


def find_cameras(max_index: int = 5) -> list[dict]:
    cameras: list[dict] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if ok and frame is not None:
            cameras.append({"index": index, "width": width, "height": height})
    return cameras


def serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [port.device for port in list_ports.comports()]


def main() -> int:
    root = Path(__file__).resolve().parent
    config_path = root / "config.robot.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    calibration = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "calibration"
        / "robots"
        / "so_follower"
        / "so101-follower.json"
    )
    result = {
        "workspace": str(root),
        "serial_ports": serial_ports(),
        "cameras": find_cameras(),
        "configured_robot_port": config.get("robot", {}).get("port"),
        "configured_camera_index": config.get("camera", {}).get("index"),
        "calibration_present": calibration.is_file(),
        "calibration_path": str(calibration),
        "env_present": (root / ".env").is_file(),
        "models_present": all(
            path.is_file()
            for path in (
                root / "jarvis" / "perception" / "models" / "blaze_face_short_range.tflite",
                root / "jarvis" / "perception" / "models" / "hand_landmarker.task",
            )
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
