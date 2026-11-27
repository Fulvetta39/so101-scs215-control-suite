from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from .config import CameraSettings

LOGGER = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when no usable wrist-camera frame can be acquired."""


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    return ImageFont.load_default()


def make_demo_frame(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "#d9c7a8")
    draw = ImageDraw.Draw(image)

    for y in range(height):
        blend = y / max(1, height - 1)
        color = (
            int(222 - 24 * blend),
            int(211 - 20 * blend),
            int(188 - 16 * blend),
        )
        draw.line((0, y, width, y), fill=color)

    # Tabletop perspective.
    draw.polygon(
        [
            (width * 0.10, height * 0.28),
            (width * 0.92, height * 0.28),
            (width * 1.04, height * 1.0),
            (-width * 0.04, height * 1.0),
        ],
        fill="#b88a5f",
        outline="#8f6847",
        width=max(2, width // 400),
    )
    for i in range(1, 7):
        x = width * i / 7
        draw.line((x, height * 0.29, x - width * 0.02, height), fill="#a97b53", width=2)

    # Work area and objects.
    draw.rounded_rectangle(
        (
            width * 0.24,
            height * 0.47,
            width * 0.79,
            height * 0.79,
        ),
        radius=18,
        outline="#f5e9cf",
        width=4,
        fill="#c99769",
    )
    draw.rounded_rectangle(
        (
            width * 0.33,
            height * 0.56,
            width * 0.44,
            height * 0.72,
        ),
        radius=10,
        fill="#dc4b3e",
        outline="#9e2d25",
        width=3,
    )
    draw.ellipse(
        (
            width * 0.58,
            height * 0.53,
            width * 0.69,
            height * 0.58,
        ),
        fill="#3f77c5",
        outline="#244f8d",
        width=3,
    )
    draw.rounded_rectangle(
        (
            width * 0.56,
            height * 0.56,
            width * 0.71,
            height * 0.75,
        ),
        radius=24,
        fill="#3f77c5",
        outline="#244f8d",
        width=4,
    )

    # Wrist-camera gripper silhouette.
    draw.rounded_rectangle(
        (
            width * 0.44,
            height * 0.83,
            width * 0.56,
            height * 0.98,
        ),
        radius=18,
        fill="#3b414a",
        outline="#171b20",
        width=3,
    )
    draw.line(
        (width * 0.45, height * 0.88, width * 0.39, height * 0.76),
        fill="#252a30",
        width=max(8, width // 80),
    )
    draw.line(
        (width * 0.55, height * 0.88, width * 0.61, height * 0.76),
        fill="#252a30",
        width=max(8, width // 80),
    )

    small = _font(max(14, width // 48))
    large = _font(max(20, width // 32))
    draw.rounded_rectangle(
        (20, 18, width * 0.29, 70),
        radius=10,
        fill="#172029",
    )
    draw.text((34, 27), "DEMO WRIST CAM", fill="#f5f7fa", font=small)
    draw.text(
        (width * 0.32, height * 0.80),
        "RED BLOCK",
        fill="#5b211c",
        font=large,
    )
    draw.text(
        (width * 0.65, height * 0.79),
        "BLUE CUP",
        fill="#17345f",
        font=large,
    )

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


class CameraService:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self._capture: cv2.VideoCapture | None = None
        self._latest_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mode = settings.backend
        self.last_error: str | None = None

    def start(self) -> None:
        if self.settings.backend == "demo":
            self._latest_jpeg = make_demo_frame(self.settings.width, self.settings.height)
            self.mode = "demo"
            return

        capture = cv2.VideoCapture(self.settings.index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.settings.index)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
            capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture = capture
            self.mode = "device"
        elif self.settings.fallback_to_demo:
            self.last_error = f"无法打开摄像头 {self.settings.index}，已切换到演示画面。"
            self._latest_jpeg = make_demo_frame(self.settings.width, self.settings.height)
            self.mode = "demo"
            LOGGER.warning(self.last_error)
        else:
            raise CameraError(f"无法打开腕部摄像头，索引为 {self.settings.index}。")

        self._thread = threading.Thread(target=self._capture_loop, name="wrist-camera", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        period = 1.0 / max(1, self.settings.fps)
        while not self._stop.is_set():
            started = time.monotonic()
            if self._capture is not None:
                ok, frame = self._capture.read()
                if ok and frame is not None:
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality],
                    )
                    if ok:
                        with self._lock:
                            self._latest_jpeg = encoded.tobytes()
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.01, period - elapsed))

    def jpeg(self) -> bytes:
        with self._lock:
            frame = self._latest_jpeg
        if frame:
            return frame
        if self.settings.fallback_to_demo:
            return make_demo_frame(self.settings.width, self.settings.height)
        raise CameraError("相机尚无可用画面。")

    def status(self) -> dict[str, str | int | None]:
        return {
            "mode": self.mode,
            "index": self.settings.index,
            "width": self.settings.width,
            "height": self.settings.height,
            "error": self.last_error,
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

