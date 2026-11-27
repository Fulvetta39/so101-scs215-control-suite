from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

_MPL_CONFIG = Path(tempfile.gettempdir()) / "jarvis-matplotlib"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

from .pipeline import PerceptionPipeline
from .naming import PointedObjectNamer
from .servo import (
    FaceServoConfig,
    FaceServoPlanner,
    HttpPointingServoBridge,
    HttpServoBridge,
    PointingGazeConfig,
    PointingGazePlanner,
    annotate_pointing_gaze,
    annotate_servo,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JARVIS local face and hand perception")
    parser.add_argument("--source", default="0", help="Camera index, video path, or 'synthetic'")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--models", default=None, help="Directory containing MediaPipe model files")
    parser.add_argument("--jsonl", default=None, help="Write one JSON perception result per line")
    parser.add_argument("--record", default=None, help="Write an annotated MP4")
    parser.add_argument("--headless", action="store_true", help="Do not open a preview window")
    parser.add_argument(
        "--web-preview",
        type=int,
        default=None,
        metavar="PORT",
        help="Serve the annotated camera stream on http://127.0.0.1:PORT/",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help="Web preview bind address; use 0.0.0.0 for same-network access",
    )
    parser.add_argument("--demo-title", default=None, help="Custom title for the web preview")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal camera mirror")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--face-confidence", type=float, default=0.5)
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument(
        "--servo-shadow",
        action="store_true",
        help="Show face-centering corrections without moving the robot",
    )
    parser.add_argument(
        "--servo-execute",
        action="store_true",
        help="Send rate-limited corrections to the local robot safety API",
    )
    parser.add_argument(
        "--pointing-shadow",
        action="store_true",
        help="Show pointing-direction turn suggestions without moving the robot",
    )
    parser.add_argument(
        "--pointing-execute",
        action="store_true",
        help="Send stable pointing-direction corrections to the robot safety API",
    )
    parser.add_argument(
        "--servo-endpoint",
        default="http://127.0.0.1:8000/api/servo",
    )
    parser.add_argument("--servo-rate-hz", type=float, default=2.5)
    parser.add_argument("--servo-vertical-rate-hz", type=float, default=0.2)
    parser.add_argument("--servo-step-deg", type=float, default=3.0)
    parser.add_argument("--servo-vertical-max-step", type=float, default=6.0)
    parser.add_argument("--servo-vertical-min-step", type=float, default=6.0)
    parser.add_argument("--servo-vertical-deadband", type=float, default=0.2)
    parser.add_argument(
        "--enable-vertical-servo",
        action="store_true",
        help="Also send vertical corrections; default is horizontal only",
    )
    parser.add_argument(
        "--allow-experimental-vertical",
        action="store_true",
        help="Explicitly allow the not-yet-stable vertical tracking loop",
    )
    parser.add_argument(
        "--servo-vertical-joint",
        choices=("shoulder_lift", "wrist_flex"),
        default="wrist_flex",
        help="Joint used for vertical face correction",
    )
    parser.add_argument("--pan-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--vertical-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--servo-pan-gain", type=float, default=10.0)
    parser.add_argument("--servo-vertical-gain", type=float, default=8.0)
    parser.add_argument("--servo-max-step", type=float, default=3.0)
    parser.add_argument("--servo-deadband", type=float, default=0.08)
    parser.add_argument("--pointing-pan-gain", type=float, default=12.0)
    parser.add_argument("--pointing-vertical-gain", type=float, default=8.0)
    parser.add_argument("--pointing-max-step", type=float, default=3.0)
    parser.add_argument("--pointing-deadband", type=float, default=0.18)
    parser.add_argument("--pointing-min-confidence", type=float, default=0.62)
    parser.add_argument("--pointing-rate-hz", type=float, default=1.0)
    parser.add_argument("--pointing-vertical-rate-hz", type=float, default=0.2)
    parser.add_argument("--pointing-step-deg", type=float, default=2.0)
    parser.add_argument("--pointing-vertical-max-step", type=float, default=2.0)
    parser.add_argument("--pointing-vertical-min-step", type=float, default=0.0)
    parser.add_argument(
        "--pointing-max-steps",
        type=int,
        default=3,
        help="Maximum steps per gesture; 0 disables the per-gesture step cap",
    )
    parser.add_argument(
        "--name-pointed-object",
        action="store_true",
        help="After a bounded pointing gesture, ask the multimodal agent to name the target",
    )
    parser.add_argument(
        "--name-endpoint",
        default="http://127.0.0.1:8000/api/chat",
    )
    parser.add_argument("--name-delay-s", type=float, default=1.0)
    parser.add_argument("--pointing-settle-s", type=float, default=0.6)
    parser.add_argument(
        "--enable-pointing-vertical",
        action="store_true",
        help="Also send vertical pointing corrections; default is horizontal only",
    )
    parser.add_argument(
        "--allow-experimental-pointing-vertical",
        action="store_true",
        help="Explicitly allow vertical pointing corrections",
    )
    parser.add_argument(
        "--pointing-vertical-joint",
        choices=("shoulder_lift", "wrist_flex"),
        default="wrist_flex",
    )
    return parser


def open_source(source: str, width: int, height: int):
    if source == "synthetic":
        return None
    if source.startswith(("http://", "https://")):
        return HttpFrameSource(source)
    if source.isdigit():
        index = int(source)
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        return capture
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    return capture


class HttpFrameSource:
    """Consume a single JPEG endpoint without taking ownership of the camera."""

    def __init__(self, url: str):
        self.url = url

    def read(self):
        for _ in range(3):
            try:
                request = urllib.request.Request(
                    f"{self.url}?t={time.time_ns()}",
                    headers={"Cache-Control": "no-cache"},
                )
                with urllib.request.urlopen(request, timeout=2.0) as response:
                    encoded = np.frombuffer(response.read(), dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
            except OSError:
                time.sleep(0.08)
        return False, None

    def release(self) -> None:
        return None


class _PreviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, preview: "WebPreview"):
        super().__init__(address, _PreviewRequestHandler)
        self.preview = preview


class _PreviewRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._html()
        elif path == "/stream":
            self._stream()
        elif path == "/api/latest":
            self._latest()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/stop":
            self._stop()
        elif path == "/api/resume":
            self._resume()
        else:
            self.send_error(404)

    def log_message(self, *_args) -> None:
        return None

    def _html(self) -> None:
        executing = self.server.preview.executing
        title = self.server.preview.title or (
            "JARVIS 指向实机测试" if executing else "JARVIS 指向影子测试"
        )
        disclaimer = (
            "机械臂会在方向稳定后小步转向；手离开画面或指向丢失时立即停止。"
            if executing
            else "当前只计算建议，不控制机械臂。"
        )
        instructions = (
            "先看向机械臂让人脸跟随；再伸出右手食指控制转向。"
            "手移开后立即停止，当前版本不执行语音或物体命名。"
            if title.startswith("JARVIS Midterm Demo")
            else "伸出右手食指，其余三指弯曲。先指向画面右侧，再依次测试左、下、上。"
        )
        body = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    body { margin: 0; background: #111820; color: #eef4f8; font-family: sans-serif; }
    main { max-width: 1100px; margin: 0 auto; padding: 18px; }
    h1 { font-size: 22px; margin: 0 0 12px; }
    img { display: block; width: 100%; border-radius: 10px; background: #000; }
    p { color: #b9c7d2; line-height: 1.7; }
    #status { white-space: pre-wrap; padding: 10px 12px; border-radius: 8px;
              background: #1b2732; color: #ffdc73; font: 15px/1.6 monospace; }
    #stop { margin: 8px 0 16px; padding: 12px 18px; border: 0; border-radius: 8px;
            background: #c73535; color: white; font-size: 16px; font-weight: 700;
            cursor: pointer; }
    #resume { margin: 8px 0 16px 8px; padding: 12px 18px; border: 0; border-radius: 8px;
              background: #16885c; color: white; font-size: 16px; font-weight: 700;
              cursor: pointer; }
  </style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <img src="/stream" alt="实时摄像头画面">
  <p>__INSTRUCTIONS__ 黄色箭头表示建议转向；__DISCLAIMER__</p>
  <button id="stop" type="button">急停并保持位置</button>
  <button id="resume" type="button">确认现场安全并恢复</button>
  <pre id="status">正在读取状态...</pre>
</main>
<script>
async function updateStatus() {
  try {
    const response = await fetch('/api/latest', { cache: 'no-store' });
    const state = await response.json();
    const gaze = state.pointing_gaze || {};
    const execution = state.pointing_execution || {};
    const demo = state.demo_status || {};
    document.getElementById('status').textContent =
      `演示状态：${demo.behavior || '等待中'}\n` +
      `检测：脸 ${state.face_count}，手 ${state.hand_count}，指向 ${state.pointing ? '是' : '否'}\n` +
      `建议：pan=${gaze.pan_delta_deg ?? 0}，wrist=${gaze.vertical_delta_deg ?? 0}\n` +
      `执行：${execution.sent ? '已发送' : '未发送'}，${execution.reason || gaze.reason || '等待中'}`;
  } catch (_) {
    document.getElementById('status').textContent = '预览服务未连接';
  }
}
document.getElementById('stop').addEventListener('click', async () => {
  document.getElementById('stop').disabled = true;
  try {
    const response = await fetch('/api/stop', { method: 'POST' });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.message || '急停失败');
    document.getElementById('status').textContent = '急停已触发，机械臂已保持当前位置。';
    document.getElementById('resume').disabled = false;
  } catch (error) {
    document.getElementById('status').textContent = `急停失败：${error.message}`;
    document.getElementById('stop').disabled = false;
  }
});
document.getElementById('resume').addEventListener('click', async () => {
  if (!window.confirm('请确认机械臂周围清空、急停可触达。是否恢复力矩并继续？')) return;
  document.getElementById('resume').disabled = true;
  try {
    const response = await fetch('/api/resume', { method: 'POST' });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.detail || result.message || '恢复失败');
    document.getElementById('status').textContent = result.message;
    document.getElementById('stop').disabled = false;
  } catch (error) {
    document.getElementById('status').textContent = `恢复失败：${error.message}`;
    document.getElementById('resume').disabled = false;
  }
});
setInterval(updateStatus, 250);
updateStatus();
</script>
</body>
</html>
"""
        body = (
            body.replace("__TITLE__", title)
            .replace("__DISCLAIMER__", disclaimer)
            .replace("__INSTRUCTIONS__", instructions)
        )
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        last_version = -1
        try:
            while not self.server.preview.closed:
                version, jpeg = self.server.preview.latest()
                if jpeg is None or version == last_version:
                    time.sleep(0.03)
                    continue
                last_version = version
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _latest(self) -> None:
        _, payload = self.server.preview.latest_state()
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _stop(self) -> None:
        self._proxy_robot_post("/api/stop")

    def _resume(self) -> None:
        self._proxy_robot_post("/api/resume")

    def _proxy_robot_post(self, path: str) -> None:
        request = urllib.request.Request(
            f"http://127.0.0.1:8000{path}",
            data=b"",
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                encoded = response.read()
            self.send_response(200)
        except OSError as exc:
            encoded = json.dumps(
                {"ok": False, "message": str(exc)},
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(503)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


class WebPreview:
    """Publish the latest annotated frame as a local browser MJPEG stream."""

    def __init__(
        self,
        port: int,
        *,
        host: str = "127.0.0.1",
        executing: bool = False,
        title: str | None = None,
    ):
        if not 1 <= port <= 65535:
            raise ValueError("web preview port must be between 1 and 65535")
        self.closed = False
        self.executing = executing
        self.title = title
        self._lock = threading.Lock()
        self._version = 0
        self._jpeg: bytes | None = None
        self._payload: dict = {}
        self._server = _PreviewHttpServer((host, port), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="jarvis-web-preview",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def publish(self, image: np.ndarray, payload: dict) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        if not ok:
            return
        with self._lock:
            self._version += 1
            self._jpeg = encoded.tobytes()
            self._payload = payload

    def latest(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._version, self._jpeg

    def latest_state(self) -> tuple[int, dict]:
        with self._lock:
            return self._version, self._payload

    def close(self) -> None:
        self.closed = True
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_handle = None
    writer = None
    web_preview = (
        WebPreview(
            args.web_preview,
            host=args.web_host,
            executing=args.pointing_execute or args.servo_execute,
            title=args.demo_title,
        )
        if args.web_preview
        else None
    )
    pipeline = PerceptionPipeline(
        args.models,
        face_confidence=args.face_confidence,
        hand_confidence=args.hand_confidence,
        mirror=not args.no_mirror,
        max_hands=args.max_hands,
    )
    if args.servo_shadow and args.servo_execute:
        raise ValueError("--servo-shadow and --servo-execute are mutually exclusive")
    if args.pointing_shadow and args.pointing_execute:
        raise ValueError("--pointing-shadow and --pointing-execute are mutually exclusive")
    if args.pointing_execute and args.servo_shadow:
        raise ValueError("Pointing execution cannot be combined with face shadow mode")
    if args.name_pointed_object and not args.pointing_execute:
        raise ValueError("--name-pointed-object requires --pointing-execute")
    if args.enable_vertical_servo and not args.allow_experimental_vertical:
        raise ValueError(
            "Vertical face tracking is experimental. Add "
            "--allow-experimental-vertical to enable it."
        )
    if (
        args.enable_pointing_vertical
        and not args.allow_experimental_pointing_vertical
    ):
        raise ValueError(
            "Vertical pointing control is experimental. Add "
            "--allow-experimental-pointing-vertical to enable it."
        )
    servo_active = args.servo_shadow or args.servo_execute
    servo_planner = (
        FaceServoPlanner(
            FaceServoConfig(
                pan_gain_deg=args.servo_pan_gain,
                vertical_gain_deg=args.servo_vertical_gain,
                max_step_deg=args.servo_max_step,
                deadband=args.servo_deadband,
                pan_sign=args.pan_sign,
                vertical_sign=args.vertical_sign,
                vertical_joint=args.servo_vertical_joint,
                vertical_max_step_deg=args.servo_vertical_max_step,
                vertical_min_step_deg=args.servo_vertical_min_step,
                vertical_deadband=args.servo_vertical_deadband,
            )
        )
        if servo_active
        else None
    )
    servo_bridge = (
        HttpServoBridge(
            args.servo_endpoint,
            rate_hz=args.servo_rate_hz,
            vertical_rate_hz=args.servo_vertical_rate_hz,
            max_step_deg=args.servo_step_deg,
            vertical_max_step_deg=args.servo_vertical_max_step,
            enable_vertical=args.enable_vertical_servo,
        )
        if args.servo_execute
        else None
    )
    pointing_planner = (
        PointingGazePlanner(
            PointingGazeConfig(
                pan_gain_deg=args.pointing_pan_gain,
                vertical_gain_deg=args.pointing_vertical_gain,
                max_step_deg=args.pointing_max_step,
                deadband=args.pointing_deadband,
                pan_sign=args.pan_sign,
                vertical_sign=args.vertical_sign,
                vertical_joint=args.pointing_vertical_joint,
                min_confidence=args.pointing_min_confidence,
            )
        )
        if args.pointing_shadow or args.pointing_execute
        else None
    )
    pointing_bridge = (
        HttpPointingServoBridge(
            args.servo_endpoint,
            rate_hz=args.pointing_rate_hz,
            vertical_rate_hz=args.pointing_vertical_rate_hz,
            max_step_deg=args.pointing_step_deg,
            vertical_max_step_deg=args.pointing_vertical_max_step,
            vertical_min_step_deg=args.pointing_vertical_min_step,
            max_steps_per_gesture=args.pointing_max_steps,
            settle_s=args.pointing_settle_s,
            enable_vertical=args.enable_pointing_vertical,
        )
        if args.pointing_execute
        else None
    )
    object_namer = (
        PointedObjectNamer(
            args.name_endpoint,
            delay_s=args.name_delay_s,
        )
        if args.name_pointed_object
        else None
    )
    try:
        if web_preview:
            web_preview.start()
        if args.jsonl:
            output = Path(args.jsonl)
            output.parent.mkdir(parents=True, exist_ok=True)
            json_handle = output.open("w", encoding="utf-8")

        capture = open_source(args.source, args.width, args.height)
        started = time.monotonic()
        frame_index = 0
        read_failures = 0
        gesture_activity = False
        pointing_release_since: float | None = None
        previous = time.monotonic()
        fps = 0.0
        while True:
            if capture is None:
                frame = _synthetic_frame(args.width, args.height, frame_index)
            else:
                ok, frame = capture.read()
                if not ok:
                    read_failures += 1
                    if isinstance(capture, HttpFrameSource) and read_failures < 20:
                        time.sleep(0.2)
                        continue
                    break
                read_failures = 0

            now = time.monotonic() - started
            result = pipeline.process(frame, now)
            annotated = pipeline.annotate(frame, result)
            servo_recommendation = servo_planner.update(result) if servo_planner else None
            pointing_recommendation = (
                pointing_planner.update(result) if pointing_planner else None
            )
            pointing_execution = (
                pointing_bridge.maybe_send(pointing_recommendation)
                if pointing_bridge and pointing_recommendation
                else None
            )
            if object_namer and pointing_execution and pointing_execution.get("sent"):
                gesture_activity = True
                pointing_release_since = None
            if object_namer and gesture_activity:
                if pointing_recommendation and pointing_recommendation.tracking:
                    pointing_release_since = None
                elif pointing_release_since is None:
                    pointing_release_since = now
                elif now - pointing_release_since >= 0.35:
                    object_namer.schedule()
                    gesture_activity = False
                    pointing_release_since = None
            if object_namer:
                object_namer.maybe_start()
            naming_state = object_namer.snapshot() if object_namer else {}
            naming_hold = naming_state.get("status") in {"scheduled", "processing"}
            post_gesture_hold = gesture_activity
            servo_execution = None
            if servo_bridge and servo_recommendation:
                if pointing_recommendation and pointing_recommendation.tracking:
                    servo_execution = {
                        "sent": False,
                        "reason": "pointing_has_priority",
                    }
                elif naming_hold or post_gesture_hold:
                    servo_execution = {
                        "sent": False,
                        "reason": "object_naming_in_progress",
                    }
                else:
                    servo_execution = servo_bridge.maybe_send(servo_recommendation)
            if servo_recommendation is not None:
                annotated = annotate_servo(
                    annotated,
                    servo_recommendation,
                    pan_sign=args.pan_sign,
                    vertical_sign=args.vertical_sign,
                )
            if pointing_recommendation is not None:
                annotated = annotate_pointing_gaze(
                    annotated,
                    pointing_recommendation,
                    pan_sign=args.pan_sign,
                    vertical_sign=args.vertical_sign,
                    mode_label="EXEC" if args.pointing_execute else "SHADOW",
                )
            current = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(1e-6, current - previous)
            previous = current
            cv2.putText(
                annotated,
                f"FPS {fps:.1f}  face={len(result.faces)} hand={len(result.hands)}",
                (12, annotated.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            payload = result.to_dict()
            if servo_recommendation is not None:
                payload["servo_shadow"] = servo_recommendation.to_dict()
            if servo_execution is not None:
                payload["servo_execution"] = servo_execution
            if pointing_recommendation is not None:
                payload["pointing_gaze"] = pointing_recommendation.to_dict()
            if pointing_execution is not None:
                payload["pointing_execution"] = pointing_execution
                if pointing_execution.get("sent") or pointing_execution.get("reason") in {
                    "server_rejected",
                    "actuator_no_response",
                    "transport_error",
                }:
                    print(
                        f"pointing_execution={json.dumps(pointing_execution, ensure_ascii=False)}",
                        flush=True,
                    )
            if object_namer:
                payload["object_naming"] = naming_state
            if args.servo_execute or args.pointing_execute:
                if post_gesture_hold:
                    behavior = "正在等待动作稳定"
                elif naming_hold:
                    behavior = "正在识别物体"
                elif pointing_recommendation and pointing_recommendation.tracking:
                    behavior = "正在顺着手指看"
                elif (
                    servo_recommendation
                    and servo_recommendation.reason == "search_last_direction"
                ):
                    behavior = "正在沿最后方向搜索"
                elif (
                    servo_recommendation
                    and servo_recommendation.reason == "search_return_sweep"
                ):
                    behavior = "正在反向回扫"
                elif (
                    servo_recommendation
                    and servo_recommendation.reason == "face_search_reverse_hold"
                ):
                    behavior = "搜索换向停顿"
                elif servo_recommendation and servo_recommendation.tracking:
                    behavior = "正在看向说话人"
                else:
                    behavior = "正在搜索目标"
                payload["demo_status"] = {
                    "behavior": behavior,
                    "face_tracking": bool(
                        servo_recommendation and servo_recommendation.tracking
                    ),
                    "pointing_tracking": bool(
                        pointing_recommendation and pointing_recommendation.tracking
                    ),
                    "naming_status": naming_state.get("status", "idle"),
                }
            if json_handle:
                json_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                json_handle.flush()
            if web_preview:
                web_preview.publish(annotated, payload)
            if not args.headless and web_preview is None:
                cv2.imshow("JARVIS Perception", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.record:
                if writer is None:
                    output = Path(args.record)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0,
                        (width, height),
                    )
                writer.write(annotated)

            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
        if capture is not None:
            capture.release()
    finally:
        if json_handle:
            json_handle.close()
        if writer:
            writer.release()
        if web_preview:
            web_preview.close()
        if object_namer:
            object_namer.close()
        pipeline.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


def _synthetic_frame(width: int, height: int, index: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x = int((index * 12) % max(1, width))
    cv2.circle(image, (x, height // 2), 50, (20, 180, 120), -1)
    return image


if __name__ == "__main__":
    raise SystemExit(main())

