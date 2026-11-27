from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class PointedObjectNamer:
    """Ask the local multimodal agent to name the current camera target."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/api/chat",
        *,
        prompt: str = (
            "只观察当前画面，不要移动机械臂。请用一句简短中文说出画面中央"
            "最明显的物体及其颜色和形状；如果中央没有物体，就明确说没有。"
        ),
        delay_s: float = 1.0,
        timeout_s: float = 45.0,
    ):
        if delay_s < 0:
            raise ValueError("naming delay cannot be negative")
        if timeout_s <= 0:
            raise ValueError("naming timeout must be positive")
        self.endpoint = endpoint
        self.prompt = prompt
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._due_s: float | None = None
        self._state: dict = {"status": "idle"}
        self._closed = False

    def schedule(self) -> None:
        with self._lock:
            if self._closed or self._due_s is not None:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._due_s = time.monotonic() + self.delay_s
            self._state = {
                "status": "scheduled",
                "reply": None,
                "error": None,
            }

    def maybe_start(self, now: float | None = None) -> None:
        with self._lock:
            if self._closed or self._due_s is None:
                return
            current = time.monotonic() if now is None else now
            if current < self._due_s:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._due_s = None
            self._state = {
                "status": "processing",
                "reply": None,
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._request_name,
                name="jarvis-object-namer",
                daemon=True,
            )
            self._thread.start()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _request_name(self) -> None:
        body = urllib.parse.urlencode({"text": self.prompt}).encode("utf-8")
        state = {
            "status": "error",
            "reply": None,
            "error": "vision model request did not complete",
        }
        for attempt in range(2):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                reply = str(payload.get("reply", "")).strip()
                if not reply:
                    raise ValueError("vision model returned an empty response")
                state = {"status": "done", "reply": reply, "error": None}
                break
            except urllib.error.HTTPError as exc:
                state = {
                    "status": "error",
                    "reply": None,
                    "error": f"HTTP {exc.code}: {exc.reason}",
                }
                if exc.code < 500 or attempt == 1:
                    break
            except (
                OSError,
                urllib.error.URLError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                state = {
                    "status": "error",
                    "reply": None,
                    "error": str(exc),
                }
                if attempt == 1:
                    break
            time.sleep(0.8)
        with self._lock:
            self._state = state
