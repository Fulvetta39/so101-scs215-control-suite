from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .camera import CameraError, CameraService
from .config import Settings, load_settings
from .qwen_agent import AgentError, QwenRobotAgent
from .robot import ArmController, RobotError

LOGGER = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    camera: CameraService
    arm: ArmController
    agent: QwenRobotAgent | None = None
    startup_errors: list[str] = field(default_factory=list)
    agent_error: str | None = None
    servo_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_servo_command_s: float = 0.0
    last_smooth_command_s: float = 0.0


class ServoCorrection(BaseModel):
    deltas: dict[str, float] = Field(min_length=1, max_length=3)


def build_runtime(settings: Settings) -> Runtime:
    load_dotenv(settings.root / ".env")
    camera = CameraService(settings.camera)
    arm = ArmController(settings.robot)
    runtime = Runtime(settings=settings, camera=camera, arm=arm)

    try:
        camera.start()
    except Exception as exc:
        runtime.startup_errors.append(f"相机启动失败：{exc}")
        LOGGER.exception("Camera startup failed")

    try:
        arm.connect()
    except Exception as exc:
        runtime.startup_errors.append(f"机械臂连接失败：{exc}")
        LOGGER.exception("Robot startup failed")

    if settings.llm.enabled:
        try:
            runtime.agent = QwenRobotAgent(
                settings.llm,
                arm,
                camera,
                max_turn_motion_degrees=settings.robot.max_turn_motion_degrees,
            )
        except Exception as exc:
            runtime.agent_error = str(exc)
            runtime.startup_errors.append(str(exc))
            LOGGER.warning("Agent model client is not ready: %s", exc)
    return runtime


def runtime_from_request(request: Request) -> Runtime:
    return request.app.state.runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    runtime = await asyncio.to_thread(build_runtime, settings)
    app.state.runtime = runtime
    yield
    await asyncio.to_thread(runtime.camera.stop)
    if runtime.arm.status()["connected"]:
        await asyncio.to_thread(runtime.arm.disconnect)


app = FastAPI(
    title="SO101 Omni Agent",
    description="Voice-and-vision control for an SO101 robot arm.",
    lifespan=lifespan,
)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse(url="/demo")


@app.get("/agent", include_in_schema=False)
async def legacy_agent() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/demo", include_in_schema=False)
async def unified_demo() -> FileResponse:
    return FileResponse(WEB_DIR / "demo.html")


@app.get("/api/status")
async def status(request: Request) -> dict:
    runtime = runtime_from_request(request)
    llm = runtime.settings.llm
    return {
        "ok": not runtime.startup_errors,
        "agent_ready": runtime.agent is not None,
        "agent_error": runtime.agent_error,
        "startup_errors": runtime.startup_errors,
        "features": {
            "voice_agent": runtime.settings.llm.enabled,
            "visual_tracking": True,
        },
        "model": llm.model,
        "base_url": llm.base_url,
        "camera": runtime.camera.status(),
        "robot": runtime.arm.status(),
    }


@app.get("/api/frame")
async def frame(request: Request) -> Response:
    runtime = runtime_from_request(request)
    try:
        jpeg = await asyncio.to_thread(runtime.camera.jpeg)
    except CameraError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    runtime = runtime_from_request(request)

    async def frames():
        boundary = b"--frame\r\n"
        while not await request.is_disconnected():
            try:
                jpeg = await asyncio.to_thread(runtime.camera.jpeg)
            except CameraError:
                await asyncio.sleep(0.5)
                continue
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            await asyncio.sleep(1.0 / max(1, runtime.settings.camera.fps))

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/api/chat")
async def chat(
    request: Request,
    text: str = Form(default=""),
    audio: UploadFile | None = File(default=None),
) -> dict:
    runtime = runtime_from_request(request)
    if runtime.agent is None:
        raise HTTPException(
            status_code=503,
            detail=runtime.agent_error or "多模态 Agent 尚未就绪。",
        )
    if not text.strip() and audio is None:
        raise HTTPException(status_code=400, detail="请提供语音或文字指令。")

    wav: bytes | None = None
    if audio is not None:
        if audio.content_type and not (
            audio.content_type.startswith("audio/")
            or audio.content_type == "application/octet-stream"
        ):
            raise HTTPException(status_code=415, detail="上传内容不是音频文件。")
        wav = await audio.read()
        if len(wav) < 44:
            raise HTTPException(status_code=400, detail="录音为空，请重新录制。")
        if len(wav) > 9_500_000:
            raise HTTPException(status_code=413, detail="录音过长，请控制在约 60 秒以内。")

    try:
        result = await asyncio.to_thread(runtime.agent.chat, text, wav)
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RobotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "reply": result.text,
        "actions": result.actions,
        "rounds": result.rounds,
    }


@app.post("/api/stop")
async def stop(request: Request) -> dict:
    """Latched stop that keeps torque on and holds the current pose."""
    runtime = runtime_from_request(request)
    await asyncio.to_thread(runtime.arm.halt)
    return {"ok": True, "message": "已停止，机械臂保持当前位置。"}


@app.post("/api/estop")
async def estop(request: Request) -> dict:
    """Latched stop that removes torque. Loaded joints will drop."""
    runtime = runtime_from_request(request)
    await asyncio.to_thread(runtime.arm.emergency_stop)
    return {"ok": True, "message": "急停已触发，已卸力。"}


@app.post("/api/resume")
async def resume(request: Request) -> dict:
    runtime = runtime_from_request(request)
    async with runtime.servo_lock:
        try:
            result = await asyncio.to_thread(runtime.arm.resume_from_emergency)
        except RobotError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return result


@app.post("/api/servo")
async def servo(request: Request, correction: ServoCorrection) -> dict:
    runtime = runtime_from_request(request)
    allowed = {"shoulder_pan", "shoulder_lift", "wrist_flex"}
    if not set(correction.deltas).issubset(allowed):
        raise HTTPException(status_code=400, detail="视觉伺服只允许规定的关节。")
    if any(abs(float(value)) > 6.0 for value in correction.deltas.values()):
        raise HTTPException(status_code=400, detail="单次视觉伺服修正不能超过 6°。")

    async with runtime.servo_lock:
        now = time.monotonic()
        if now - runtime.last_servo_command_s < 0.12:
            raise HTTPException(status_code=429, detail="视觉伺服命令过密。")
        try:
            result = await asyncio.to_thread(runtime.arm.move_joints, correction.deltas)
        except RobotError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        runtime.last_servo_command_s = time.monotonic()
    return result


@app.post("/api/servo/smooth")
async def servo_smooth(request: Request, correction: ServoCorrection) -> dict:
    runtime = runtime_from_request(request)
    allowed = {"shoulder_pan", "shoulder_lift", "wrist_flex"}
    if not set(correction.deltas).issubset(allowed):
        raise HTTPException(status_code=400, detail="平滑伺服只允许规定的关节。")
    if any(abs(float(value)) > 6.0 for value in correction.deltas.values()):
        raise HTTPException(status_code=400, detail="单次平滑目标不能超过 6°。")

    async with runtime.servo_lock:
        now = time.monotonic()
        if now - runtime.last_smooth_command_s < 0.04:
            raise HTTPException(status_code=429, detail="平滑目标更新过密。")
        try:
            result = await asyncio.to_thread(
                runtime.arm.update_smooth_target,
                correction.deltas,
            )
        except RobotError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        runtime.last_smooth_command_s = time.monotonic()
    return result


@app.get("/api/poses")
async def poses(request: Request) -> dict:
    runtime = runtime_from_request(request)
    return {"poses": runtime.arm.poses.names()}


@app.post("/api/poses/{pose_name}")
async def save_pose(request: Request, pose_name: str) -> dict:
    runtime = runtime_from_request(request)
    try:
        pose = await asyncio.to_thread(runtime.arm.save_current_pose, pose_name)
    except RobotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "name": pose_name, "pose": pose}


@app.post("/api/history/reset")
async def reset_history(request: Request) -> dict:
    runtime = runtime_from_request(request)
    if runtime.agent is not None:
        runtime.agent.reset_history()
    return {"ok": True}

