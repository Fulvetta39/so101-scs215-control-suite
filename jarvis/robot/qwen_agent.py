from __future__ import annotations

import base64
import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .camera import CameraService
from .config import LLMSettings
from .prompts import SYSTEM_PROMPT, tool_definitions
from .robot import ArmController
from .tools import ArmToolbox, tool_result_json

LOGGER = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Raised when the multimodal agent cannot complete a turn."""


@dataclass(slots=True)
class AgentResponse:
    text: str
    actions: list[dict]
    rounds: int


class QwenRobotAgent:
    def __init__(
        self,
        settings: LLMSettings,
        arm: ArmController,
        camera: CameraService,
        max_turn_motion_degrees: float,
        client: Any | None = None,
    ):
        self.settings = settings
        self.arm = arm
        self.camera = camera
        self.client = client or self._make_client()
        self.toolbox = ArmToolbox(arm, camera, max_turn_motion_degrees)
        self.history: deque[dict[str, str]] = deque(
            maxlen=max(2, settings.history_turns * 2)
        )
        self._turn_lock = threading.Lock()
        self.tools = tool_definitions()

    def _make_client(self) -> OpenAI:
        api_key = os.getenv(self.settings.api_key_env, "").strip()
        if not api_key:
            raise AgentError(
                f"未设置环境变量 {self.settings.api_key_env}。请参考 .env.example 配置 API Key。"
            )
        return OpenAI(
            api_key=api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
        )

    @property
    def ready(self) -> bool:
        return self.client is not None

    def chat(self, text: str, audio_wav: bytes | None = None) -> AgentResponse:
        if not self._turn_lock.acquire(blocking=False):
            raise AgentError("上一轮对话仍在执行，请稍候。")
        try:
            self.toolbox.begin_turn()
            image = self.camera.jpeg()
            positions = self.arm.positions()
            poses = self.arm.poses.names()
            context = {
                "current_controller_positions": {
                    key: round(value, 2) for key, value in positions.items()
                },
                "position_units": (
                    "身体关节使用 LeRobot 归一化位置 -100..100；"
                    "move_joint 的 delta_degrees 已由程序转换成控制单位。"
                ),
                "available_pose_names": poses,
                "robot_backend": self.arm.settings.backend,
                "emergency_stop_active": self.arm.stopped,
            }

            content: list[dict[str, Any]] = []
            if audio_wav:
                encoded_audio = base64.b64encode(audio_wav).decode("ascii")
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:audio/wav;base64,{encoded_audio}",
                            "format": "wav",
                        },
                    }
                )
            encoded_image = base64.b64encode(image).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                }
            )
            user_text = (
                f"本轮用户文字：{text.strip() or '（无文字，请直接理解语音）'}\n"
                f"实时上下文：{json.dumps(context, ensure_ascii=False)}"
            )
            content.append({"type": "text", "text": user_text})

            messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(self.history)
            messages.append({"role": "user", "content": content})

            final_text = ""
            rounds = 0
            for rounds in range(1, self.settings.max_tool_rounds + 1):
                response = self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.1,
                    extra_body={"enable_thinking": False},
                )
                message = response.choices[0].message
                assistant_message = message.model_dump(exclude_none=True)
                if assistant_message.get("tool_calls") and not assistant_message.get("content"):
                    assistant_message["content"] = ""
                messages.append(assistant_message)
                if not message.tool_calls:
                    final_text = (message.content or "").strip()
                    break

                refresh_image = False
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result = self.toolbox.execute(name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result_json(result),
                        }
                    )
                    refresh_image = refresh_image or bool(result.get("refresh_image"))

                if refresh_image and rounds < self.settings.max_tool_rounds:
                    fresh_image = self.camera.jpeg()
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "这是执行刚才工具动作后的最新腕部相机画面，请继续判断。",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            "data:image/jpeg;base64,"
                                            + base64.b64encode(fresh_image).decode("ascii")
                                        )
                                    },
                                },
                            ],
                        }
                    )
            else:
                final_text = "这一轮工具调用次数达到上限，我已经停在当前状态。"

            if not final_text:
                final_text = "我收到了指令，但模型没有返回可朗读的答复。"

            input_summary = text.strip() or "（语音）"
            self.history.append({"role": "user", "content": input_summary})
            self.history.append({"role": "assistant", "content": final_text})
            return AgentResponse(
                text=final_text,
                actions=[event.as_dict() for event in self.toolbox.events],
                rounds=rounds,
            )
        except AgentError:
            raise
        except Exception as exc:
            LOGGER.exception("Agent turn failed")
            raise AgentError(f"多模态模型调用失败：{exc}") from exc
        finally:
            self._turn_lock.release()

    def reset_history(self) -> None:
        self.history.clear()

