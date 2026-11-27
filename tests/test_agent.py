from types import SimpleNamespace

from so101_agent.camera import CameraService
from so101_agent.config import load_settings
from so101_agent.qwen_agent import QwenRobotAgent
from so101_agent.robot import ArmController


class FakeMessage(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        payload = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": self.tool_calls,
        }
        if exclude_none:
            payload = {key: value for key, value in payload.items() if value is not None}
        return payload


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="move_joint",
                    arguments='{"joint":"elbow_flex","delta_degrees":2}',
                ),
            )
            message = FakeMessage(content=None, tool_calls=[tool_call])
        else:
            message = FakeMessage(content="我看到了红色方块和蓝色杯子，并稍微抬起了机械臂。", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


class StaticCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = FakeMessage(content="我看到了红色方块和蓝色杯子。", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_executes_tool_and_returns_text() -> None:
    settings = load_settings("config.toml")
    settings.camera.backend = "demo"
    arm = ArmController(settings.robot)
    camera = CameraService(settings.camera)
    arm.connect()
    camera.start()
    fake_client = FakeClient()
    try:
        agent = QwenRobotAgent(
            settings.llm,
            arm,
            camera,
            max_turn_motion_degrees=30,
            client=fake_client,
        )
        before = arm.positions()["elbow_flex"]
        response = agent.chat("抬一下机械臂")
        assert "红色方块" in response.text
        assert len(response.actions) == 1
        assert response.actions[0]["tool"] == "move_joint"
        assert arm.positions()["elbow_flex"] > before
        assert len(fake_client.chat.completions.calls) == 2
    finally:
        camera.stop()
        arm.disconnect()


def test_agent_sends_audio_and_wrist_image_in_one_turn() -> None:
    settings = load_settings("config.toml")
    settings.camera.backend = "demo"
    arm = ArmController(settings.robot)
    camera = CameraService(settings.camera)
    arm.connect()
    camera.start()
    completions = StaticCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    try:
        agent = QwenRobotAgent(
            settings.llm,
            arm,
            camera,
            max_turn_motion_degrees=30,
            client=fake_client,
        )
        wav = b"RIFF" + b"\x00" * 40
        response = agent.chat("", wav)
        assert "红色方块" in response.text
        content = next(
            message["content"]
            for message in completions.calls[0]["messages"]
            if message["role"] == "user" and isinstance(message["content"], list)
        )
        types = {item["type"] for item in content}
        assert types == {"input_audio", "image_url", "text"}
        assert completions.calls[0]["model"] == "qwen3.8-omni-flash"
        assert completions.calls[0]["tools"]
    finally:
        camera.stop()
        arm.disconnect()

