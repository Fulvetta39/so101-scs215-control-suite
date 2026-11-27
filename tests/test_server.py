from pathlib import Path

from fastapi.testclient import TestClient

from so101_agent.server import app


def test_server_status_and_demo_frame(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "server.toml"
    config.write_text(
        """
[robot]
backend = "mock"
poses_file = "poses.json"

[camera]
backend = "demo"
width = 640
height = 360

[llm]
backend = "qwen"
api_key_env = "SO101_TEST_MISSING_KEY"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("SO101_CONFIG", str(config))
    monkeypatch.delenv("SO101_TEST_MISSING_KEY", raising=False)

    with TestClient(app) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code in (302, 307)
        assert root.headers["location"] == "/demo"

        status = client.get("/api/status")
        assert status.status_code == 200
        payload = status.json()
        assert payload["camera"]["mode"] == "demo"
        assert payload["robot"]["connected"] is True
        assert payload["agent_ready"] is False

        frame = client.get("/api/frame")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.content.startswith(b"\xff\xd8")

        demo = client.get("/demo")
        assert demo.status_code == 200
        assert "JARVIS Face" in demo.text
        assert "语音 Agent" not in demo.text

        chat = client.post("/api/chat", data={"text": "看看桌上有什么"})
        assert chat.status_code == 503
        assert "SO101_TEST_MISSING_KEY" in chat.json()["detail"]

        servo = client.post(
            "/api/servo",
            json={"deltas": {"shoulder_pan": 0.5}},
        )
        assert servo.status_code == 200
        assert servo.json()["ok"] is True

        smooth = client.post(
            "/api/servo/smooth",
            json={"deltas": {"shoulder_pan": 0.5}},
        )
        assert smooth.status_code == 200
        assert smooth.json()["tracking"] is True

        stopped = client.post("/api/stop")
        assert stopped.status_code == 200
        assert stopped.json()["ok"] is True

        resumed = client.post("/api/resume")
        assert resumed.status_code == 200
        assert resumed.json()["ok"] is True


def test_server_can_run_without_voice_agent(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "visual_only.toml"
    config.write_text(
        """
[robot]
backend = "mock"
poses_file = "poses.json"

[camera]
backend = "demo"
width = 320
height = 240

[llm]
enabled = false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("SO101_CONFIG", str(config))

    with TestClient(app) as client:
        payload = client.get("/api/status").json()
        assert payload["agent_ready"] is False
        assert payload["features"]["voice_agent"] is False
        assert payload["startup_errors"] == []

