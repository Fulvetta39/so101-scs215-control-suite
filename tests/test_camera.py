from jarvis.robot.camera import make_demo_frame


def test_demo_frame_is_jpeg() -> None:
    frame = make_demo_frame(640, 360)
    assert frame.startswith(b"\xff\xd8")
    assert frame.endswith(b"\xff\xd9")
    assert len(frame) > 10_000

