"""Local microphone VAD + Paraformer ASR with optional dependencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


SAMPLE_RATE = 16_000


def _escape_pressed() -> bool:
    if os.name != "nt":
        return False
    import msvcrt

    while msvcrt.kbhit():
        if msvcrt.getwch() == "\x1b":
            return True
    return False


def _import_audio_dependencies():
    missing = []
    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")
    try:
        import sounddevice as sd
    except ImportError:
        sd = None
        missing.append("sounddevice")
    try:
        import sherpa_onnx
    except ImportError:
        sherpa_onnx = None
        missing.append("sherpa-onnx")
    if missing:
        raise RuntimeError(
            "缺少语音依赖："
            + "、".join(missing)
            + "。请在项目根目录执行 .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )
    return np, sd, sherpa_onnx


def list_input_devices() -> list[tuple[int, str, int, float]]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "尚未安装 sounddevice；请在项目根目录安装 requirements.txt"
        ) from exc
    result = []
    for index, device in enumerate(sd.query_devices()):
        channels = int(device.get("max_input_channels", 0))
        if channels > 0:
            result.append(
                (
                    index,
                    str(device.get("name", "")),
                    channels,
                    float(device.get("default_samplerate", 0.0)),
                )
            )
    return result


class MicrophoneASR:
    def __init__(
        self,
        model_dir: Path,
        vad_model: Path,
        *,
        device: int | None = None,
        num_threads: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.paraformer = self.model_dir / "model.int8.onnx"
        self.tokens = self.model_dir / "tokens.txt"
        self.vad_model = Path(vad_model)
        self.device = device
        self.num_threads = int(num_threads)
        missing = [
            str(path)
            for path in (self.paraformer, self.tokens, self.vad_model)
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "语音模型文件不完整："
                + "；".join(missing)
                + "。请按 README 的模型准备步骤操作。"
            )
        if self.num_threads <= 0:
            raise ValueError("识别线程数必须大于 0")

        self.np, self.sd, self.sherpa_onnx = _import_audio_dependencies()
        self.recognizer = self.sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(self.paraformer),
            tokens=str(self.tokens),
            num_threads=self.num_threads,
            decoding_method="greedy_search",
            debug=False,
        )
        config = self.sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self.vad_model)
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.6
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.max_speech_duration = 8.0
        config.sample_rate = SAMPLE_RATE
        self.vad = self.sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=30
        )
        self.window_size = int(config.silero_vad.window_size)

    def _decode(self, samples) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def listen(self) -> Iterator[str]:
        self.sd.check_input_settings(
            device=self.device, channels=1, dtype="float32", samplerate=SAMPLE_RATE
        )
        print("语音识别已启动，按 Esc 退出。")
        print("请输入语音指令：")
        with self.sd.InputStream(
            device=self.device,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
        ) as stream:
            while True:
                if _escape_pressed():
                    return
                samples, overflowed = stream.read(self.window_size)
                if overflowed:
                    print("提示：麦克风缓冲区曾溢出，本帧仍继续处理。")
                mono = self.np.asarray(samples[:, 0], dtype=self.np.float32)
                self.vad.accept_waveform(mono)
                while not self.vad.empty():
                    segment = self.np.asarray(
                        self.vad.front.samples, dtype=self.np.float32
                    )
                    self.vad.pop()
                    print("检测到一段语音，正在识别……")
                    text = self._decode(segment)
                    if text:
                        yield text
