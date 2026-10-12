"""Download the reference document's official Paraformer and Silero VAD models."""

from __future__ import annotations

import shutil
import hashlib
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
MODEL_NAME = "sherpa-onnx-paraformer-zh-small-2024-03-09"
MODEL_BASE_URL = f"https://huggingface.co/csukuangfj/{MODEL_NAME}/resolve/main"
MODEL_SHA256 = "3ef6c19369b912f7caf3cef8e545c5ccd1a33d9d7ec792a46668dc41c4b229ec"
VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    print(f"下载：{url}")
    if existing:
        print(f"从已保存的 {existing / 1024 / 1024:.1f} MB 继续。")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = existing > 0 and response.status == 206
            mode = "ab" if resumed else "wb"
            with partial.open(mode) as output:
                shutil.copyfileobj(response, output)
        partial.replace(destination)
    except BaseException:
        if partial.exists():
            print(f"下载中断，已保留 {partial.stat().st_size / 1024 / 1024:.1f} MB 分片。")
        raise


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = MODELS_DIR / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / "model.int8.onnx"
    tokens = model_dir / "tokens.txt"
    vad = MODELS_DIR / "silero_vad.onnx"
    if not model.is_file():
        download(f"{MODEL_BASE_URL}/model.int8.onnx?download=true", model)
    else:
        print("Paraformer ONNX 模型已存在，跳过下载。")
    if not tokens.is_file():
        download(f"{MODEL_BASE_URL}/tokens.txt?download=true", tokens)
    else:
        print("Paraformer tokens.txt 已存在，跳过下载。")
    if not vad.is_file():
        download(VAD_URL, vad)
    else:
        print("Silero VAD 模型已存在，跳过下载。")
    required = (model, tokens, vad)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("模型准备失败：" + "；".join(missing))
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Paraformer 模型 SHA256 校验失败：{digest}")
    legacy = MODELS_DIR / f"{MODEL_NAME}.tar.bz2.part"
    if legacy.is_file():
        legacy.unlink()
    print("语音模型准备完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
