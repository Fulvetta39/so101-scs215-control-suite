# 本地中文语音控制

该模块使用 Silero VAD 和 sherpa-onnx Paraformer 在本地完成中文语音识别，再通过白名单解析器调用预设姿态或 IK 直线移动。

默认入口只运行语音识别，不连接机械臂。运动控制必须显式增加 `--real` 并通过终端确认。

## 安装

```powershell
..\.venv\Scripts\python.exe -m pip install -r ..\requirements.txt
..\.venv\Scripts\python.exe setup_models.py
```

模型会下载到 `models/`，该目录不纳入 Git 版本控制。

## 设备与识别

列出麦克风：

```powershell
..\.venv\Scripts\python.exe voice_control.py devices
```

仅识别语音：

```powershell
..\.venv\Scripts\python.exe voice_control.py listen --asr-only
```

指定麦克风：

```powershell
..\.venv\Scripts\python.exe voice_control.py listen --asr-only --device 1
```

## 指令示例

- `移动到姿态三`
- `执行往返演示`
- `向前移动一厘米`
- `向上移动零点五公分`
- `不要前进`
- `退出程序`

未识别、歧义、否定或超过距离上限的指令不会执行动作。

## 真机控制

只读诊断：

```powershell
..\.venv\Scripts\python.exe voice_control.py --port COM5 diagnose
```

真机语音控制：

```powershell
..\.venv\Scripts\python.exe voice_control.py --port COM5 listen --real --device 1
```

真机模式会先读取型号、位置、限位和扭矩，检查安全范围，写入并回读当前位置保持目标，然后才允许开启扭矩。退出、`Ctrl+C`、`Esc` 或异常时会尝试关闭六轴扭矩。

## 配置

- `poses.json`：由多关节模块维护的预设姿态。
- `joint_mapping.json`：IK 舵机映射。
- `voice_calibration.json`：仅用于语音控制的方向负载补偿。

负载、电源、紧固状态或机械结构变化后，应重新验证补偿数据。默认移动距离为 0.5 cm，单条语音指令的上限为 20 cm。
