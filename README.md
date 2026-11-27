# JARVIS：SO101 共同注意社交机器人（方案B）

复旦《具身智能引论》课程项目。SO101 机械臂（Feetech SCS215 舵机）+ 腕部相机：
看着说话的人、顺着手指看过去、说出看到的东西，并能用语音命名和回放示教动作。

当前进度、已知限制和路线见 [PROJECT_STATE.md](PROJECT_STATE.md) 与 [ROADMAP.md](ROADMAP.md)；
现场操作见 [MIDTERM_DEMO.md](MIDTERM_DEMO.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `jarvis/robot/` | 机器人服务（FastAPI，端口 8000）、关节控制器、停止/急停、Qwen Agent |
| `jarvis/perception/` | MediaPipe 人脸/手部、指向几何、视觉伺服规划（独立环境 `.venv_perception`） |
| `web/` | 统一控制台 `http://127.0.0.1:8000/demo` |
| `config.robot.toml` | 真机配置（COM 口、相机索引、关节换算） |
| `calibration/` | 这台 SO101 的 LeRobot 关节校准 |
| `tests/` | 单元测试，不访问串口和相机 |
| `legacy/` | 2026-10 课程小任务的旧模块（舵机校准、IK、ArUco 视觉伺服、语音），只读保留 |

## 第一次在新电脑上运行

1. 把交接包里的 `jarvis/perception/models/` 两个模型文件复制到本仓库同一位置（模型不进 git）。
2. 运行 `.\bootstrap_handoff.ps1`：创建 `.venv` 和 `.venv_perception`、安装依赖、安装关节校准、跑测试。
3. 核对 `config.robot.toml` 的 COM 口和相机索引（`.\.venv\Scripts\python.exe handoff_doctor.py`）。
4. 清空机械臂周围后双击 `启动JARVIS.cmd`。

## 停止的两种方式

- **停止并保持**（网页红色按钮、`POST /api/stop`、语音“停”）：保持力矩，机械臂停在原地。
- **急停卸力**（`POST /api/estop`、`停止JARVIS.cmd`）：关闭力矩，大臂会下落。

两者都会锁住所有动作，确认现场安全后在网页点“确认安全并恢复”。
