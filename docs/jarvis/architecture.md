# JARVIS 代码架构（草案）

本文档描述 JARVIS 课程项目在本仓库中的重构方向。旧的 `SO101_SCS215_*` 模块在新代码稳定前保留，逐步迁移后再删除。

## 现状与问题

- `SCS215Bus`、`find_port`、`ServoState` 在 `IK键盘直线控制`、`多关节连续控制` 等模块中各写了一份，协议细节和安全检查分散在各处。
- 模块之间通过 `sys.path.insert` 互相导入（例如 `aruco_simple_servo.py` 导入 `ik_keyboard_control`），目录名含中文，无法作为 Python 包使用。
- 相机、机械臂都直接操作硬件，没有可回放的实现，算法无法在云端离线测试。

## 目标结构

```text
jarvis/                     # 新的 Python 包
  hal/
    arm.py                  # Arm 接口：read_joints / command_joints / torque / estop
    scs215.py               # SCSCL 协议与 SCS215 真机实现（合并现有的多份 Bus 实现）
    camera.py               # Camera 接口：read() -> Frame(image, t, intrinsics)
    opencv_camera.py        # 真机相机
    replay.py               # 回放实现：按时间戳回放日志中的关节与图像
    sim.py                  # 纯运动学模拟臂（无图像，用于单元测试）
  safety.py                 # 关节限位、速度/加速度限制、工作空间盒、急停
  kinematics.py             # URDF FK/IK（从 ik_core.py 迁移）
  calibration.py            # 关节零点、相机内参、T_gripper_camera 的读写与校验
  perception/
    detector.py             # VLM 语义定位（API）
    tracker.py              # 本地快速追踪
    localize.py             # 射线-桌面求交、多视角三角化
  skills/                   # 确定性的动作技能
    scan.py  track.py  point.py  touch.py  grasp.py
  memory.py                 # 空间记忆：物品 -> 位置、时间、置信度
  agent/
    dialog.py               # LLM + function calling，工具即 skills
    speech.py               # ASR / TTS（API，本地 sherpa-onnx 兜底）
  logging.py                # 运行日志（见下）
  config.py                 # 读取仓库默认配置 + 本地覆盖
scripts/                    # 上机测试脚本，每个都自动记录日志
tests/                      # 不访问硬件的单元测试（用 sim / replay）
config/
  default.yaml              # 可提交的默认参数（限位、速度上限等）
  local.example.yaml        # 本地配置模板：COM 口、相机编号、API key 的变量名
```

## 关键约定

### HAL

`Arm` 和 `Camera` 都是 Protocol 接口，上层代码只依赖接口。三种实现：

| 实现 | 用途 |
|---|---|
| `SCS215Arm` / `OpenCVCamera` | 本地真机 |
| `ReplayArm` / `ReplayCamera` | 用真机日志在云端复现问题、回归测试感知算法 |
| `SimArm` | 无硬件的运动学模拟，用于单元测试和规划验证 |

### 安全

所有运动指令都经过 `safety.py`，真机实现不暴露绕过它的写入接口：

- 软件关节限位（比 EEPROM 限位更严）、单步最大关节变化、最大速度；
- 末端工作空间盒，以及桌面以下的禁区；
- 急停：按键或异常即卸力，并写入日志；
- 通信超时、编码器读数跳变、过载报警时停止运动。

沿用现有的确认习惯：运动类脚本需要显式加 `--real` 参数。

### 日志

每次运行生成 `outputs/runs/<时间戳>_<脚本名>/`：

- `meta.json`：git commit、配置快照（去掉密钥）、硬件信息；
- `joints.jsonl`：指令与反馈（含时间戳）；
- `frames/`：抽样图像，帧索引与时间戳对应；
- `events.jsonl`：VLM/LLM 请求与返回、技能开始/结束、错误。

这个格式就是 `replay.py` 的输入，上机后把目录打包发回即可离线复现。

### 配置与密钥

- `config/local.yaml` 和 `.env` 不进仓库（`.gitignore` 已覆盖 `.env` 和 `config.local.*`，需要补上 `config/local.yaml`）。
- API key 只从环境变量读取。
- 标定文件记录适用的机械臂，换机器后需要重新验证。

## 迁移顺序

1. `hal/scs215.py` + `safety.py` + `logging.py`，用现有的 `diagnose` 只读流程验证；
2. 迁移 `kinematics.py`、`calibration.py`，复用现有的单元测试；
3. 感知与技能；
4. 对话 Agent 与空间记忆；
5. 新代码覆盖旧功能后，删除旧模块，并更新 `launcher.py`。
