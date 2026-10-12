# SO-101 SCS215 Control Suite

面向 Windows 的 SO-101 机械臂控制工具集，适配 Feetech SCS215 舵机。项目提供统一启动器，同时保留每个功能模块的独立命令行入口。

## 功能模块

| 模块 | 说明 |
| --- | --- |
| 多关节动作 | 记录、执行和组合预设姿态 |
| IK 直线控制 | 沿基座坐标系执行笛卡尔直线移动 |
| 语音控制 | 本地中文语音识别与白名单动作解析 |
| 视觉闭环 | 检测黑球并用笛卡尔微步进对准画面中心 |
| 校准与监视 | 六轴状态监视、范围测量和 EEPROM 限位管理 |

## 硬件与环境

- Windows 10/11
- Python 3.12
- SO-101 机械臂
- Feetech SCS215（SCSCL 协议，位置范围 0–1023）
- CH343 或兼容串口适配器
- 6–8.4 V 舵机电源，并与控制器共地
- 摄像头和麦克风仅在对应模块中需要

该项目不适用于 STS3215 的 0–4095 控制路径。

## 安装

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

语音识别模型不保存在 Git 仓库中，需要单独下载：

```powershell
.\.venv\Scripts\python.exe SO101_SCS215_语音控制\setup_models.py
```

## 统一启动器

双击：

```text
运行SO101控制平台.cmd
```

或在 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -B launcher.py
```

启动器一次只运行一个模块，避免多个进程同时占用串口。操作分为四类：

- **离线**：图像处理、语音识别或设备枚举，不访问机械臂。
- **只读**：读取舵机或传感器状态，不改变机械臂状态。
- **硬件写入**：关闭扭矩或写入 EEPROM，需要输入 `HARDWARE`。
- **运动控制**：可能驱动机械臂，需要输入 `REAL`，并继续通过模块内部的安全检查。

检查任务清单和入口文件，不连接硬件：

```powershell
.\.venv\Scripts\python.exe -B launcher.py check
```

列出所有模块与操作：

```powershell
.\.venv\Scripts\python.exe -B launcher.py list
```

也可以直接选择操作，例如打开六轴监视器：

```powershell
.\.venv\Scripts\python.exe -B launcher.py run calibration monitor
```

## 配置文件

- `SO101_SCS215_多关节连续控制/poses.json`：预设姿态和组合动作。
- `SO101_SCS215_IK键盘直线控制/joint_mapping.json`：舵机值与 URDF 关节角映射。
- `SO101_SCS215_语音控制/voice_calibration.json`：语音控制使用的负载补偿。
- `task_catalog.json`：统一启动器的模块与操作清单。

这些标定和姿态数据与具体机械臂相关。连接其他机械臂前，应先执行只读诊断并重新验证映射、限位和运动方向。

## 项目结构

```text
.
├─ launcher.py
├─ task_catalog.json
├─ SO101/                                  # URDF
├─ SO101_SCS215_多关节连续控制/
├─ SO101_SCS215_IK键盘直线控制/
├─ SO101_SCS215_语音控制/
├─ SO101_SCS215_视觉闭环控制/
└─ SO101_SCS215_舵机校准与限位/
```

各模块的详细参数见对应目录中的 README。

## 测试

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_launcher.py

Push-Location SO101_SCS215_多关节连续控制
..\.venv\Scripts\python.exe -B -m unittest -v test_multi_joint_control.py
Pop-Location

Push-Location SO101_SCS215_IK键盘直线控制
..\.venv\Scripts\python.exe -B -m unittest -v test_ik_keyboard_control.py
Pop-Location

Push-Location SO101_SCS215_视觉闭环控制
..\.venv\Scripts\python.exe -B -m unittest -v test_visual_closed_loop.py
Pop-Location

Push-Location SO101_SCS215_舵机校准与限位
..\.venv\Scripts\python.exe -B -m unittest -v test_scs215_calibrate.py
Pop-Location
```

单元测试不会连接机械臂。测试通过仅表示软件逻辑满足断言，不代表真实运动路径不存在碰撞。

## 添加模块

新模块可以保留独立目录和命令行入口，再在 `task_catalog.json` 中登记：

1. 设置模块目录、入口文件和操作参数。
2. 为每个操作指定 `safe`、`read_only`、`hardware` 或 `real`。
3. 真机运动操作必须显式包含 `--real`。
4. 为可变参数定义输入占位符，例如 `{port}`、`{camera}`。
5. 运行 `launcher.py check` 和 `test_launcher.py`。

## 安全须知

- 真机操作前读取舵机型号、位置、EEPROM 限位和扭矩状态。
- 开启扭矩前先写入并回读当前位置保持目标。
- 写 EEPROM 前关闭扭矩、预览提案并确认当前位置在新范围内。
- 保持机械臂周围无障碍物，并准备随时切断舵机电源。
- 软件无法检测桌面、人体、自碰撞、结构松动或打印件弯曲。
- 摄像头、麦克风、串口和标定值均应按实际设备重新确认。

协作与提交规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。
