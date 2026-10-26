# 多关节预设姿态与连续动作

该模块用于记录和执行六轴 SCS215 预设姿态，并将多个姿态组合为连续动作。舵机编号从底座到夹爪依次为 ID1–ID6。

## 只读诊断

```powershell
..\.venv\Scripts\python.exe multi_joint_control.py --port COM5 diagnose
```

该命令读取型号、位置、EEPROM 限位和扭矩（不读电压；看电压用 IK 模块的 `diagnose`），不发送目标位置，也不改变扭矩状态。省略 `--port` 时会尝试识别唯一的 CH343 设备。

## 记录姿态

保持扭矩关闭，手动将机械臂放到安全姿态后执行：

```powershell
..\.venv\Scripts\python.exe multi_joint_control.py --port COM5 record --name 抬起 --key 1 --transition 1.5
```

`record` 只读取当前位置并更新 `poses.json`。如果某个安全姿态确实跨越原始位置 `1023/0`，可使用 `--confirm-wrap-id ID` 显式记录该关节；这个参数不会绕过运行时路径保护。

## 运行

```powershell
..\.venv\Scripts\python.exe multi_joint_control.py --port COM5 run --real
```

交互模式：

- `A`：连续选择单个预设姿态。
- `B`：输入姿态序列，例如 `0134` 或 `0 1 3 4`，再设置各段时长。
- `Esc`：退出并关闭六轴扭矩。

也可以直接运行 `poses.json` 中保存的组合动作：

```powershell
..\.venv\Scripts\python.exe multi_joint_control.py --port COM5 run --real --action 动作名称
```

默认最大速度为每秒 220 个原始位置单位。程序使用平滑插值，并拒绝 ID1 数值差超过 511 的跨接缝目标，避免舵机沿错误方向长距离旋转。

## 安全机制

- 型号必须为 SCS215 `0x0523`。
- 目标位置必须位于现场 EEPROM 限位内。
- 开启扭矩前写入并回读当前位置保持目标。
- 运行时检查位置跟随误差和舵机报警。
- 正常退出或异常时尝试关闭六轴扭矩。

软件不执行碰撞检测。保存姿态和组合动作前，需要确认姿态本身及姿态之间的过渡路径安全。

## 测试

```powershell
..\.venv\Scripts\python.exe -B -m unittest -v test_multi_joint_control.py
```
