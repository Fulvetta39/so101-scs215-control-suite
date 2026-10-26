# SCS215 校准、限位与状态监视

该模块提供六轴 SCS215 状态监视、手动范围测量和 EEPROM 限位管理。所有写入操作都要求扭矩关闭，并在写入后执行多次回读验证。

## 统一入口

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -B launcher.py run calibration monitor
```

统一启动器还提供诊断、关闭扭矩、范围测量、限位预览和限位写入。关闭扭矩或写 EEPROM 前需要输入 `HARDWARE`。

## 六轴只读监视器

在模块目录运行：

```powershell
..\.venv\Scripts\python.exe servo_monitor.py --port COM5
```

省略 `--port` 时会尝试识别唯一的 CH343 设备。界面约每 250 ms 刷新：

- 蓝色：位置在限位内且扭矩关闭；
- 橙色：扭矩开启；
- 红色：超出限位或通信异常。

监视器只读取数据。运行其他串口程序前应先关闭监视器。

## 校准流程

只读诊断：

```powershell
..\.venv\Scripts\python.exe scs215_calibrate.py --port COM5 diagnose
```

托住机械臂并关闭扭矩：

```powershell
..\.venv\Scripts\python.exe scs215_calibrate.py --port COM5 torque-off
```

手动缓慢移动指定关节并记录范围：

```powershell
..\.venv\Scripts\python.exe scs215_calibrate.py --port COM5 measure --seconds 90 --margin 5 --ids 1,2,3,4,5,6
```

测量不会命令机械臂运动，结果默认保存到 `records/limit_proposal.json`。

预览限位提案：

```powershell
..\.venv\Scripts\python.exe scs215_calibrate.py --port COM5 apply --proposal records/limit_proposal.json
```

确认后写入 EEPROM：

```powershell
..\.venv\Scripts\python.exe scs215_calibrate.py --port COM5 apply --proposal records/limit_proposal.json --confirm
```

写入前会验证型号、旧限位、当前位置和扭矩；写入后连续回读三次，并再次确认扭矩关闭。

## 安全要求

- 使用 6–7.4 V 独立直流舵机电源并共地（SCS215 规格上限 7.4 V），不要直接接充满电的 2S 锂电（8.4 V），禁止接入 12 V。
- 手动移动前托住机械臂并确认扭矩关闭。
- 不要撞击机械止挡，也不要在扭矩开启时手掰关节。
- EEPROM 提案中的旧限位必须与现场回读一致。
- 当前位置必须位于新的限位范围内。
- 不要将其他机械臂的限位文件直接写入当前设备。

## 测试

```powershell
..\.venv\Scripts\python.exe -B -m unittest -v test_scs215_calibrate.py
```
