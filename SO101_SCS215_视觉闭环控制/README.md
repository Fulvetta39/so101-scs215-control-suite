# ArUco 视觉伺服

该模块为 SO-101 提供眼在手上的 ArUco 视觉伺服、相机标定、手眼标定和四角点 IBVS 仿真。

## 功能

- 简单闭环：根据 ArUco 中心所在图像区域完成上下、左右修正，居中后用 PnP 和手眼标定估计 ArUco 平面，沿平面法线接近到设定间隙（`--clearance-mm`）。
- 四角点 IBVS：以四个角点组成的八维图像特征计算六自由度相机速度，可在纯仿真中验证收敛。
- 相机标定：从棋盘格图像估计针孔相机内参和畸变系数。
- 手眼标定：根据末端位姿与棋盘格观测估计 `T_gripper_camera`。
- 受保护真机运动：每次运动后重新观测，并在目标丢失、IK 不可达、关节越限、舵机报警或编码器无有效响应时停止。

## 统一入口

在项目根目录启动 `launcher.py`，选择“ArUco 视觉伺服”。菜单提供：

1. 软件、URDF 与标定文件检查；
2. 摄像头枚举；
3. 四角点 IBVS 纯仿真；
4. 相机和机械臂只读预检；
5. 简单版 ArUco 真机闭环。

真机闭环需要在统一入口输入大写 `REAL`。子程序仍会检查舵机型号、当前位置、限位、扭矩、视觉目标和首步规划。

## 独立命令

以下命令在项目根目录执行：

```powershell
# 离线检查
.\.venv\Scripts\python.exe -B SO101_SCS215_视觉闭环控制\aruco_visual_servo.py check

# 四角点 IBVS 纯仿真
.\.venv\Scripts\python.exe -B SO101_SCS215_视觉闭环控制\aruco_visual_servo.py simulate-ibvs

# 只读预检；串口留空时自动识别唯一 CH343 设备
.\.venv\Scripts\python.exe -B SO101_SCS215_视觉闭环控制\aruco_simple_servo.py --check --camera 0 --rotate 0
```

## 文件

- `aruco_visual_servo.py`：相机标定、ArUco 检测、动作规划及 IBVS 仿真。
- `aruco_simple_servo.py`：简单版真机闭环与安全停止逻辑。
- `hand_eye_capture.py`：手眼标定采集工具。
- `data/camera_calibration.json`：相机内参与畸变参数。
- `data/hand_eye_calibration.json`：相机到末端的固定变换。

## 配置与安全

仓库中的标定文件只代表一套已测试装置。更换摄像头、分辨率、镜头、相机支架或机械臂后，应重新标定并验证坐标方向。

- 运行前确认 ArUco 实际边长与 `--marker-size-mm` 一致。
- 摄像头画面颠倒时使用 `--rotate 180`，不要在标定后额外改变图像方向。
- 真机程序不自动恢复固定初始姿态，闭环从当前安全姿态开始。
- 运动期间保持机械臂周围无障碍物，并准备随时切断舵机电源。
- 程序退出时会尝试关闭六个舵机扭矩；仍应确认机械臂不会因重力突然下落。
