# IK 笛卡尔直线控制

该模块根据 `SO101/so101_new_calib.urdf` 建立运动学链，使末端沿基座坐标系的前、后、左、右、上、下方向执行直线移动。

## 启动

在模块目录运行：

```powershell
..\.venv\Scripts\python.exe ik_keyboard_control.py
```

启动菜单提供：

- `A`：输入方向和距离的定距模式。
- `B`：按住按键连续移动的自由模式。
- `Esc`：退出并关闭扭矩。

自由模式按键：

- `W / S`：前 / 后
- `A / D`：左 / 右
- `↑ / ↓`：上 / 下
- `+ / -`：调整速度
- `P`：打印末端坐标
- `Q`：返回模式选择
- `Esc`：退出并关闭扭矩

## 诊断与运行

只读检查：

```powershell
..\.venv\Scripts\python.exe ik_keyboard_control.py --port COM5 diagnose
```

真机键盘控制：

```powershell
..\.venv\Scripts\python.exe ik_keyboard_control.py --port COM5 run --real
```

单次真机移动：

```powershell
..\.venv\Scripts\python.exe ik_keyboard_control.py --port COM5 move --direction up --distance-cm 0.2 --real
```

## 运动学方法

1. 从 URDF 解析 `base_link` 到 `gripper_frame_link` 的旋转关节链。
2. 通过齐次变换计算正运动学 `x=f(q)`。
3. 根据 `z_i × (x-o_i)` 构造位置雅可比矩阵。
4. 使用阻尼最小二乘求解关节增量：

   `Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e`

   实际实现是加权版本：按方向设置关节权重，接近限位的关节权重自动增大，并对步长做线搜索；只约束末端位置，不约束姿态。

5. 将笛卡尔直线拆分为默认 2 mm 的路点，并使用上一点的解作为下一点初值。
6. 在不可达边界处保留同方向上最后一个可达位置，不跳到其他路径。

## 标定与限制

`joint_mapping.json` 定义舵机原始值、关节零位、方向和软件保护范围。运行时允许范围取以下三者的交集：

- 软件保护范围；
- 舵机 EEPROM 限位；
- URDF 关节限位。

仓库中的映射来自一台已标定机械臂，仅可作为配置示例。其他机械臂必须重新验证零位、方向、限位和参考姿态。

夹爪 ID6 在笛卡尔移动中保持启动位置。控制器包含路点预规划、限位感知 IK 和负载补偿。跟随误差和舵机报警只在逐点发送时每 5 个路点检查一次；自由模式和定距移动使用连续发送，只在一段轨迹结束后回读检查，按住按键期间不检查。控制器不包含环境或自碰撞检测。

## 测试

```powershell
..\.venv\Scripts\python.exe -B -m unittest -v test_ik_keyboard_control.py
```
