# 给组员电脑上的 Codex

把下面整段文字粘贴到 Codex：

```text
你正在接手完整 JARVIS 方案 B 项目，不是只维护一个人脸/手指子模块。当前交接版本
只保留了稳定的人脸跟踪、手指指向、平滑伺服和急停恢复；这只是项目基线，语音 Agent、
统一注意控制、语音命名示教和最终测评仍需要继续完成。请先读取 AGENTS.md、
PROJECT_STATE.md、ROADMAP.md、TEAMMATE_HANDOFF.md 和 MIDTERM_DEMO.md。

1. 确认项目位于解压后的目录，Python 3.12 可用。
2. 运行 .\bootstrap_handoff.ps1 创建 .venv 和 .venv_perception、安装依赖、
   安装随包校准并运行测试。如果安装失败，逐项诊断，不要跳过失败。
   如果系统提示找不到 python，但项目中已经存在 .venv，请先运行
   .\bootstrap_handoff.ps1 -VerifyOnly，不要误判为环境缺失。
   如果 Python 已安装但不在 PATH，请找到实际 python.exe 后使用
   .\bootstrap_handoff.ps1 -PythonExe "C:\完整路径\python.exe"。
3. 当前纯视觉基线不需要 API Key。进入语音和物体命名阶段后，再让用户在本地 .env
   中填写自己的 DASHSCOPE_API_KEY，不要把 Key 打印在对话中。
4. 运行 .\.venv\Scripts\python.exe handoff_doctor.py，读取串口、相机索引、
   配置、校准和模型检查结果。
5. 根据检查结果修改 config.robot.toml 中真实的 COM 口和相机索引。修改前说明
   你选择该值的原因，不要直接打开机械臂运动。
6. 再次运行 pytest。确认通过后，请用户清空机械臂运动范围并确认急停可触达。
7. 用户确认现场安全后，启动：
   .\run.ps1 -Config config.robot.toml
   确认 /api/status 返回机械臂已连接、相机正常、voice_agent=false、急停未触发。
8. 再启动：
   .\run_midterm_demo.ps1 -Smooth
   打开 http://127.0.0.1:8000/demo。
9. 最后进行最小验收：人脸左右跟随、上下跟随、指向一次、急停一次。
   任何一步失败都停止运动并报告。
10. 环境恢复并通过最小验收后，继续完成整个项目，不能在视觉模块通过后停止。
    按 ROADMAP.md 执行：
    a. 完成 10 次指向统计、响应和恢复指标、演示视频；
    b. 恢复并验证语音 Agent；
    c. 统一人脸、指向、语音和其他模块的控制权，确保同一时间只有一个控制器；
    d. 实现 20–50 Hz 连续轨迹示教、语音命名和限速回放；
    e. 根据优先级完成三维指向；
    f. 完成最终稳定版、测试、视频和报告。
    每完成一个阶段，更新 PROJECT_STATE.md 和 ROADMAP.md，并运行测试。

安全要求：
- 不要分享或输出 API Key；
- 不要在用户确认现场安全前发送机械臂动作；
- 不要同时运行两个机械臂服务；
- 不要修改交接包中的校准文件，除非用户明确要求重新校准；
- 所有动作必须保持软件限位和急停可用。
```

## 不使用 Codex 时

- 启动：双击 `启动JARVIS.cmd`，输入 `READY`；
- 停止：双击 `停止JARVIS.cmd`。

## 只需要组员手动完成的步骤

1. 当前视觉基线不需要 API Key；恢复语音阶段时再填写自己的百炼 API Key。
2. 现场确认机械臂周围清空、急停可触达。
3. 在最终验收时观察动作是否正常。
