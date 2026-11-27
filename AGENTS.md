# JARVIS 方案 B 项目规则

本仓库的最终目标是完整方案 B：“共同注意社交机器人 + 语音命名示教”，当前使用
SO101 和 Feetech SCS215。现有可运行版本只是视觉基线，不代表项目最终范围。

## 每次开始工作前

1. 先阅读：
   - `PROJECT_STATE.md`
   - `ROADMAP.md`
   - `TEAMMATE_HANDOFF.md`
   - `MIDTERM_DEMO.md`
2. 检查 `git status`，保留用户已有修改。
3. 运行 `.\.venv\Scripts\python.exe -m pytest -q`。
4. 需要检查硬件时先运行：

   ```powershell
   .\.venv\Scripts\python.exe handoff_doctor.py
   ```

## 当前技术约束

- 当前稳定基线只启用人脸跟踪和手指指向；语音 Agent 与物体命名默认停用。
- 继续完整项目时，按 `ROADMAP.md` 分阶段恢复语音、统一控制和连续示教，
  不能把当前视觉精简版误认为项目已经完成。
- 机械臂使用课程适配版 LeRobot SCS215，不要直接替换成官方 STS3215 版本。
- 串口和相机索引取决于当前电脑，以 `config.robot.toml` 和 `handoff_doctor.py` 为准。
- 视觉环境是 `.venv_perception`，机械臂和服务环境是 `.venv`。
- 感知模型位于 `jarvis/perception/models/`。
- 校准文件属于具体机械臂。没有明确理由时不要修改或重新校准。
- 当前视觉路线是二维指向闭环；不要声称已经完成桌面三维射线和 IK。

## 真机安全规则

- 用户未确认现场清空、急停可触达前，不发送机械臂动作。
- 同一时间只能运行一个机器人服务和一个机械臂控制者。
- 保持关节绝对限位、连续伺服目标超前限制、速度/加速度限制和网页急停。
- 机械臂异常、通信失败或连续伺服异常时，优先停止并报告，不要反复重试。
- 不要在对话、日志或代码中输出 API Key。
- 不保留或提交摄像头人像、人脸/手部关键点原始日志；需要指标时只保留聚合结果。

## 代码与测试规则

- 修改代码使用小范围补丁，不覆盖无关修改。
- 新增功能必须补测试；至少保证现有 35 项测试继续通过。
- 重要行为使用真机验证时，先做影子模式或单步测试。
- 不在未经授权的公共网络上暴露摄像头和机器人控制接口。
- 完成重要里程碑后更新 `PROJECT_STATE.md` 和 `ROADMAP.md`。

## 主要入口

```powershell
# 完整环境安装或检查
.\bootstrap_handoff.ps1
.\bootstrap_handoff.ps1 -VerifyOnly

# 机器人主服务
.\run.ps1 -Config config.robot.toml

# 中期视觉演示
.\run_midterm_demo.ps1 -Smooth
```

统一视觉入口：`http://127.0.0.1:8000/demo`  
视觉子服务：`http://127.0.0.1:8768/`  
旧语音页面：`http://127.0.0.1:8000/agent`（默认停用）
