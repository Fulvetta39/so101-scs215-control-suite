# JARVIS 视觉感知

这是方案 B“共同注意 + 语音命名示教”的视觉感知底座。当前版本在本地完成：

- 多人脸检测；
- 主目标人脸选择与平滑；
- 人脸中心误差输出，供后续机械臂视觉伺服；
- 双手 21 点关键点；
- 食指指向判断；
- 从指尖沿食指方向的二维屏幕射线；
- 标注画面、JSONL 日志和 MP4 录制。

## 独立环境

MediaPipe 安装在独立的 `.venv_perception`，不修改已验证的 LeRobot 环境。

```powershell
.\.venv_perception\Scripts\python.exe -m jarvis.perception.cli --source 0
```

无窗口测试：

```powershell
.\.venv_perception\Scripts\python.exe -m jarvis.perception.cli `
  --source synthetic --headless --max-frames 10 --jsonl data/perception_test.jsonl
```

保存标注视频：

```powershell
.\.venv_perception\Scripts\python.exe -m jarvis.perception.cli `
  --source 0 --record data/perception_demo.mp4
```

人脸居中“影子模式”（只显示建议动作，不控制机械臂）：

```powershell
.\run_perception.ps1 --source 1 --servo-shadow
```

上下方向最终采用 `wrist_flex` 小步、低速控制。实测单步 `+3°` 并等待 5 秒时，纵向误差
连续从 `0.4395` 降至 `0.4130`、再降至 `0.2102`。肩部闭环响应滞后，不用于垂直轴。

```powershell
.\run_perception.ps1 `
  --source 1 --servo-shadow --servo-vertical-joint wrist_flex
```

垂直执行默认禁用。显式测试时可使用 0.2 Hz 的垂直频率：

```powershell
.\run_perception.ps1 `
  --source 1 --servo-execute `
  --enable-vertical-servo --allow-experimental-vertical `
  --servo-vertical-joint wrist_flex --servo-vertical-rate-hz 0.2
```

当前负载下小于 3° 的反向脉冲可能被死区吞掉，因此实验模式使用：

```text
垂直死区：0.20
最小有效脉冲：6°
最大脉冲：6°
```

这属于 bang-bang 低速控制，不适合快速连续上下跟随。

已验证的一键双轴测试：

```powershell
.\run_face_tracking_2d.ps1
```

该脚本要求机器人服务已运行。按 `Q` 结束；默认最长约 60 秒。

阴影模式下会显示：

- `shoulder_pan` 建议修正角度；
- `wrist_flex` 建议修正角度；
- 绿色箭头表示下一步相机应转向的方向；
- 人脸位于中心死区时显示 `inside_deadband`；
- 人脸丢失或正在指向时立即停止建议。

方向符号先用 `--pan-sign -1|1` 和 `--vertical-sign -1|1` 校准。只有影子模式方向与真机
实际方向一致后，才可以把修正量交给硬件安全层。

## 指向转向影子模式

这一阶段只验证“人指哪里，机械臂应该往哪里转”，不会驱动机械臂。机器人服务必须先
运行，因为摄像头由服务占用，视觉程序通过 HTTP 获取画面：

```powershell
.\run_pointing_shadow.ps1
```

现场验证时：

1. 画面中央出现你的脸后，伸出右手食指，其余三指自然弯曲；
2. 先只向画面右侧指，观察黄色建议箭头和 `pan` 的正负；
3. 再向画面左侧指，`pan` 应变成相反的符号；
4. 慢慢向下、向上指，分别观察 `wrist` 的两个方向；
5. 按 `Q` 或 `Esc` 结束。

当前每一项建议最多 `3°`，并且经过平滑和单轴死区过滤。左右转向优先验证；上下方向
只记录建议，不直接跟随时，不会造成腕部连续抖动。任何指向丢失或置信度不足都会立即
输出 `move=false`。

四个方向都验证正确后，可以进行只控制水平轴的实机短步测试：

```powershell
.\run_pointing_execute.ps1
```

然后在浏览器打开 `http://127.0.0.1:8766/`。实机模式的安全限制是：

- 同一方向连续稳定 `0.8` 秒后才发送第一小步；
- `shoulder_pan` 每次最多 `3°`，最多 `1 Hz`；
- 默认每次抬手最多发送 `3` 小步；中期演示使用 `--pointing-max-steps 0` 取消累计步数上限；
- 默认不控制 `wrist_flex`，上下指向只显示建议；
- 指向丢失、置信度不足或冻结姿态时立即停止发送；
- 服务端仍会执行绝对关节范围、单次 `6°` 上限和请求间隔检查。

实机测试时手不要快速来回改变方向，完成一次左右转向后先让手离开画面，确认动作停止。
上下轴只有在左右轴稳定后再单独启用。

垂直轴单步验证脚本会锁定左右轴，每次抬手只允许 `wrist_flex` 执行一次最多 `6°` 的脉冲：

```powershell
.\run_pointing_vertical_execute.ps1
```

对应浏览器地址为 `http://127.0.0.1:8767/`。

实机脚本已经启用 `--name-pointed-object`。一次受限指向动作完成并等待约 `1.2` 秒后，
程序会把当前相机画面交给本地多模态 Agent，并把“画面中央看到了什么”的回复显示在网页
状态框中。该调用被明确要求只观察、不移动机械臂。

## 输出字段

- `primary_face.center_uv`：人脸中心归一化坐标 `[0, 1]`；
- `primary_face.center_error`：相对画面中心 `[-1, 1]`，可直接作为视觉伺服误差；
- `hands[].landmarks_uv`：21 个归一化关键点；
- `hands[].is_pointing`：食指伸直且其余手指收起的保守判断；
- `pointing.direction_uv`：二维指向单位向量；
- `pointing.end_uv`：指向射线离开画面的位置；
- `attention`：`face`、`pointing` 或 `searching`。

启用指向控制时，JSONL 中的 `pointing_gaze` 保存建议，`pointing_execution` 保存发送结果
和速率限制原因。

当前的指向射线是屏幕平面结果。接入机械臂“看向指向位置”时，需要使用已有点击标定文件
`camera_calibration.json`、`hand_eye_calibration.json`，再结合桌面平面求交和 IK；这部分
不应在未验证手眼变换时直接执行真机运动。

