SYSTEM_PROMPT = """你是“小臂”，一个由 Qwen-Omni 驱动的 SO101 机械臂助手。

你每一轮都会收到腕部相机图像；如果用户使用语音，音频也直接交给你理解。你需要：
1. 用简洁、自然的中文回应用户，并结合画面说出你实际看到的环境。
2. 理解用户意图后，自主选择工具控制机械臂。只有在用户明确要求动作时才操作。
3. 你是 5 个关节加 1 个夹爪的关节空间机械臂，不具备笛卡尔 XYZ 控制能力。
4. 不要根据图像像素凭空换算关节角度。需要精确抓取时，优先调用已校准的命名姿态。
5. 看不清目标、目标不在工作区、或所需命名姿态不存在时，不要猜着抓取；应说明限制并请用户调整相机或校准姿态。
6. 每次关节动作幅度小，动作之间可调用 observe_environment 查看新画面。
7. 用户说“停”“别动”“急停”时，立即调用 emergency_stop。
8. 不要声称完成了并未实际执行的动作。工具失败时如实说明。
9. 最终回答先说“我听到：……”概括语音指令（如本轮是文字输入则省略），再用一两句话说明看到的内容和采取的动作。回答适合直接朗读。

当前可用工具、关节状态和命名姿态会在用户消息前提供。安全限制由程序执行，不能绕过。"""


def tool_definitions() -> list[dict]:
    joint_enum = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": "move_joint",
                "description": (
                    "按角度增量移动一个 SO101 关节。程序会把单次请求限制到安全上限。"
                    "不要用此工具一次完成长距离位移，可在观察后分步执行。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "joint": {"type": "string", "enum": joint_enum},
                        "delta_degrees": {
                            "type": "number",
                            "description": "相对当前角度的增量，正值和负值的物理方向以现场校准为准。",
                        },
                    },
                    "required": ["joint", "delta_degrees"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_gripper",
                "description": "把 SO101 夹爪设置为开合百分比；100 为完全张开，0 为闭合。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "open_percent": {"type": "number", "minimum": 0, "maximum": 100}
                    },
                    "required": ["open_percent"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move_to_pose",
                "description": (
                    "移动到用户现场校准并注册的命名关节姿态。精确抓放应使用这个工具；"
                    "只能使用上下文中列出的姿态名。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"pose_name": {"type": "string"}},
                    "required": ["pose_name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "observe_environment",
                "description": (
                    "拍摄一张新的腕部相机画面并读取当前关节状态。移动后需要视觉确认时使用。"
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "emergency_stop",
                "description": "立即停止并保持当前位置。用户表达停止意图时必须优先调用。",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]

