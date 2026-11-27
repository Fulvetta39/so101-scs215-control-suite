from __future__ import annotations

import json
import threading
from dataclasses import dataclass

from .camera import CameraService
from .robot import ArmController, RobotError


@dataclass(slots=True)
class ToolEvent:
    tool: str
    arguments: dict
    ok: bool
    message: str
    refresh_image: bool = False

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "message": self.message,
        }


class ArmToolbox:
    def __init__(self, arm: ArmController, camera: CameraService, max_turn_motion_degrees: float):
        self.arm = arm
        self.camera = camera
        self.max_turn_motion_degrees = max_turn_motion_degrees
        self._motion_used = 0.0
        self._lock = threading.Lock()
        self.events: list[ToolEvent] = []

    def begin_turn(self) -> None:
        with self._lock:
            self._motion_used = 0.0
            self.events = []

    def _reserve_motion(self, distance: float) -> None:
        if self._motion_used + distance > self.max_turn_motion_degrees + 1e-6:
            raise RobotError(
                f"本轮累计动作 {self._motion_used + distance:.1f}° 超过安全上限 "
                f"{self.max_turn_motion_degrees:.1f}°。请观察并分下一轮继续。"
            )

    def execute(self, name: str, arguments: dict) -> dict:
        try:
            if name == "move_joint":
                delta = float(arguments["delta_degrees"])
                self._reserve_motion(abs(delta))
                result = self.arm.move_joint(str(arguments["joint"]), delta)
                self._motion_used += abs(float(result["delta_degrees"]))
                event = self._event(name, arguments, result, refresh_image=True)
            elif name == "set_gripper":
                result = self.arm.set_gripper(float(arguments["open_percent"]))
                event = self._event(name, arguments, result, refresh_image=True)
            elif name == "move_to_pose":
                pose_name = str(arguments["pose_name"])
                pose = self.arm.poses.load().get(pose_name)
                if pose is None:
                    raise RobotError(f"没有名为 {pose_name!r} 的校准姿态。")
                current = self.arm.positions()
                largest_units = max(abs(pose[joint] - current[joint]) for joint in pose)
                self._reserve_motion(
                    largest_units / self.arm.settings.control_units_per_degree
                )
                result = self.arm.move_to_pose(pose_name)
                self._motion_used += abs(float(result["delta_degrees"]))
                event = self._event(name, arguments, result, refresh_image=True)
            elif name == "observe_environment":
                self.camera.jpeg()
                positions = self.arm.positions()
                result = {
                    "ok": True,
                    "message": "已获取最新腕部相机画面。",
                    "positions": positions,
                    "refresh_image": True,
                }
                event = self._event(name, arguments, result, refresh_image=True)
            elif name == "emergency_stop":
                self.arm.emergency_stop()
                result = {"ok": True, "message": "急停已触发，机械臂已保持当前位置。"}
                event = self._event(name, arguments, result, refresh_image=False)
            else:
                raise RobotError(f"未知工具：{name}")
            self.events.append(event)
            result["refresh_image"] = event.refresh_image
            return result
        except Exception as exc:
            result = {"ok": False, "message": str(exc), "refresh_image": False}
            self.events.append(ToolEvent(name, arguments, False, str(exc)))
            return result

    def _event(
        self,
        name: str,
        arguments: dict,
        result: dict,
        refresh_image: bool,
    ) -> ToolEvent:
        return ToolEvent(
            tool=name,
            arguments=arguments,
            ok=bool(result.get("ok")),
            message=str(result.get("message", "")),
            refresh_image=refresh_image,
        )


def tool_result_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)

