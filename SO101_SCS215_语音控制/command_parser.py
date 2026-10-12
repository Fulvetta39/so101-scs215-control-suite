"""Conservative text-to-command parsing for SO-101 voice control."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


DIRECTIONS = ("forward", "back", "left", "right", "up", "down")
DIRECTION_LABELS = {
    "forward": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "up": "上",
    "down": "下",
}


@dataclass(frozen=True)
class VoiceCommand:
    kind: str
    text: str
    target: str | None = None
    direction: str | None = None
    distance_cm: float | None = None
    reason: str = ""

    @property
    def executable(self) -> bool:
        return self.kind in {"pose", "action", "move", "stop", "exit"}

    def describe(self) -> str:
        if self.kind == "pose":
            return f"预设姿态：{self.target}"
        if self.kind == "action":
            return f"组合动作：{self.target}"
        if self.kind == "move":
            label = DIRECTION_LABELS.get(self.direction or "", self.direction or "?")
            return f"末端向{label}移动 {self.distance_cm:.2f} cm"
        if self.kind == "stop":
            return "停止并保持当前位置"
        if self.kind == "exit":
            return "退出程序并卸力"
        return f"不执行：{self.reason or '未识别到安全指令'}"


_PUNCTUATION = re.compile(r"[\s，。！？、,!.?;:；：]+")
_ARABIC_OR_CHINESE_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十点]+)"
_DISTANCE_RE = re.compile(
    rf"(?P<value>{_ARABIC_OR_CHINESE_NUMBER})\s*(?P<unit>毫米|mm|厘米|cm|公分|米|m)",
    re.IGNORECASE,
)

_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _chinese_integer(value: str) -> int:
    if not value:
        return 0
    if "十" in value:
        left, right = value.split("十", 1)
        tens = 1 if not left else _DIGITS[left]
        ones = 0 if not right else _DIGITS[right]
        return tens * 10 + ones
    result = 0
    for char in value:
        result = result * 10 + _DIGITS[char]
    return result


def parse_number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        pass
    if value.count("点") > 1:
        raise ValueError(f"无效数字：{value}")
    integer, dot, fraction = value.partition("点")
    result = float(_chinese_integer(integer))
    if dot:
        decimal = "".join(str(_DIGITS[char]) for char in fraction)
        result += float(f"0.{decimal}")
    return result


def parse_distance_cm(text: str) -> float | None:
    match = _DISTANCE_RE.search(text)
    if not match:
        return None
    value = parse_number(match.group("value"))
    unit = match.group("unit").lower()
    if unit in {"毫米", "mm"}:
        return value / 10.0
    if unit in {"米", "m"}:
        return value * 100.0
    return value


class CommandParser:
    """Parse only commands that can be mapped to an allow-listed robot action."""

    _DIRECTION_ALIASES = {
        "forward": ("向前移动", "往前移动", "向前", "往前", "前进", "forward"),
        "back": ("向后移动", "往后移动", "向后", "往后", "后退", "倒退", "backward"),
        "left": ("向左移动", "往左移动", "向左", "往左", "左移", "moveleft"),
        "right": ("向右移动", "往右移动", "向右", "往右", "右移", "moveright"),
        "up": ("向上移动", "往上移动", "向上", "往上", "上移", "moveup"),
        "down": ("向下移动", "往下移动", "向下", "往下", "下移", "movedown"),
    }
    _OPPOSITE = {
        "forward": "back",
        "back": "forward",
        "left": "right",
        "right": "left",
        "up": "down",
        "down": "up",
    }

    def __init__(
        self,
        pose_names: Iterable[str],
        action_names: Iterable[str],
        *,
        default_distance_cm: float = 0.5,
        max_distance_cm: float = 20.0,
    ) -> None:
        if default_distance_cm <= 0 or max_distance_cm <= 0:
            raise ValueError("默认距离和最大距离必须大于 0")
        if default_distance_cm > max_distance_cm:
            raise ValueError("默认距离不能大于最大距离")
        self.pose_names = tuple(sorted(set(pose_names), key=len, reverse=True))
        self.action_names = tuple(sorted(set(action_names), key=len, reverse=True))
        self.default_distance_cm = float(default_distance_cm)
        self.max_distance_cm = float(max_distance_cm)
        self._pose_by_number = self._index_poses()

    def _index_poses(self) -> dict[int, str]:
        result: dict[int, str] = {}
        for name in self.pose_names:
            match = re.search(r"(\d+)$", name)
            if match:
                result[int(match.group(1))] = name
            elif name == "初始姿态":
                result[0] = name
        return result

    @staticmethod
    def normalize(text: str) -> str:
        return _PUNCTUATION.sub("", text.strip().lower())

    def parse(self, text: str) -> VoiceCommand:
        normalized = self.normalize(text)
        if not normalized:
            return VoiceCommand("none", text, reason="空文本")

        if any(word in normalized for word in ("退出程序", "关闭程序", "结束程序", "quit", "exit")):
            return VoiceCommand("exit", text)

        if any(
            word in normalized
            for word in ("停止", "停下", "别动", "不要动", "取消动作", "不要", "别", "stop")
        ):
            return VoiceCommand("stop", text, reason="检测到停止或否定表达")

        for name in self.pose_names:
            if name in normalized:
                return VoiceCommand("pose", text, target=name)

        pose_match = re.search(r"(?:姿态|姿势|位置)(\d+|[零〇一二两三四五六七八九十]+)", normalized)
        if pose_match:
            number = int(parse_number(pose_match.group(1)))
            name = self._pose_by_number.get(number)
            if name:
                return VoiceCommand("pose", text, target=name)
            return VoiceCommand("none", text, reason=f"不存在姿态 {number}")

        for name in self.action_names:
            if name in normalized:
                return VoiceCommand("action", text, target=name)
        if "往返" in normalized and "往返演示" in self.action_names:
            return VoiceCommand("action", text, target="往返演示")

        found = []
        compact_english = normalized.replace(" ", "")
        for direction, aliases in self._DIRECTION_ALIASES.items():
            if any(alias in compact_english for alias in aliases):
                found.append(direction)
        found = list(dict.fromkeys(found))
        if len(found) > 1:
            return VoiceCommand("none", text, reason="同时出现多个移动方向")
        if len(found) == 1:
            direction = found[0]
            if "反方向" in normalized:
                direction = self._OPPOSITE[direction]
            distance = parse_distance_cm(normalized)
            if distance is None:
                if re.search(_ARABIC_OR_CHINESE_NUMBER, normalized):
                    return VoiceCommand(
                        "none",
                        text,
                        reason="检测到数字但距离单位未识别",
                    )
                distance = self.default_distance_cm
            if distance <= 0:
                return VoiceCommand("none", text, reason="移动距离必须大于 0")
            if distance > self.max_distance_cm:
                return VoiceCommand(
                    "none",
                    text,
                    reason=f"距离 {distance:g} cm 超过语音指令上限 {self.max_distance_cm:g} cm",
                )
            return VoiceCommand("move", text, direction=direction, distance_cm=distance)

        return VoiceCommand("none", text, reason="没有匹配到白名单姿态、动作或末端移动")
