"""Unified, safety-oriented launcher for the SO-101 control modules.

The launcher deliberately runs each task as a separate blocking subprocess.
That keeps the original projects independent and prevents two tasks from
opening the robot serial port at the same time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "task_catalog.json"
SAFETY_LEVELS = {"safe", "read_only", "hardware", "real"}
PLACEHOLDER = re.compile(r"^\{([a-z][a-z0-9_]*)\}$")


class CatalogError(ValueError):
    """Raised when the task catalogue cannot be used safely."""


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    issues = validate_catalog(catalog, path.parent)
    if issues:
        raise CatalogError("任务清单无效：\n- " + "\n- ".join(issues))
    return catalog


def _inside_root(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def validate_catalog(catalog: object, root: Path = ROOT) -> list[str]:
    issues: list[str] = []
    if not isinstance(catalog, dict):
        return ["顶层必须是 JSON 对象"]
    if catalog.get("version") != 1:
        issues.append("version 必须为 1")
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        issues.append("tasks 必须是非空数组")
        return issues

    task_ids: set[str] = set()
    for task_index, task in enumerate(tasks, start=1):
        prefix = f"tasks[{task_index}]"
        if not isinstance(task, dict):
            issues.append(f"{prefix} 必须是对象")
            continue
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            issues.append(f"{prefix}.id 缺失")
            task_id = prefix
        elif task_id in task_ids:
            issues.append(f"任务 id 重复：{task_id}")
        task_ids.add(task_id)

        working_directory = task.get("working_directory")
        entry = task.get("entry")
        if not isinstance(working_directory, str) or not working_directory:
            issues.append(f"{task_id}.working_directory 缺失")
            task_dir = None
        else:
            task_dir = _inside_root(root, working_directory)
            if task_dir is None:
                issues.append(f"{task_id}.working_directory 必须位于项目根目录内")
            elif not task_dir.is_dir():
                issues.append(f"{task_id} 的目录不存在：{working_directory}")
        if not isinstance(entry, str) or not entry:
            issues.append(f"{task_id}.entry 缺失")
        elif task_dir is not None and not (task_dir / entry).is_file():
            issues.append(f"{task_id} 的入口不存在：{working_directory}/{entry}")

        actions = task.get("actions")
        if not isinstance(actions, list) or not actions:
            issues.append(f"{task_id}.actions 必须是非空数组")
            continue
        action_ids: set[str] = set()
        for action_index, action in enumerate(actions, start=1):
            action_prefix = f"{task_id}.actions[{action_index}]"
            if not isinstance(action, dict):
                issues.append(f"{action_prefix} 必须是对象")
                continue
            action_id = action.get("id")
            if not isinstance(action_id, str) or not action_id:
                issues.append(f"{action_prefix}.id 缺失")
                action_id = action_prefix
            elif action_id in action_ids:
                issues.append(f"{task_id} 的操作 id 重复：{action_id}")
            action_ids.add(action_id)

            action_entry = action.get("entry")
            if action_entry is not None:
                if not isinstance(action_entry, str) or not action_entry:
                    issues.append(f"{task_id}/{action_id}.entry 必须是非空字符串")
                elif task_dir is not None and not (task_dir / action_entry).is_file():
                    issues.append(
                        f"{task_id}/{action_id} 的入口不存在："
                        f"{working_directory}/{action_entry}"
                    )

            safety = action.get("safety")
            if safety not in SAFETY_LEVELS:
                issues.append(f"{task_id}/{action_id} 的 safety 无效：{safety}")
            arguments = action.get("arguments")
            if not isinstance(arguments, list) or not all(
                isinstance(value, str) for value in arguments
            ):
                issues.append(f"{task_id}/{action_id}.arguments 必须是字符串数组")
                arguments = []
            inputs = action.get("inputs", [])
            if not isinstance(inputs, list):
                issues.append(f"{task_id}/{action_id}.inputs 必须是数组")
                inputs = []
            input_names = {
                value.get("name")
                for value in inputs
                if isinstance(value, dict) and isinstance(value.get("name"), str)
            }
            placeholders = {
                match.group(1)
                for argument in arguments
                if (match := PLACEHOLDER.fullmatch(argument))
            }
            missing = placeholders - input_names
            if missing:
                issues.append(
                    f"{task_id}/{action_id} 缺少输入定义：{', '.join(sorted(missing))}"
                )
            if safety == "real" and "--real" not in arguments:
                issues.append(f"{task_id}/{action_id} 是真机操作但没有 --real")
            if safety != "real" and "--real" in arguments:
                issues.append(f"{task_id}/{action_id} 非真机操作却包含 --real")
    return issues


def find_task(catalog: Mapping, task_id: str) -> dict:
    for task in catalog["tasks"]:
        if task["id"] == task_id:
            return task
    raise CatalogError(f"没有任务：{task_id}")


def find_action(task: Mapping, action_id: str) -> dict:
    for action in task["actions"]:
        if action["id"] == action_id:
            return action
    raise CatalogError(f"任务 {task['id']} 没有操作：{action_id}")


def default_values(action: Mapping) -> dict[str, str]:
    return {
        spec["name"]: str(spec.get("default", ""))
        for spec in action.get("inputs", [])
    }


def expand_arguments(action: Mapping, values: Mapping[str, str]) -> list[str]:
    specs = {spec["name"]: spec for spec in action.get("inputs", [])}
    result: list[str] = []
    for argument in action["arguments"]:
        match = PLACEHOLDER.fullmatch(argument)
        if not match:
            result.append(argument)
            continue
        name = match.group(1)
        spec = specs[name]
        value = str(values.get(name, spec.get("default", ""))).strip()
        if not value:
            if spec.get("required", False):
                raise CatalogError(f"{spec.get('label', name)}不能为空")
            if result and result[-1].startswith("--"):
                result.pop()
            continue
        result.append(value)
    return result


def project_python(root: Path = ROOT) -> Path:
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def build_command(
    task: Mapping,
    action: Mapping,
    values: Mapping[str, str] | None = None,
    root: Path = ROOT,
) -> tuple[list[str], Path]:
    working_directory = (root / task["working_directory"]).resolve()
    entry = (working_directory / action.get("entry", task["entry"])).resolve()
    merged = default_values(action)
    if values:
        merged.update({key: str(value) for key, value in values.items()})
    arguments = expand_arguments(action, merged)
    command = [str(project_python(root)), "-B", str(entry), *arguments]
    return command, working_directory


def _validate_input(spec: Mapping, value: str) -> str:
    if not value:
        if spec.get("required", False):
            raise ValueError("不能为空")
        return ""
    choices = spec.get("choices")
    if choices:
        allowed = {str(choice["value"]) for choice in choices}
        if value not in allowed:
            raise ValueError("不在允许范围内")
    input_type = spec.get("type", "text")
    if input_type == "int":
        number = int(value)
        if "minimum" in spec and number < int(spec["minimum"]):
            raise ValueError(f"不能小于 {spec['minimum']}")
        return str(number)
    if input_type == "float":
        number = float(value)
        if "minimum" in spec and number < float(spec["minimum"]):
            raise ValueError(f"不能小于 {spec['minimum']}")
        return str(number)
    return value


def collect_values(
    action: Mapping,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for spec in action.get("inputs", []):
        choices = spec.get("choices", [])
        if choices:
            print_fn(f"\n{spec['label']}：")
            for index, choice in enumerate(choices, start=1):
                print_fn(f"  {index}. {choice['label']} ({choice['value']})")
        default = str(spec.get("default", ""))
        default_hint = f" [{default}]" if default else " [留空自动选择]"
        while True:
            raw = input_fn(f"{spec['label']}{default_hint}：").strip()
            if not raw:
                raw = default
            if choices and raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < len(choices):
                    raw = str(choices[index]["value"])
            try:
                values[spec["name"]] = _validate_input(spec, raw)
                break
            except (TypeError, ValueError) as exc:
                print_fn(f"输入无效：{exc}")
    return values


def confirm_action(
    action: Mapping,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> bool:
    if action["safety"] in {"safe", "read_only"}:
        return True
    if action["safety"] == "hardware":
        print_fn("\n警告：该操作会修改舵机状态或 EEPROM。")
        print_fn("请先托住机械臂、确认目标设备和参数，并确保没有其他程序占用串口。")
        return (
            input_fn("输入大写 HARDWARE 继续，其他输入取消：").strip()
            == "HARDWARE"
        )
    print_fn("\n警告：即将启动机械臂运动控制。")
    print_fn("请确认机械臂周围无障碍物、只有本任务占用串口，并准备随时断电。")
    print_fn("子程序仍会执行自己的只读检查和现场确认。")
    return input_fn("输入大写 REAL 继续，其他输入取消：").strip() == "REAL"


def run_action(task: Mapping, action: Mapping, values: Mapping[str, str]) -> int:
    command, working_directory = build_command(task, action, values)
    print("\n启动命令：")
    print("  " + subprocess.list2cmdline(command))
    print(f"工作目录：{working_directory}")
    sys.stdout.flush()
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def print_catalog(catalog: Mapping) -> None:
    print(catalog.get("project_name", "SO-101 控制平台"))
    for task in catalog["tasks"]:
        print(f"\n- {task['id']}: {task['name']}")
        print(f"  {task['description']}")
        for action in task["actions"]:
            print(f"    {action['id']}: {action['name']} ({action['safety']})")


def check_project(catalog: Mapping, root: Path = ROOT) -> tuple[list[str], list[str]]:
    errors = validate_catalog(catalog, root)
    warnings: list[str] = []
    python_path = project_python(root)
    if not python_path.is_file():
        errors.append(f"Python 不存在：{python_path}")
    if python_path.resolve() == Path(sys.executable).resolve() and not (
        root / ".venv"
    ).is_dir():
        warnings.append("没有发现项目 .venv，将使用当前 Python")
    model = (
        root
        / "SO101_SCS215_语音控制"
        / "models"
        / "sherpa-onnx-paraformer-zh-small-2024-03-09"
        / "model.int8.onnx"
    )
    if not model.is_file():
        warnings.append("语音模型尚未下载；语音识别前请运行 setup_models.py")
    return errors, warnings


def interactive(catalog: Mapping) -> int:
    while True:
        print("\n" + "=" * 54)
        print(catalog.get("project_name", "SO-101 控制平台"))
        print("一次只运行一个任务；返回本菜单表示该任务已经结束。")
        print("=" * 54)
        for index, task in enumerate(catalog["tasks"], start=1):
            print(f"  {index}. {task['name']}")
        print("  0. 退出")
        selected = input("选择任务：").strip()
        if selected in {"0", "q", "Q"}:
            return 0
        if not selected.isdigit() or not 1 <= int(selected) <= len(catalog["tasks"]):
            print("请输入菜单中的数字。")
            continue
        task = catalog["tasks"][int(selected) - 1]
        while True:
            print(f"\n{task['name']}\n{task['description']}")
            for index, action in enumerate(task["actions"], start=1):
                marker = {
                    "safe": "离线",
                    "read_only": "只读",
                    "hardware": "硬件写入",
                    "real": "运动控制",
                }[
                    action["safety"]
                ]
                print(f"  {index}. {action['name']} [{marker}]")
            print("  0. 返回任务列表")
            selected_action = input("选择操作：").strip()
            if selected_action in {"0", "q", "Q"}:
                break
            if not selected_action.isdigit() or not 1 <= int(selected_action) <= len(
                task["actions"]
            ):
                print("请输入菜单中的数字。")
                continue
            action = task["actions"][int(selected_action) - 1]
            values = collect_values(action)
            if not confirm_action(action):
                print("已取消，未启动子程序。")
                continue
            try:
                return_code = run_action(task, action, values)
            except OSError as exc:
                print(f"启动失败：{exc}", file=sys.stderr)
                return_code = 1
            print(f"\n子程序已结束，退出码 {return_code}。")
            input("按回车返回当前任务菜单……")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO-101 控制平台统一入口")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="列出任务与操作，不启动子程序")
    sub.add_parser("check", help="检查入口和清单，不连接任何硬件")
    run = sub.add_parser("run", help="从命令行选择一个任务操作")
    run.add_argument("task_id")
    run.add_argument("action_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        catalog = load_catalog(args.catalog.resolve())
        if args.command == "list":
            print_catalog(catalog)
            return 0
        if args.command == "check":
            errors, warnings = check_project(catalog, args.catalog.resolve().parent)
            for warning in warnings:
                print(f"提示：{warning}")
            if errors:
                for error in errors:
                    print(f"错误：{error}", file=sys.stderr)
                return 1
            print(f"检查通过：{len(catalog['tasks'])} 个任务，统一入口未连接任何硬件。")
            return 0
        if args.command == "run":
            task = find_task(catalog, args.task_id)
            action = find_action(task, args.action_id)
            values = collect_values(action)
            if not confirm_action(action):
                print("已取消，未启动子程序。")
                return 0
            return run_action(task, action, values)
        return interactive(catalog)
    except KeyboardInterrupt:
        print("\n已中止。")
        return 130
    except (CatalogError, json.JSONDecodeError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
