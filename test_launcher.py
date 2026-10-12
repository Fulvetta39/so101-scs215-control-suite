from __future__ import annotations

import unittest

from launcher import (
    DEFAULT_CATALOG,
    build_command,
    confirm_action,
    default_values,
    find_action,
    find_task,
    load_catalog,
    validate_catalog,
)


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(DEFAULT_CATALOG)

    def test_all_modules_are_registered(self) -> None:
        self.assertEqual(
            [task["id"] for task in self.catalog["tasks"]],
            ["multi_joint", "ik", "voice", "vision", "calibration"],
        )

    def test_catalog_paths_and_actions_are_valid(self) -> None:
        self.assertEqual(validate_catalog(self.catalog, DEFAULT_CATALOG.parent), [])

    def test_every_module_has_a_non_real_action(self) -> None:
        for task in self.catalog["tasks"]:
            self.assertTrue(
                any(action["safety"] != "real" for action in task["actions"]),
                task["id"],
            )

    def test_menu_has_no_status_labels_or_simulation_actions(self) -> None:
        removed_action_ids = {"dry_run", "text_dry_run", "demo"}
        for task in self.catalog["tasks"]:
            self.assertNotIn("status", task)
            self.assertTrue(removed_action_ids.isdisjoint(
                action["id"] for action in task["actions"]
            ))
            self.assertTrue(all(
                "模拟" not in action["name"] for action in task["actions"]
            ))

class CommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(DEFAULT_CATALOG)

    def test_blank_optional_port_removes_flag_and_value(self) -> None:
        task = find_task(self.catalog, "ik")
        action = find_action(task, "diagnose")
        command, _ = build_command(task, action, {"port": ""})
        self.assertNotIn("--port", command)
        self.assertEqual(command[-1], "diagnose")

    def test_explicit_port_is_preserved(self) -> None:
        task = find_task(self.catalog, "multi_joint")
        action = find_action(task, "diagnose")
        command, _ = build_command(task, action, {"port": "COM7"})
        self.assertIn("--port", command)
        self.assertIn("COM7", command)

    def test_all_real_actions_keep_explicit_real_gate(self) -> None:
        for task in self.catalog["tasks"]:
            for action in task["actions"]:
                command, _ = build_command(task, action, default_values(action))
                if action["safety"] == "real":
                    self.assertIn("--real", command, f"{task['id']}/{action['id']}")
                else:
                    self.assertNotIn("--real", command, f"{task['id']}/{action['id']}")

    def test_calibration_monitor_uses_monitor_entry(self) -> None:
        task = find_task(self.catalog, "calibration")
        action = find_action(task, "monitor")
        command, _ = build_command(task, action)
        self.assertTrue(command[2].endswith("servo_monitor.py"))

    def test_calibration_writes_require_hardware_confirmation_class(self) -> None:
        task = find_task(self.catalog, "calibration")
        self.assertEqual(find_action(task, "torque_off")["safety"], "hardware")
        self.assertEqual(find_action(task, "write_limits")["safety"], "hardware")

    def test_hardware_action_requires_exact_confirmation(self) -> None:
        action = {"safety": "hardware"}
        quiet = lambda _message: None
        self.assertFalse(confirm_action(action, lambda _prompt: "hardware", quiet))
        self.assertTrue(confirm_action(action, lambda _prompt: "HARDWARE", quiet))

    def test_motion_action_requires_exact_confirmation(self) -> None:
        action = {"safety": "real"}
        quiet = lambda _message: None
        self.assertFalse(confirm_action(action, lambda _prompt: "real", quiet))
        self.assertTrue(confirm_action(action, lambda _prompt: "REAL", quiet))

    def test_visual_real_action_keeps_conservative_limits(self) -> None:
        task = find_task(self.catalog, "vision")
        action = find_action(task, "real")
        command, _ = build_command(task, action)
        self.assertEqual(command[command.index("--max-moves") + 1], "3")
        self.assertEqual(command[command.index("--max-total-cm") + 1], "0.8")


if __name__ == "__main__":
    unittest.main()
