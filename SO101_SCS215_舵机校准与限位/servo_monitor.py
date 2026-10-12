"""Read-only live window for all six SO-101 SCS215 servos."""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from tkinter import ttk

from scs215_calibrate import (
    ADDR_POSITION,
    ADDR_TORQUE,
    Bus,
    SERVO_IDS,
    require_models,
    resolve_port,
)


class SixServoMonitor:
    def __init__(self, root: tk.Tk, requested_port: str | None) -> None:
        self.root = root
        self.requested_port = requested_port
        self.bus: Bus | None = None
        self.port = ""
        self.closed = False
        self.limits: dict[int, tuple[int, int]] = {}
        self.position_labels: dict[int, ttk.Label] = {}
        self.detail_labels: dict[int, ttk.Label] = {}

        self.root.title("SO-101 · 六舵机实时监视（只读）")
        self.root.geometry("900x650")
        self.root.minsize(760, 560)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui()
        self.root.after(20, self.poll)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="ID1–ID6 实时状态",
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(pady=(0, 4))
        ttk.Label(
            outer,
            text="只读取位置、限位和扭矩状态，不会开启扭矩或发送运动指令",
            font=("Microsoft YaHei UI", 10),
            foreground="#555555",
        ).pack(pady=(0, 12))

        grid = ttk.Frame(outer)
        grid.pack(fill="both", expand=True)
        for index, servo_id in enumerate(SERVO_IDS):
            card = ttk.LabelFrame(grid, text=f" ID{servo_id} ", padding=12)
            card.grid(
                row=index // 3,
                column=index % 3,
                padx=7,
                pady=7,
                sticky="nsew",
            )
            position = ttk.Label(
                card,
                text="----",
                font=("Consolas", 40, "bold"),
                foreground="#777777",
                anchor="center",
            )
            position.pack(fill="x", expand=True)
            detail = ttk.Label(
                card,
                text="等待连接……",
                font=("Microsoft YaHei UI", 10),
                justify="center",
                anchor="center",
            )
            detail.pack(fill="x", pady=(4, 0))
            self.position_labels[servo_id] = position
            self.detail_labels[servo_id] = detail

        for row in range(2):
            grid.rowconfigure(row, weight=1, uniform="servo-row")
        for column in range(3):
            grid.columnconfigure(column, weight=1, uniform="servo-column")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        self.connection_label = ttk.Label(
            footer,
            text="正在查找 CH343……",
            font=("Microsoft YaHei UI", 10),
            foreground="#555555",
        )
        self.connection_label.pack(side="left")
        ttk.Button(footer, text="关闭窗口", command=self.close).pack(side="right")

    def disconnect(self) -> None:
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None
        self.limits.clear()

    def connect(self) -> None:
        self.port = resolve_port(self.requested_port)
        bus = Bus(self.port, timeout=0.08)
        try:
            states = bus.snapshot()
            require_models(states)
        except Exception:
            bus.close()
            raise
        self.bus = bus
        self.limits = {state.id: (state.minimum, state.maximum) for state in states}

    def poll(self) -> None:
        if self.closed:
            return
        started = time.monotonic()
        try:
            if self.bus is None:
                self.connect()
            assert self.bus is not None

            for servo_id in SERVO_IDS:
                position = self.bus.read_u16(servo_id, ADDR_POSITION)
                torque = self.bus.read(servo_id, ADDR_TORQUE, 1)[0]
                minimum, maximum = self.limits[servo_id]
                inside = minimum <= position <= maximum
                if not inside:
                    color = "#b42318"
                    state_text = "超出限位"
                elif torque:
                    color = "#b54708"
                    state_text = "扭矩开启"
                else:
                    color = "#1261a0"
                    state_text = "扭矩关闭"
                self.position_labels[servo_id].configure(text=str(position), foreground=color)
                self.detail_labels[servo_id].configure(
                    text=f"限位 {minimum}–{maximum}\n{state_text}"
                )

            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.connection_label.configure(
                text=f"{self.port} · 已连接 · 本轮读取 {elapsed_ms} ms"
            )
            delay_ms = max(40, 250 - elapsed_ms)
        except Exception as exc:
            self.disconnect()
            for servo_id in SERVO_IDS:
                self.position_labels[servo_id].configure(text="----", foreground="#b42318")
                self.detail_labels[servo_id].configure(text="暂时无法读取")
            self.connection_label.configure(text=f"{exc} · 1秒后自动重连")
            delay_ms = 1000

        if not self.closed:
            self.root.after(delay_ms, self.poll)

    def close(self) -> None:
        self.closed = True
        self.disconnect()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="SCS215 六舵机实时只读监视窗口")
    parser.add_argument("--port", help="串口，例如 COM5；省略时自动识别 CH343")
    args = parser.parse_args()
    root = tk.Tk()
    SixServoMonitor(root, args.port)
    root.mainloop()


if __name__ == "__main__":
    main()
