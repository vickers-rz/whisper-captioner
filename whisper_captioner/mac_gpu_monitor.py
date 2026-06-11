from __future__ import annotations

import os
import plistlib
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal


IOREG = "/usr/sbin/ioreg"
POWERMETRICS = "/usr/bin/powermetrics"
GPU_POWER_PATTERN = re.compile(r"GPU Power:\s*([\d.]+)\s*(mW|W)", re.IGNORECASE)


@dataclass(frozen=True)
class MacGpuSample:
    utilization: float
    renderer_utilization: float
    tiler_utilization: float
    memory_used_bytes: int
    memory_allocated_bytes: int
    power_watts: Optional[float] = None

    def format_log(self) -> str:
        used_mib = self.memory_used_bytes / (1024 * 1024)
        allocated_mib = self.memory_allocated_bytes / (1024 * 1024)
        power = f"{self.power_watts:.2f} W" if self.power_watts is not None else "unavailable"
        return (
            "Mac GPU | "
            f"util {self.utilization:.0f}% | renderer {self.renderer_utilization:.0f}% | "
            f"tiler {self.tiler_utilization:.0f}% | memory {used_mib:.0f}/{allocated_mib:.0f} MiB | "
            f"power {power}"
        )


def read_ioreg_gpu_sample(
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> MacGpuSample:
    result = runner(
        [IOREG, "-r", "-c", "AGXAccelerator", "-a"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=3,
    )
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(error or f"ioreg exited with status {result.returncode}")
    entries = plistlib.loads(result.stdout)
    for entry in entries:
        stats = entry.get("PerformanceStatistics")
        if not isinstance(stats, dict):
            continue
        return MacGpuSample(
            utilization=float(stats.get("Device Utilization %", 0)),
            renderer_utilization=float(stats.get("Renderer Utilization %", 0)),
            tiler_utilization=float(stats.get("Tiler Utilization %", 0)),
            memory_used_bytes=int(stats.get("In use system memory", 0)),
            memory_allocated_bytes=int(stats.get("Alloc system memory", 0)),
        )
    raise RuntimeError("AGXAccelerator performance statistics were not found")


def parse_powermetrics_gpu_power(output: str) -> Optional[float]:
    match = GPU_POWER_PATTERN.search(output)
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000 if match.group(2).lower() == "mw" else value


def read_powermetrics_gpu_power(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Optional[float]:
    result = runner(
        [
            "/usr/bin/sudo",
            "-n",
            POWERMETRICS,
            "--samplers",
            "gpu_power",
            "-i",
            "500",
            "-n",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=3,
    )
    if result.returncode != 0:
        return None
    return parse_powermetrics_gpu_power(result.stdout)


class MacGpuMonitor(QObject):
    sample_ready = Signal(str)
    notice = Signal(str)

    def __init__(self, interval_seconds: Optional[float] = None) -> None:
        super().__init__()
        configured = interval_seconds
        if configured is None:
            try:
                configured = float(os.getenv("WHISPER_CAPTIONER_GPU_MONITOR_INTERVAL", "3"))
            except ValueError:
                configured = 3.0
        self.interval_seconds = max(1.0, configured)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._power_probe_after = 0.0
        self._power_notice_emitted = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="mac-gpu-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=4)
        self._thread = None

    def _run(self) -> None:
        self.notice.emit(f"Mac GPU monitoring started (interval {self.interval_seconds:g}s)")
        while not self._stop_event.is_set():
            try:
                sample = read_ioreg_gpu_sample()
                now = time.monotonic()
                power = None
                if now >= self._power_probe_after:
                    power = read_powermetrics_gpu_power()
                    if power is None:
                        self._power_probe_after = now + 60
                        if not self._power_notice_emitted:
                            self.notice.emit(
                                "Mac GPU power is unavailable: macOS requires sudo authorization "
                                "for powermetrics; utilization and memory monitoring remain active."
                            )
                            self._power_notice_emitted = True
                    else:
                        self._power_probe_after = now
                sample = MacGpuSample(
                    utilization=sample.utilization,
                    renderer_utilization=sample.renderer_utilization,
                    tiler_utilization=sample.tiler_utilization,
                    memory_used_bytes=sample.memory_used_bytes,
                    memory_allocated_bytes=sample.memory_allocated_bytes,
                    power_watts=power,
                )
                self.sample_ready.emit(sample.format_log())
            except Exception as exc:
                self.notice.emit(f"Mac GPU monitoring sample failed: {exc}")
            self._stop_event.wait(self.interval_seconds)
        self.notice.emit("Mac GPU monitoring stopped")
