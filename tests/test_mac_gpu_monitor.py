import plistlib
import subprocess
import unittest

from whisper_captioner.mac_gpu_monitor import (
    MacGpuSample,
    parse_powermetrics_gpu_power,
    read_ioreg_gpu_sample,
)


class MacGpuMonitorTest(unittest.TestCase):
    def test_reads_ioreg_performance_statistics(self):
        payload = plistlib.dumps(
            [
                {
                    "PerformanceStatistics": {
                        "Device Utilization %": 79,
                        "Renderer Utilization %": 76,
                        "Tiler Utilization %": 23,
                        "In use system memory": 1024 * 1024 * 900,
                        "Alloc system memory": 1024 * 1024 * 3800,
                    }
                }
            ]
        )

        def runner(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout=payload, stderr=b"")

        sample = read_ioreg_gpu_sample(runner)
        self.assertEqual(sample.utilization, 79)
        self.assertEqual(sample.renderer_utilization, 76)
        self.assertEqual(sample.memory_used_bytes, 1024 * 1024 * 900)

    def test_parses_powermetrics_milliwatts_and_watts(self):
        self.assertEqual(parse_powermetrics_gpu_power("GPU Power: 4321 mW"), 4.321)
        self.assertEqual(parse_powermetrics_gpu_power("GPU Power: 7.5 W"), 7.5)
        self.assertIsNone(parse_powermetrics_gpu_power("CPU Power: 1000 mW"))

    def test_formats_log_line(self):
        sample = MacGpuSample(
            utilization=81,
            renderer_utilization=80,
            tiler_utilization=19,
            memory_used_bytes=1024 * 1024 * 900,
            memory_allocated_bytes=1024 * 1024 * 4000,
            power_watts=5.25,
        )
        line = sample.format_log()
        self.assertIn("util 81%", line)
        self.assertIn("memory 900/4000 MiB", line)
        self.assertIn("power 5.25 W", line)


if __name__ == "__main__":
    unittest.main()
