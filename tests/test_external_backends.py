from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from whisper_captioner.external_backends import (
    CliAlignmentBackend,
    run_omnivad_shadow,
)


class ExternalBackendTests(unittest.TestCase):
    def test_alignment_cli_missing_is_reported(self) -> None:
        backend = CliAlignmentBackend("/definitely/missing/lai")
        self.assertFalse(backend.available)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            backend.align(Path("/tmp/audio.wav"), "text", Path("/tmp/output"))

    def test_omnivad_missing_falls_back_without_exception(self) -> None:
        with patch.dict(
            "os.environ",
            {"WHISPER_CAPTIONER_OMNIVAD_COMMAND": "/missing/omnivad {audio} -o {output}"},
        ):
            result = run_omnivad_shadow(Path("/tmp/audio.wav"), Path("/tmp/output"))
        self.assertEqual(result.status, "unavailable")
        self.assertTrue(result.warning)

    def test_omnivad_json_tiers_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "fake-omnivad"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "omnivad-shadow.json").write_text(
                json.dumps(
                    {
                        "tiers": {
                            "VAD": [
                                {"start": 1.0, "end": 2.5, "label": "speech"}
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "WHISPER_CAPTIONER_OMNIVAD_COMMAND": (
                        f"{executable} {{audio}} -o {{output}}"
                    )
                },
            ):
                result = run_omnivad_shadow(root / "audio.wav", output_dir)
        self.assertEqual(result.status, "completed")
        self.assertEqual((result.regions[0].start, result.regions[0].end), (1.0, 2.5))


if __name__ == "__main__":
    unittest.main()
