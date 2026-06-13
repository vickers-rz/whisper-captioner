from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROXY_PATH = ROOT / "scripts" / "nuc_faster_whisper_busy_proxy.py"


class FasterWhisperProxySourceTests(unittest.TestCase):
    def test_proxy_defaults_to_automatic_language_detection(self) -> None:
        source = PROXY_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        defaults = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for argument, default in zip(
                    node.args.args[-len(node.args.defaults):],
                    node.args.defaults,
                ):
                    if argument.arg != "language":
                        continue
                    if isinstance(default, ast.Constant):
                        defaults.append(default.value)
                    elif (
                        isinstance(default, ast.Call)
                        and default.args
                        and isinstance(default.args[0], ast.Constant)
                    ):
                        defaults.append(default.args[0].value)

        self.assertTrue(defaults)
        self.assertTrue(all(default == "auto" for default in defaults))

    def test_proxy_omits_auto_language_from_upstream_form(self) -> None:
        source = PROXY_PATH.read_text(encoding="utf-8")

        self.assertIn('language.strip().lower() not in {"", "auto"}', source)
        self.assertIn('form_data["language"] = language', source)
        self.assertIn('"vad_filter": "true"', source)


if __name__ == "__main__":
    unittest.main()
