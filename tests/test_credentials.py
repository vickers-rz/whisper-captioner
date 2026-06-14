from __future__ import annotations

import unittest
from unittest.mock import patch

from whisper_captioner.credentials import delete_secret, load_secret, save_secret


class CredentialTests(unittest.TestCase):
    @patch("whisper_captioner.credentials._run_security")
    def test_keychain_read_write_delete(self, run_security) -> None:
        run_security.return_value.returncode = 0
        run_security.return_value.stdout = "secret\n"
        run_security.return_value.stderr = ""

        self.assertEqual(load_secret("WhisperCaptioner", "gemini-api-key"), "secret")
        save_secret("WhisperCaptioner", "gemini-api-key", "new-secret")
        delete_secret("WhisperCaptioner", "gemini-api-key")

        commands = [call.args[0][0] for call in run_security.call_args_list]
        self.assertEqual(
            commands,
            ["find-generic-password", "add-generic-password", "delete-generic-password"],
        )

    @patch("whisper_captioner.credentials._run_security")
    def test_missing_keychain_item_returns_empty(self, run_security) -> None:
        run_security.return_value.returncode = 44
        run_security.return_value.stdout = ""
        run_security.return_value.stderr = "The specified item could not be found."
        self.assertEqual(load_secret("WhisperCaptioner", "gemini-api-key"), "")
