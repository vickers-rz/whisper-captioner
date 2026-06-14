import unittest
from types import SimpleNamespace
from unittest.mock import patch

from whisper_captioner.app import MainWindow
from whisper_captioner.chrome_control import (
    ChromeMediaTab,
    _is_likely_media_url,
    chrome_get_url,
    chrome_media_tabs,
    run_chrome_script_for_url,
)


class ChromeControlTest(unittest.TestCase):
    def test_media_url_detection_rejects_internal_and_regular_pages(self):
        self.assertFalse(_is_likely_media_url("chrome://history/"))
        self.assertFalse(_is_likely_media_url("https://translate.google.com/"))
        self.assertTrue(
            _is_likely_media_url("https://www.youtube.com/watch?v=CgLo7dZ7tOU")
        )

    @patch("whisper_captioner.chrome_control._chrome_active_tabs")
    def test_get_url_falls_back_to_media_tab_in_another_window(self, active_tabs):
        active_tabs.return_value = [
            ChromeMediaTab("History", "chrome://history/"),
            ChromeMediaTab("Translate", "https://translate.google.com/"),
            ChromeMediaTab(
                "Target video",
                "https://www.youtube.com/watch?v=CgLo7dZ7tOU",
            ),
        ]

        self.assertEqual(
            chrome_get_url(),
            "https://www.youtube.com/watch?v=CgLo7dZ7tOU",
        )

    @patch("whisper_captioner.chrome_control._chrome_active_tabs")
    def test_multiple_media_windows_are_returned_for_selection(self, active_tabs):
        active_tabs.return_value = [
            ChromeMediaTab("First", "https://www.youtube.com/watch?v=first"),
            ChromeMediaTab("Translate", "https://translate.google.com/"),
            ChromeMediaTab("Second", "https://www.youtube.com/watch?v=second"),
        ]

        self.assertEqual(
            chrome_media_tabs(),
            [
                ChromeMediaTab("First", "https://www.youtube.com/watch?v=first"),
                ChromeMediaTab("Second", "https://www.youtube.com/watch?v=second"),
            ],
        )

    @patch("whisper_captioner.app.QMessageBox.warning")
    @patch("whisper_captioner.app.QInputDialog.getItem")
    @patch("whisper_captioner.app.chrome_media_tabs")
    def test_controlled_url_prompts_for_multiple_chrome_videos(
        self,
        media_tabs,
        get_item,
        warning,
    ):
        tabs = [
            ChromeMediaTab("First video", "https://www.youtube.com/watch?v=first"),
            ChromeMediaTab("Second video", "https://www.youtube.com/watch?v=second"),
        ]
        media_tabs.return_value = tabs
        get_item.return_value = (
            "2. Second video — https://www.youtube.com/watch?v=second",
            True,
        )

        class UrlInput:
            value = ""

            def text(self):
                return self.value

            def setText(self, value):
                self.value = value

        window = SimpleNamespace(
            controlled_thread=None,
            url_input=UrlInput(),
            queue=SimpleNamespace(currentItem=lambda: None),
            _last_auto_url="",
            log=lambda *_args: None,
            current_mode=lambda: SimpleNamespace(available=False, label="Unavailable"),
        )

        MainWindow.start_controlled_url(window)

        get_item.assert_called_once()
        self.assertEqual(
            window.url_input.value,
            "https://www.youtube.com/watch?v=second",
        )
        warning.assert_called_once()

    @patch("whisper_captioner.app.QInputDialog.getItem", return_value=("", False))
    @patch("whisper_captioner.app.chrome_media_tabs")
    def test_cancelling_chrome_video_selection_stops_cleanly(self, media_tabs, get_item):
        media_tabs.return_value = [
            ChromeMediaTab("First", "https://www.youtube.com/watch?v=first"),
            ChromeMediaTab("Second", "https://www.youtube.com/watch?v=second"),
        ]

        window = SimpleNamespace(
            controlled_thread=None,
            url_input=SimpleNamespace(text=lambda: ""),
            queue=SimpleNamespace(currentItem=lambda: None),
            _last_auto_url="",
            log=lambda *_args: None,
        )

        MainWindow.start_controlled_url(window)

        get_item.assert_called_once()

    @patch("whisper_captioner.app.validate_url_for_yt_dlp", return_value=(True, ""))
    @patch("whisper_captioner.app.chrome_media_tabs", return_value=[])
    def test_gemini_preflight_cancel_does_not_create_controlled_thread(
        self, _media_tabs, _validate
    ):
        window = SimpleNamespace(
            controlled_thread=None,
            controlled_worker=None,
            url_input=SimpleNamespace(
                text=lambda: "https://example.com/video",
                setText=lambda _value: None,
            ),
            queue=SimpleNamespace(currentItem=lambda: None),
            _last_auto_url="",
            log=lambda *_args: None,
            current_mode=lambda: SimpleNamespace(
                available=True,
                label="NUC",
                backend="nuc_asr",
                model_name="large-v3-turbo",
            ),
            _check_gemini_fusion_ready=lambda: (False, ""),
        )

        MainWindow.start_controlled_url(window)

        self.assertIsNone(window.controlled_thread)
        self.assertIsNone(window.controlled_worker)

    @patch("whisper_captioner.chrome_control.chrome_is_running", return_value=True)
    @patch("whisper_captioner.chrome_control.subprocess.run")
    def test_url_script_searches_inactive_tabs_without_activation(self, run, _running):
        run.return_value.stdout = "12.5\n"
        run.return_value.stderr = ""
        run.return_value.returncode = 0

        result = run_chrome_script_for_url(
            "https://example.com/video",
            "return 1",
            activate_tab=False,
        )

        script = run.call_args.args[0][-1]
        self.assertEqual(result, "12.5")
        self.assertIn("repeat with tabIndex from 1 to (count of tabs of w)", script)
        self.assertIn("set t to tab tabIndex of w", script)
        self.assertNotIn("set active tab index", script)

    @patch("whisper_captioner.chrome_control.chrome_is_running", return_value=True)
    @patch("whisper_captioner.chrome_control.subprocess.run")
    def test_url_script_activates_exact_tab_only_when_requested(self, run, _running):
        run.return_value.stdout = ""
        run.return_value.stderr = ""
        run.return_value.returncode = 0
        run_chrome_script_for_url("https://example.com/video", "return 1")
        script = run.call_args.args[0][-1]
        self.assertIn("set active tab index of w to tabIndex", script)


if __name__ == "__main__":
    unittest.main()
