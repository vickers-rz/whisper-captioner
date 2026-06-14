import unittest
from types import SimpleNamespace
from unittest.mock import patch

from whisper_captioner.app import MainWindow
from whisper_captioner.chrome_control import (
    ChromeMediaTab,
    _is_likely_media_url,
    chrome_get_url,
    chrome_media_tabs,
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


if __name__ == "__main__":
    unittest.main()
