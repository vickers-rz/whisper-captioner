import unittest

from PySide6.QtWidgets import QApplication

from whisper_captioner.chaptering import VideoChapter
from whisper_captioner.overlay import SubtitleOverlay


class OverlayChapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_chapter_overlay_tracks_time_and_emits_seek(self):
        overlay = SubtitleOverlay()
        chapters = [
            VideoChapter(0, "开场", "介绍"),
            VideoChapter(60, "主题", "正文"),
            VideoChapter(120, "结尾", "总结"),
        ]
        overlay.set_chapters(chapters)
        overlay.set_chapter_at_time(75)
        self.assertEqual(overlay.chapter_title_label.text(), "主题")
        self.assertEqual(overlay.previous_chapter_button.text(), "开场")
        self.assertEqual(overlay.next_chapter_button.text(), "结尾")
        sought = []
        overlay.chapter_seek_requested.connect(sought.append)
        overlay.next_chapter_button.click()
        self.assertEqual(sought, [120])
        overlay.close()


if __name__ == "__main__":
    unittest.main()
