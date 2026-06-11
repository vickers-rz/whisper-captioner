import unittest

from whisper_captioner.chaptering import (
    VideoChapter,
    add_chapters_to_subtitles,
    chapters_to_markdown,
    parse_chapters_response,
)
from whisper_captioner.models import SubtitleSegment


class ChapteringTest(unittest.TestCase):
    def test_parses_fenced_json_and_sorts_chapters(self):
        chapters = parse_chapters_response(
            """```json
            [
              {"start_seconds": 60, "title": "第二章", "description": "说明"},
              {"start_seconds": 0, "title": "第一章", "description": ""}
            ]
            ```"""
        )
        self.assertEqual([chapter.title for chapter in chapters], ["第一章", "第二章"])

    def test_rejects_missing_title(self):
        with self.assertRaises(ValueError):
            parse_chapters_response('[{"start_seconds": 0, "title": ""}]')

    def test_markdown_contains_clickable_style_timestamps(self):
        markdown = chapters_to_markdown([VideoChapter(65, "主题", "描述")])
        self.assertIn("[00:01:05] 主题", markdown)
        self.assertIn("描述", markdown)

    def test_adds_chapter_information_to_first_subtitle_after_start(self):
        segments = [
            SubtitleSegment(0, 2, "开场字幕"),
            SubtitleSegment(61, 64, "正文字幕"),
        ]
        chapters = [
            VideoChapter(0, "开场", "介绍"),
            VideoChapter(60, "正文", "主题说明"),
        ]
        updated = add_chapters_to_subtitles(segments, chapters)
        self.assertIn("【章节：开场】", updated[0].text)
        self.assertIn("【章节：正文】", updated[1].text)
        self.assertIn("正文字幕", updated[1].text)


if __name__ == "__main__":
    unittest.main()
