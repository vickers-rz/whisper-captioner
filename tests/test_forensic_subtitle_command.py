import unittest

from scripts.forensic_subtitle_command import choose_breaks, normalized_with_indices


class ForensicSubtitleCommandTest(unittest.TestCase):
    def test_choose_breaks_preserves_gemini_sentence_lines(self):
        text = "大家好，我是青蛙刀圣，我们又见面了。\n今天我们会开启一个新的IP系列《沙丘》。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.2 for index in range(len(chars))]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=8,
            max_chars=28,
            max_duration=5.5,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertIn("大家好，我是青蛙刀圣，我们又见面了。", cue_texts)
        self.assertIn("今天我们会开启一个新的IP系列《沙丘》。", cue_texts)
        self.assertNotIn("大家好，我是青蛙刀圣，我们又见面了。\n今天我们会", cue_texts)


if __name__ == "__main__":
    unittest.main()
