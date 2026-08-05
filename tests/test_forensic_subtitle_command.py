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

    def test_choose_breaks_prefers_punctuation_over_tail_fragment(self):
        text = "大家最近刷国际新闻里面刷到欧美的移民话题，其中比较捕捉人眼球的是穆斯林群体。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.13 for index in range(len(chars))]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=10,
            max_chars=36,
            max_duration=6.5,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertNotIn("大家最近刷国际新闻里面刷到欧美的移民话题，其中", cue_texts)
        self.assertFalse(any(text.endswith("其中") for text in cue_texts))
        self.assertFalse(any(text.startswith("比较") for text in cue_texts))

    def test_choose_breaks_avoids_known_cjk_mid_word_split(self):
        text = "故事的舞台是在未来的一个沙漠星球，当地的土著人弗雷曼人长期受到外来殖民者的压迫。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.16 for index in range(len(chars))]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=10,
            max_chars=36,
            max_duration=4.0,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertFalse(any(text.endswith("土") for text in cue_texts))
        self.assertFalse(any(text.startswith("著人") for text in cue_texts))

    def test_choose_breaks_accepts_valid_llm_boundaries(self):
        text = "你的父亲将会失去沙丘，那是必然的，我们阻止不了事情的发生。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.18 for index in range(len(chars))]

        def provider(_text, _indices, _times, start, _end, _max_chars, _max_duration):
            return [start + 15]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=8,
            max_chars=18,
            max_duration=6.5,
            llm_break_provider=provider,
        )

        self.assertEqual(cues[0]["template_character_end"], 15)

    def test_choose_breaks_rejects_tiny_llm_boundaries(self):
        text = "《沙丘》的故事有自己的纪年法，但如果按照1960年代的创作时间来看。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.1 for index in range(len(chars))]

        def provider(_text, _indices, _times, start, _end, _max_chars, _max_duration):
            return [start + 2]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=8,
            max_chars=28,
            max_duration=6.5,
            llm_break_provider=provider,
        )

        self.assertNotEqual(cues[0]["text"], "《沙丘》")

    def test_choose_breaks_avoids_ascii_word_split(self):
        text = "这导致人类社会已经高度依赖于AI工具和OpenAI模型。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.2 for index in range(len(chars))]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=8,
            max_chars=22,
            max_duration=2.8,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertFalse(any(text.endswith("Open") for text in cue_texts))
        self.assertFalse(any(text.startswith("AI") and not text.startswith("AI工具") for text in cue_texts))

    def test_choose_breaks_rejects_llm_tail_after_prior_comma(self):
        text = "在某些城镇，随着该群体移民数量的增加，有的人就视为提出要推行穆斯林律法，增加议会席位。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.14 for index in range(len(chars))]

        def provider(_text, _indices, _times, start, _end, _max_chars, _max_duration):
            return [start + 28]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=10,
            max_chars=36,
            max_duration=4.0,
            llm_break_provider=provider,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertFalse(any(text.endswith("提出") for text in cue_texts))
        self.assertIn("在某些城镇，随着该群体移民数量的增加，", cue_texts)

    def test_choose_breaks_avoids_short_tail_after_late_comma(self):
        text = "在某些城镇，随着该群体移民数量的增加，有的人就视为提出要推行穆斯林律法，增加议会席位。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.14 for index in range(len(chars))]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=10,
            max_chars=36,
            max_duration=6.5,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertNotIn("增加议会席位。", cue_texts)
        self.assertIn("在某些城镇，随着该群体移民数量的增加，", cue_texts)

    def test_choose_breaks_splits_fast_comma_dense_cue_by_target_duration(self):
        text = "而当圣母登门亲访时，杰西卡更是瑟瑟发抖，因为这一天还是来了。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.19 for index in range(len(chars))]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=8,
            max_chars=36,
            max_duration=6.5,
            target_duration=4.2,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertIn("而当圣母登门亲访时，", cue_texts)
        self.assertIn("杰西卡更是瑟瑟发抖，", cue_texts)
        self.assertIn("因为这一天还是来了。", cue_texts)

    def test_choose_breaks_does_not_prefer_sentence_end_for_fast_internal_commas(self):
        text = "随后杰西卡领着儿子来到一个大房间，声音颤抖地叮嘱保罗务必听从圣母的安排。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.18 for index in range(len(chars))]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=8,
            max_chars=30,
            max_duration=6.5,
            target_duration=4.2,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertIn("随后杰西卡领着儿子来到一个大房间，", cue_texts)

    def test_choose_breaks_rejects_llm_boundary_before_de(self):
        text = "否则我会把毒针刺入你的脖子，让你立刻死亡。"
        chars, indices = normalized_with_indices(text)
        times = [index * 0.18 for index in range(len(chars))]

        def provider(_text, _indices, _times, start, _end, _max_chars, _max_duration):
            return [start + 11]

        cues = choose_breaks(
            text,
            indices,
            times,
            min_chars=8,
            max_chars=30,
            max_duration=6.5,
            target_duration=4.2,
            llm_break_provider=provider,
        )

        cue_texts = [cue["text"] for cue in cues]
        self.assertFalse(any(text.endswith("你") for text in cue_texts))
        self.assertFalse(any(text.startswith("的脖子") for text in cue_texts))


if __name__ == "__main__":
    unittest.main()
