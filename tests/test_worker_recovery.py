import os
import threading
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from whisper_captioner.models import MODES, SubtitleSegment
from whisper_captioner.workers import (
    QueueRunConfig,
    QueueWorker,
    RollingPrefetchWorker,
    _nuc_asr_model_for_mode,
    controlled_cache_dir,
    parse_silencedetect_voice_window,
    prepare_url_audio_cache,
    prune_local_audio_cache,
)
from whisper_captioner.cache import controlled_cache_dir_name


class WorkerRecoveryTest(unittest.TestCase):
    def test_controlled_cache_name_is_human_readable_and_stable(self):
        name = controlled_cache_dir_name(
            "https://youtu.be/J5r17YdAmqY?t=12",
            "mlx_audio",
            "mlx-community/Qwen3-ASR-0.6B-4bit",
            30,
        )
        self.assertTrue(name.startswith("youtube-J5r17YdAmqY__Qwen3-ASR-0.6B-4bit__"))
        self.assertEqual(len(name.rsplit("__", 1)[-1]), 24)

    def test_controlled_cache_dir_migrates_legacy_hash_directory(self):
        url = "https://www.youtube.com/watch?v=J5r17YdAmqY"
        backend = "mlx_audio"
        model = "mlx-community/Qwen3-ASR-0.6B-4bit"
        from whisper_captioner.cache import cache_slug, canonical_media_url

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / cache_slug(canonical_media_url(url), backend, model, 30)
            legacy.mkdir()
            (legacy / "chunk-0000-raw.json").write_text("[]", encoding="utf-8")
            with patch("whisper_captioner.workers.CACHE_DIR", root):
                migrated = controlled_cache_dir(url, backend, model, 30)

            self.assertFalse(legacy.exists())
            self.assertTrue((migrated / "chunk-0000-raw.json").exists())
            self.assertIn("youtube-J5r17YdAmqY", migrated.name)

    def test_rolling_worker_emits_first_chunk_then_appends(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = RollingPrefetchWorker("https://example.com/video", mode)
        first = []
        more = []
        worker.first_segments.connect(first.append)
        worker.more_segments.connect(more.append)

        worker._emit_incremental_segments([SubtitleSegment(0, 1, "first")])
        worker._emit_incremental_segments([SubtitleSegment(30, 31, "second")])

        self.assertEqual([[segment.text for segment in batch] for batch in first], [["first"]])
        self.assertEqual([[segment.text for segment in batch] for batch in more], [["second"]])

    def test_native_subtitles_prefer_manual_then_fall_back_to_automatic(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = RollingPrefetchWorker("https://example.com/video", mode)
        commands = []

        def fake_run(command, _label):
            commands.append(command)
            if "--write-auto-subs" in command:
                output_template = Path(command[command.index("-o") + 1])
                subtitle_path = Path(str(output_template).replace("%(ext)s", "zh-Hans.vtt"))
                subtitle_path.write_text(
                    "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n测试字幕\n",
                    encoding="utf-8",
                )

        worker._run_cmd = fake_run
        with tempfile.TemporaryDirectory() as directory:
            segments, kind = worker._load_or_fetch_native_subtitles(Path(directory))

        self.assertEqual(kind, "zh")
        self.assertEqual([segment.text for segment in segments], ["测试字幕"])
        self.assertIn("--write-subs", commands[0])
        self.assertIn("--write-auto-subs", commands[1])

    def test_controlled_native_subtitles_are_exported_and_emitted(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = RollingPrefetchWorker("https://example.com/video", mode)
        segments = [SubtitleSegment(1.0, 2.0, "原生字幕")]
        emitted = []
        worker.native_subtitles_detected.connect(
            lambda found, message: emitted.append((found, message))
        )

        with tempfile.TemporaryDirectory() as directory:
            output_base = Path(directory) / "native-output"
            with (
                patch.object(worker, "_load_or_fetch_native_subtitles", return_value=(segments, "zh")),
                patch.object(worker, "_native_output_base", return_value=output_base),
                patch("whisper_captioner.workers.CACHE_DIR", Path(directory) / "cache"),
            ):
                worker._do_rolling_prefetch()

            self.assertTrue(output_base.with_suffix(".srt").exists())
            self.assertTrue(output_base.with_suffix(".txt").exists())

        self.assertEqual(emitted[0][0], segments)
        self.assertIn("已下载、保存并载入", emitted[0][1])

    def test_raw_and_polished_outputs_use_distinct_model_suffixes(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        from whisper_captioner.models import LLM_PROVIDERS

        provider = next(
            item for item in LLM_PROVIDERS if item.key == "nuc_ollama_gemma4"
        )
        worker = RollingPrefetchWorker(
            "https://example.com/video",
            mode,
            llm_provider=provider,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("whisper_captioner.workers.GENERATED_DIR", Path(directory)),
            patch.object(worker, "_source_title", return_value="测试视频"),
        ):
            raw_name = worker._raw_output_base("unused").name
            polished_name = worker._optimized_output_base("unused").name

        self.assertEqual(
            raw_name,
            "测试视频-qwen3_asr_06b_4bit_mlx-原始识别字幕",
        )
        self.assertEqual(
            polished_name,
            "测试视频-qwen3_asr_06b_4bit_mlx-nuc_ollama_gemma4-gemma4_latest-LLM优化字幕",
        )

    def test_failed_polish_does_not_export_fake_optimized_subtitles(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        from whisper_captioner.models import LLM_PROVIDERS
        from whisper_captioner.subtitle_io import save_segments

        provider = next(
            item for item in LLM_PROVIDERS if item.key == "nuc_ollama_gemma4"
        )
        worker = RollingPrefetchWorker(
            "https://example.com/video",
            mode,
            llm_provider=provider,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_segments(
                root / "chunk-0000-raw.json",
                [SubtitleSegment(0, 1, "原始文本")],
            )
            output_base = root / "LLM优化字幕"
            with patch(
                "whisper_captioner.workers.llm_proofread",
                side_effect=TimeoutError("offline"),
            ):
                result = worker._run_full_document_polish(root, output_base)

            self.assertEqual(result[0].text, "原始文本")
            self.assertFalse(output_base.with_suffix(".srt").exists())
            self.assertFalse(output_base.with_suffix(".txt").exists())
            self.assertFalse(worker._final_subtitle_cache_path(root).exists())

    def test_controlled_qwen_parallel_buffers_out_of_order_results(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        config = QueueRunConfig(
            qwen_replicas=2,
            qwen_chunk_seconds=45,
            qwen_parallel_enabled=True,
        )
        worker = RollingPrefetchWorker(
            "https://example.com/video",
            mode,
            run_config=config,
        )
        first = []
        more = []
        progress = []
        worker.first_segments.connect(first.append)
        worker.more_segments.connect(more.append)
        worker.progress.connect(lambda done, total: progress.append((done, total)))

        def fake_parallel(_queue_worker, _audio, on_chunk_ready=None, **_kwargs):
            tasks = [
                ({"label": "1", "start": 45.0, "duration": 45.0}, "second"),
                ({"label": "0", "start": 0.0, "duration": 45.0}, "first"),
                ({"label": "2", "start": 90.0, "duration": 1.0}, "third"),
            ]
            for done, (task, text) in enumerate(tasks, start=1):
                on_chunk_ready(
                    task,
                    [SubtitleSegment(task["start"], task["start"] + 1, text)],
                )
                _queue_worker.chunk_progress.emit(
                    {
                        "done": done,
                        "total": 3,
                        "finished": done == 3,
                        "inflight": 3 - done,
                        "splits": 0,
                    }
                )
            return []

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                QueueWorker,
                "_transcribe_local_qwen3_asr_chunked",
                fake_parallel,
            ):
                rebuilt = worker._transcribe_local_qwen_parallel(
                    Path(directory) / "audio.wav",
                    Path(directory),
                    91.0,
                )
            chunk_index = __import__("json").loads(
                (Path(directory) / "chunk-index.json").read_text(encoding="utf-8")
            )

        self.assertTrue(rebuilt)
        self.assertEqual([[segment.text for segment in batch] for batch in first], [["first"]])
        self.assertEqual(
            [[segment.text for segment in batch] for batch in more],
            [["second"], ["third"]],
        )
        self.assertEqual(progress[-1], (3, 3))
        self.assertEqual(chunk_index["chunks"][1]["start_seconds"], 45.0)
        self.assertEqual(chunk_index["chunks"][1]["cache_file"], "chunk-0001-raw.json")

    def test_controlled_qwen_parallel_forwards_dynamic_split_total(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = RollingPrefetchWorker(
            "https://example.com/video",
            mode,
            run_config=QueueRunConfig(qwen_replicas=2, qwen_chunk_seconds=45),
        )
        progress = []
        worker.progress.connect(lambda done, total: progress.append((done, total)))

        def fake_parallel(_queue_worker, _audio, on_chunk_ready=None, **_kwargs):
            child_tasks = [
                {"label": "0a", "start": 0.0, "duration": 22.5},
                {"label": "0b", "start": 22.5, "duration": 22.5},
            ]
            for done, task in enumerate(child_tasks, start=1):
                on_chunk_ready(
                    task,
                    [SubtitleSegment(task["start"], task["start"] + 1, task["label"])],
                )
                _queue_worker.chunk_progress.emit(
                    {
                        "done": done,
                        "total": 2,
                        "finished": done == 2,
                        "inflight": 2 - done,
                        "splits": 1,
                    }
                )
            return []

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                QueueWorker,
                "_transcribe_local_qwen3_asr_chunked",
                fake_parallel,
            ):
                worker._transcribe_local_qwen_parallel(
                    Path(directory) / "audio.wav",
                    Path(directory),
                    45.0,
                )
            raw_names = sorted(path.name for path in Path(directory).glob("chunk-*-raw.json"))

        self.assertEqual(progress[-1], (2, 2))
        self.assertEqual(raw_names, ["chunk-0a-raw.json", "chunk-0b-raw.json"])

    def test_controlled_qwen_parallel_only_runs_missing_chunks(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = RollingPrefetchWorker(
            "https://example.com/video",
            mode,
            run_config=QueueRunConfig(qwen_replicas=2, qwen_chunk_seconds=45),
        )
        emitted = []
        worker.first_segments.connect(
            lambda segments: emitted.extend(segment.text for segment in segments)
        )
        worker.more_segments.connect(
            lambda segments: emitted.extend(segment.text for segment in segments)
        )
        received_tasks = []

        def fake_parallel(_queue_worker, _audio, on_chunk_ready=None, tasks_override=None):
            received_tasks.extend(task["label"] for task in tasks_override or [])
            for task in tasks_override or []:
                on_chunk_ready(
                    task,
                    [SubtitleSegment(task["start"], task["start"] + 1, f"new-{task['label']}")],
                )
            return []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (0, 2):
                start = index * 45.0
                from whisper_captioner.subtitle_io import save_segments

                save_segments(
                    root / f"chunk-{index:04d}-raw.json",
                    [SubtitleSegment(start, start + 1, f"cached-{index}")],
                )
            with patch.object(
                QueueWorker,
                "_transcribe_local_qwen3_asr_chunked",
                fake_parallel,
            ):
                rebuilt = worker._transcribe_local_qwen_parallel(
                    root / "audio.wav",
                    root,
                    135.0,
                )

        self.assertTrue(rebuilt)
        self.assertEqual(received_tasks, ["1"])
        self.assertEqual(emitted, ["cached-0", "new-1", "cached-2"])

    def test_controlled_qwen_parallel_resumes_split_child_caches(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = RollingPrefetchWorker(
            "https://example.com/video",
            mode,
            run_config=QueueRunConfig(qwen_replicas=2, qwen_chunk_seconds=45),
        )
        emitted = []
        worker.first_segments.connect(
            lambda segments: emitted.extend(segment.text for segment in segments)
        )
        worker.more_segments.connect(
            lambda segments: emitted.extend(segment.text for segment in segments)
        )
        received_tasks = []

        def fake_parallel(_queue_worker, _audio, on_chunk_ready=None, tasks_override=None):
            received_tasks.extend(task["label"] for task in tasks_override or [])
            for task in tasks_override or []:
                on_chunk_ready(
                    task,
                    [SubtitleSegment(task["start"], task["start"] + 1, f"new-{task['label']}")],
                )
            return []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from whisper_captioner.subtitle_io import save_segments

            save_segments(
                root / "chunk-0a-raw.json",
                [SubtitleSegment(0, 1, "cached-0a")],
            )
            with patch.object(
                QueueWorker,
                "_transcribe_local_qwen3_asr_chunked",
                fake_parallel,
            ):
                worker._transcribe_local_qwen_parallel(
                    root / "audio.wav",
                    root,
                    45.0,
                )

        self.assertEqual(received_tasks, ["0b"])
        self.assertEqual(emitted, ["cached-0a", "new-0b"])

    def test_prepare_url_audio_cache_reuses_existing_wav(self):
        class Status:
            def __init__(self):
                self.messages = []

            def emit(self, message):
                self.messages.append(message)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "url-cache"
            cache_dir.mkdir()
            wav = cache_dir / "audio-16k-mono.wav"
            wav.write_bytes(b"cached")
            status = Status()
            with (
                patch("whisper_captioner.workers.url_audio_cache_dir", return_value=cache_dir),
                patch("whisper_captioner.workers.prune_local_audio_cache"),
            ):
                result = prepare_url_audio_cache(
                    "https://example.com/video",
                    run_command=lambda *_args: self.fail("download should not run"),
                    status_signal=status,
                )

        self.assertEqual(result, wav)
        self.assertIn("Reusing URL audio cache", status.messages[-1])

    def test_prune_local_audio_cache_removes_oldest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oldest = root / "oldest"
            newest = root / "newest"
            keep = root / "keep"
            for path, size, timestamp in (
                (oldest, 6, 10),
                (newest, 6, 20),
                (keep, 6, 30),
            ):
                path.mkdir()
                (path / "audio.wav").write_bytes(b"x" * size)
                os.utime(path, (timestamp, timestamp))
            with patch("whisper_captioner.workers.LOCAL_AUDIO_CACHE_DIR", root):
                removed = prune_local_audio_cache(max_bytes=12, keep=keep)
            self.assertEqual(removed, [oldest])
            self.assertFalse(oldest.exists())
            self.assertTrue(newest.exists())
            self.assertTrue(keep.exists())

    def test_nuc_asr_turbo_and_quality_modes_select_distinct_models(self):
        turbo = next(mode for mode in MODES if mode.key == "nuc_asr_turbo")
        quality = next(mode for mode in MODES if mode.key == "nuc_asr")
        self.assertEqual(
            _nuc_asr_model_for_mode(turbo),
            "deepdml/faster-whisper-large-v3-turbo-ct2",
        )
        self.assertEqual(_nuc_asr_model_for_mode(quality), "large-v3")

    def test_vad_trims_silent_edges_with_guards(self):
        output = """
        [silencedetect] silence_start: 0
        [silencedetect] silence_end: 1.2 | silence_duration: 1.2
        [silencedetect] silence_start: 8.0
        """
        window = parse_silencedetect_voice_window(output, 10.0)
        self.assertIsNotNone(window)
        self.assertAlmostEqual(window.start, 1.1)
        self.assertAlmostEqual(window.duration, 7.05)

    def test_vad_all_silence_is_legal_empty_window(self):
        output = "[silencedetect] silence_start: 0"
        self.assertIsNone(parse_silencedetect_voice_window(output, 30.0))

    def test_vad_without_silence_means_full_voice_window(self):
        window = parse_silencedetect_voice_window("", 30.0)
        self.assertIsNotNone(window)
        self.assertEqual(window.start, 0.0)
        self.assertEqual(window.duration, 30.0)

    def test_environment_config_is_clamped(self):
        with patch.dict(
            os.environ,
            {
                "WHISPER_CAPTIONER_QWEN_REPLICAS": "9",
                "WHISPER_CAPTIONER_QWEN_CHUNK_SECONDS": "5",
                "WHISPER_CAPTIONER_QWEN_PARALLEL": "1",
                "WHISPER_CAPTIONER_ADAPTIVE_SPLIT": "true",
                "WHISPER_CAPTIONER_CPP_THREADS": "12",
                "WHISPER_CAPTIONER_CPP_FLASH_ATTN": "false",
            },
            clear=False,
        ):
            config = QueueRunConfig.from_environment()
        self.assertEqual(config.qwen_replicas, 4)
        self.assertEqual(config.qwen_chunk_seconds, 10.0)
        self.assertTrue(config.qwen_parallel_enabled)
        self.assertTrue(config.adaptive_split_enabled)
        self.assertEqual(config.cpp_threads, 8)
        self.assertFalse(config.cpp_flash_attn)

    def test_recovery_features_are_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config = QueueRunConfig.from_environment()
        self.assertTrue(config.adaptive_split_enabled)
        self.assertTrue(config.remote_vad_enabled)

    def test_cpp_runtime_args_use_selected_threads_and_flash_attention(self):
        enabled = QueueRunConfig(cpp_threads=6, cpp_flash_attn=True)
        disabled = QueueRunConfig(cpp_threads=4, cpp_flash_attn=False)
        self.assertEqual(enabled.cpp_args(), ["-t", "6", "--flash-attn"])
        self.assertEqual(disabled.cpp_args(), ["-t", "4", "--no-flash-attn"])

    def test_queue_cpp_command_uses_runtime_config(self):
        mode = next(mode for mode in MODES if mode.key == "hq_turbo")
        commands = []

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            wav = Path(directory) / "audio.wav"
            source.touch()
            wav.touch()
            worker = QueueWorker(
                [],
                mode,
                QueueRunConfig(
                    prepared_wavs={str(source): str(wav)},
                    cpp_threads=6,
                    cpp_flash_attn=True,
                ),
            )
            worker._run = lambda command, _label: commands.append(command)
            worker.history.upsert = lambda *_args, **_kwargs: None
            with patch("whisper_captioner.workers.source_output_dir", return_value=Path(directory)):
                self.assertTrue(worker._process(str(source)))

        command = commands[-1]
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], "6")
        self.assertIn("--flash-attn", command)

    def test_parallel_qwen_results_are_sorted_and_progress_is_structured(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = QueueWorker(
            [],
            mode,
            QueueRunConfig(
                qwen_replicas=2,
                qwen_chunk_seconds=45,
                qwen_parallel_enabled=True,
            ),
        )
        worker._get_duration = lambda _path: 91.0

        def fake_task(_wav, task, _cancel_event, _holder):
            time.sleep(0.003 if task["start"] == 0 else 0.001)
            return [SubtitleSegment(task["start"], task["start"] + 1, task["label"])]

        worker._run_qwen_chunk_task = fake_task
        progress = []
        worker.chunk_progress.connect(progress.append)
        segments = worker._transcribe_local_qwen3_asr_chunked("/unused.wav")
        self.assertEqual([segment.start for segment in segments], [0, 45, 90])
        self.assertEqual(progress[-1]["done"], 3)
        self.assertEqual(progress[-1]["total"], 3)
        self.assertTrue(progress[-1]["finished"])

    def test_parallel_qwen_retry_does_not_block_other_completed_chunks(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = QueueWorker(
            [],
            mode,
            QueueRunConfig(
                qwen_replicas=2,
                qwen_chunk_seconds=45,
                qwen_parallel_enabled=True,
            ),
        )
        worker._get_duration = lambda _path: 90.0
        retry_started = threading.Event()
        release_retry = threading.Event()
        callbacks = []
        outcome = {}

        def fake_task(_wav, task, _cancel_event, _holder):
            if task["label"] == "0" and task.get("attempt", 0) == 0:
                raise RuntimeError("first attempt failed")
            if task["label"] == "0":
                retry_started.set()
                release_retry.wait(timeout=5)
            elif task["label"] == "1":
                retry_started.wait(timeout=5)
            return [SubtitleSegment(task["start"], task["start"] + 1, task["label"])]

        worker._run_qwen_chunk_task = fake_task

        def run_transcription():
            try:
                worker._transcribe_local_qwen3_asr_chunked(
                    "/unused.wav",
                    on_chunk_ready=lambda task, _segments: callbacks.append(task["label"]),
                )
            except Exception as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run_transcription)
        thread.start()
        self.assertTrue(retry_started.wait(timeout=3))
        deadline = time.monotonic() + 3
        while "1" not in callbacks and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIn("1", callbacks)
        release_retry.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)

    def test_native_subtitle_fetch_removes_stale_attempt_files(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = RollingPrefetchWorker("https://example.com/video", mode)

        def fake_run(command, _label):
            if "--write-auto-subs" not in command:
                return
            output_template = Path(command[command.index("-o") + 1])
            subtitle_path = Path(str(output_template).replace("%(ext)s", "zh-Hans.vtt"))
            subtitle_path.write_text(
                "WEBVTT\n\n00:00:03.000 --> 00:00:04.000\n新字幕\n",
                encoding="utf-8",
            )

        worker._run_cmd = fake_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale_dir = root / "native-subs-zh"
            stale_dir.mkdir()
            (stale_dir / "native.old.vtt").write_text(
                "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n旧字幕\n",
                encoding="utf-8",
            )
            segments, _kind = worker._load_or_fetch_native_subtitles(root)

        self.assertEqual([segment.text for segment in segments], ["新字幕"])

    def test_chunk_groups_merge_in_chunk_order_and_deduplicate_boundary(self):
        groups = [
            (45.0, "1", [SubtitleSegment(45.0, 46.0, "重复句"), SubtitleSegment(46.0, 47.0, "后句")]),
            (0.0, "0", [SubtitleSegment(44.0, 45.0, "重复句")]),
        ]
        merged = QueueWorker._merge_qwen_chunk_groups(groups)
        self.assertEqual([segment.text for segment in merged], ["重复句", "后句"])
        self.assertEqual(merged[0].start, 44.0)
        self.assertEqual(merged[0].end, 46.0)

    def test_chunk_cancel_event_terminates_only_its_process(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = QueueWorker([], mode)
        cancel_event = threading.Event()
        holder = {}
        outcome = {}

        def run():
            try:
                worker._run_chunk_command(
                    ["/bin/sh", "-c", "sleep 30"],
                    "test chunk",
                    cancel_event,
                    holder,
                )
            except Exception as exc:
                outcome["error"] = str(exc)

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 5
        while "proc" not in holder and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIn("proc", holder)
        cancel_event.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIn("cancelled", outcome["error"].lower())
        self.assertEqual(len(worker._active_processes), 0)

    @patch("whisper_captioner.workers._request_json_url")
    def test_release_busy_only_logs(self, request):
        request.return_value = {"status": "busy"}
        mode = next(mode for mode in MODES if mode.key == "nuc_asr")
        worker = QueueWorker([], mode)
        messages = []
        worker.status.connect(messages.append)
        worker._release_nuc_asr()
        self.assertIn("busy", messages[-1])

    @patch("whisper_captioner.workers._request_json_url", side_effect=TimeoutError("offline"))
    def test_release_offline_only_logs(self, _request):
        mode = next(mode for mode in MODES if mode.key == "nuc_asr")
        worker = QueueWorker([], mode)
        messages = []
        worker.status.connect(messages.append)
        worker._release_nuc_asr()
        self.assertIn("unavailable", messages[-1])


if __name__ == "__main__":
    unittest.main()
