import json
import tempfile
import unittest
from pathlib import Path

from whisper_captioner.asr_history import ASRHistoryStore


class ASRHistoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.output = self.root / "output"
        self.cache = self.output / "cache" / "local-audio"
        self.store = ASRHistoryStore(
            self.output / "cache" / "asr-history.json",
            output_dir=self.output,
            local_audio_cache_dir=self.cache,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_upsert_deduplicates_canonical_url_and_writes_atomically(self):
        first = self.store.upsert(
            "https://www.youtube.com/watch?v=abc&feature=share",
            title="First",
            status="running",
        )
        second = self.store.upsert(
            "https://www.youtube.com/watch?v=abc",
            title="Second",
            status="ready",
        )
        self.assertEqual(first.id, second.id)
        entries = self.store.list_entries(refresh_paths=False)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].title, "Second")
        self.assertFalse(any(self.store.path.parent.glob("*.tmp")))

    def test_corrupt_json_is_backed_up(self):
        self.store.path.parent.mkdir(parents=True)
        self.store.path.write_text("{broken", encoding="utf-8")
        self.assertEqual(self.store.list_entries(), [])
        self.assertTrue(list(self.store.path.parent.glob("asr-history.json.corrupt-*")))
        self.assertEqual(json.loads(self.store.path.read_text()), [])

    def test_old_paths_are_rebased_and_metadata_relocates_wav(self):
        cache_dir = self.cache / "new-key"
        cache_dir.mkdir(parents=True)
        wav = cache_dir / "audio-16k-mono.wav"
        wav.write_bytes(b"wav")
        source = "https://www.youtube.com/watch?v=abc"
        (cache_dir / "metadata.json").write_text(
            json.dumps({"source": source, "identity": source, "wav": str(wav)}),
            encoding="utf-8",
        )
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        self.store.path.write_text(
            json.dumps(
                [
                    {
                        "id": f"url:{source}",
                        "source": source,
                        "canonical_source": source,
                        "title": "test",
                        "kind": "url",
                        "audio_cache_wav": "/Users/vickers/Movies/WhisperCaptioner/cache/local-audio/missing/audio-16k-mono.wav",
                        "output_base": "/Users/vickers/Movies/WhisperCaptioner/artifacts/generated/test/out",
                        "status": "ready",
                    }
                ]
            ),
            encoding="utf-8",
        )
        entry = self.store.list_entries()[0]
        self.assertEqual(entry.audio_cache_wav, str(wav))
        self.assertTrue(entry.audio_cache_exists)
        self.assertEqual(entry.output_base, str(self.output / "artifacts/generated/test/out"))

    def test_missing_ready_audio_becomes_pruned(self):
        entry = self.store.upsert("/missing/source.mp4", status="ready")
        self.assertEqual(entry.status, "audio_cache_pruned")


if __name__ == "__main__":
    unittest.main()
