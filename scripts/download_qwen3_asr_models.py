from __future__ import annotations

import time

from huggingface_hub import snapshot_download
from whisper_captioner.config import apply_resource_environment


REPOS = [
    "mlx-community/Qwen3-ASR-0.6B-4bit",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
    "mlx-community/Qwen3-ASR-1.7B-8bit",
]


def main() -> None:
    apply_resource_environment()
    for repo in REPOS:
        print(f"=== {repo} ===", flush=True)
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                path = snapshot_download(
                    repo_id=repo,
                    max_workers=1,
                    resume_download=True,
                )
                print(f"SUCCESS {repo} {path}", flush=True)
                break
            except Exception as exc:  # pragma: no cover - runtime/network path
                last_exc = exc
                print(
                    f"RETRY {repo} attempt={attempt} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(min(10, attempt * 3))
        else:
            assert last_exc is not None
            raise last_exc


if __name__ == "__main__":
    main()
