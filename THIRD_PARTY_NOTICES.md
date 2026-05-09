# Third-Party Notices

This repository includes or depends on third-party software, tools, and models.
Your use and redistribution of those components remain subject to their own
licenses and usage terms.

## Repository License Boundary

- The repository's original code is covered by the license in `LICENSE`.
- Third-party code, binaries, and models are not re-licensed by this project.
- If you distribute this project, you must separately comply with the licenses
  for every bundled or required third-party component.

## Known Third-Party Components

### SenseVoice.cpp

- Path: `~/Movies/whisper-captioner_APP_Resource/third_party/SenseVoice.cpp`
- Repository tracking status: intentionally not committed in this repository
- Upstream source: `https://github.com/lovemefan/SenseVoice.cpp`
- Upstream license file: `~/Movies/whisper-captioner_APP_Resource/third_party/SenseVoice.cpp/LICENSE`
- License observed locally: MIT

### whisper.cpp tools

- Used by this project through local installations such as `whisper-cli` and
  `whisper-stream`
- License: check upstream `whisper.cpp` repository before redistribution

### FFmpeg / FFprobe

- Used as external system tools
- License: check the exact build and upstream FFmpeg licensing terms before
  redistribution

### yt-dlp

- Used as an external downloader
- License: check upstream `yt-dlp` licensing terms before redistribution

### Model Files

Examples used by this project include:

- Whisper GGML model files
- SenseVoice GGUF model files
- MLX model identifiers such as Qwen3-ASR and Whisper variants

Model weights and model identifiers may have separate use restrictions,
redistribution limits, or acceptable-use terms. You must verify those terms
independently before bundling or distributing them.

## Practical Guidance

- Do not assume the repository license covers bundled model files.
- Do not assume system-installed tools can be redistributed under the same
  terms as this repository.
- If you publish releases, provide a separate dependency manifest and include
  each upstream license where required.
