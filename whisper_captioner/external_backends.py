from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import ASRResult, SpeechRegion
from .subtitle_io import parse_srt
from .subtitle_io import asr_result_from_dict, asr_result_to_dict


class AlignmentBackend(Protocol):
    name: str
    version: str

    def align(self, audio_path: Path, transcript: str, output_dir: Path) -> ASRResult:
        ...


@dataclass
class CliAlignmentBackend:
    command: str = str(Path.home() / ".local/bin/lai")
    name: str = "lattifai-cli"
    version: str = "1.5"

    @property
    def available(self) -> bool:
        parts = shlex.split(self.command)
        return bool(parts and shutil.which(parts[0]))

    def align(self, audio_path: Path, transcript: str, output_dir: Path) -> ASRResult:
        if not self.available:
            raise RuntimeError(f"alignment CLI unavailable: {self.command}")
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "input.srt"
        result_path = output_dir / "aligned.srt"
        input_path.write_text(
            f"1\n00:00:00,000 --> 99:59:59,000\n{transcript.strip()}\n",
            encoding="utf-8",
        )
        command = [
            *shlex.split(self.command),
            "alignment",
            "align",
            str(audio_path),
            str(input_path),
            str(result_path),
            "--direct",
            "-Y",
            "alignment.device=mps",
        ]
        subprocess.run(command, check=True, timeout=900, capture_output=True, text=True)
        if not result_path.exists():
            raise RuntimeError(f"alignment CLI did not create {result_path}")
        segments = parse_srt(result_path)
        aligned = ASRResult(
            language="",
            words=[],
            segments=segments,
            diagnostics={"alignment_backend": self.name, "alignment_cli": command},
        )
        aligned_text = "".join(segment.text for segment in aligned.segments).replace(" ", "")
        source_text = transcript.replace(" ", "").replace("\n", "")
        if aligned_text != source_text:
            raise RuntimeError("alignment result changed transcript text")
        return aligned


def alignment_cache_key(
    audio_path: Path,
    transcript: str,
    backend: AlignmentBackend,
) -> str:
    digest = hashlib.sha256()
    digest.update(audio_path.read_bytes())
    digest.update(transcript.encode("utf-8"))
    digest.update(f"{backend.name}:{backend.version}".encode("utf-8"))
    return digest.hexdigest()[:24]


def optional_alignment_backend() -> CliAlignmentBackend | None:
    command = os.environ.get(
        "WHISPER_CAPTIONER_ALIGNMENT_COMMAND",
        str(Path.home() / ".local/bin/lai"),
    ).strip()
    return CliAlignmentBackend(command) if command else None


@dataclass(frozen=True)
class OmniVADShadowResult:
    status: str
    regions: list[SpeechRegion]
    warning: str = ""


def run_omnivad_shadow(audio_path: Path, output_dir: Path) -> OmniVADShadowResult:
    command_template = os.environ.get(
        "WHISPER_CAPTIONER_OMNIVAD_COMMAND",
        f"{Path(sys.executable).with_name('omnivad')} "
        "{audio} -m vad -f json --chunk 600 --workers 2 -o {output}",
    ).strip()
    executable = shlex.split(command_template)[0]
    if not shutil.which(executable):
        return OmniVADShadowResult("unavailable", [], f"OmniVAD executable unavailable: {executable}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "omnivad-shadow.json"
    command = [
        part.format(audio=str(audio_path), output=str(result_path), output_dir=str(output_dir))
        for part in shlex.split(command_template)
    ]
    try:
        subprocess.run(command, check=True, timeout=300, capture_output=True, text=True)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        items = payload.get("tiers", {}).get("VAD", payload.get("regions", []))
        regions = [
            SpeechRegion(
                float(item["start"]),
                float(item["end"]),
                float(item["confidence"]) if item.get("confidence") is not None else None,
                "omnivad",
            )
            for item in items
            if float(item["end"]) > float(item["start"])
        ]
        return OmniVADShadowResult("completed", regions)
    except Exception as exc:
        return OmniVADShadowResult("failed", [], str(exc))


def save_alignment_result(path: Path, result: ASRResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asr_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
