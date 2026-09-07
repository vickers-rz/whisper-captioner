#!/usr/bin/env python3
"""Recommend a compact mono Opus bitrate for speech-oriented ASR archives.

The advisor is intentionally conservative:

1. Probe the source with ffprobe.
2. For stereo sources, estimate L/R redundancy from evenly spaced samples.
3. Generate short mono Opus samples at several candidate bitrates.
4. Decode every sample to 16 kHz mono PCM WAV and transcribe it with the
   project's whisper.cpp model.
5. Compare candidate transcripts against a per-segment consensus rather than
   trusting a single source transcription as ground truth.
6. Find the lowest stable bitrate and recommend one stable tier above it by
   default as an engineering safety margin.

This measures ASR *stability under transcoding*. It is not an absolute WER/CER
benchmark because no human reference transcript is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_MODEL = Path(
    "/Volumes/T7_APFS/MacBackup/Movies/"
    "whisper-captioner_APP_Resource/whisper-models/ggml-large-v3-q5_0.bin"
)
DEFAULT_BITRATES = (48, 32, 24, 20, 16, 12)


@dataclass
class ProbeInfo:
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str
    duration_seconds: float
    size_bytes: int
    average_bitrate_bps: int


@dataclass
class StereoRedundancy:
    evaluated: bool
    sample_suppressions_db: list[float]
    median_suppression_db: float | None
    mono_safe: bool | None
    threshold_db: float


@dataclass
class BitrateResult:
    bitrate_kbps: int
    actual_sample_bitrate_kbps: float
    estimated_full_size_mib: float
    consensus_edits: int
    consensus_chars: int
    consensus_cer_percent: float
    max_segment_cer_percent: float
    exact_consensus_segments: int
    total_segments: int
    stable: bool


@dataclass
class Advice:
    source: str
    probe: ProbeInfo
    stereo_redundancy: StereoRedundancy
    sample_starts_seconds: list[float]
    sample_seconds: float
    whisper_model: str
    bitrates: list[BitrateResult]
    stable_floor_kbps: int | None
    recommended_kbps: int | None
    recommendation_reason: str


def run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def require_binary(name_or_path: str) -> str:
    if "/" in name_or_path:
        path = Path(name_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Required executable not found: {path}")
        return str(path)
    resolved = shutil.which(name_or_path)
    if not resolved:
        raise FileNotFoundError(f"Required executable not found in PATH: {name_or_path}")
    return resolved


def probe_audio(path: Path, ffprobe: str) -> ProbeInfo:
    cp = run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,channel_layout",
            "-show_entries",
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(cp.stdout)
    stream = payload["streams"][0]
    fmt = payload["format"]
    return ProbeInfo(
        codec=str(stream.get("codec_name", "unknown")),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        channel_layout=str(stream.get("channel_layout") or "unknown"),
        duration_seconds=float(fmt.get("duration") or 0),
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        average_bitrate_bps=int(float(fmt.get("bit_rate") or 0)),
    )


def sample_starts(duration: float, count: int, seconds: float) -> list[float]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    usable = max(0.0, duration - seconds)
    if usable <= 0 or count == 1:
        return [0.0]
    # Leave a small guard near the exact endpoints when possible.
    left = min(60.0, usable * 0.05)
    right = max(left, usable - left)
    if count == 1 or math.isclose(left, right):
        return [max(0.0, usable / 2)]
    return [left + (right - left) * i / (count - 1) for i in range(count)]


def _last_rms(stderr: str) -> float:
    values = [float(v) for v in re.findall(r"RMS level dB:\s*(-?inf|-?\d+(?:\.\d+)?)", stderr)]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        raise RuntimeError("ffmpeg astats produced no finite RMS level")
    return finite[-1]


def measure_stereo_redundancy(
    source: Path,
    starts: Iterable[float],
    seconds: float,
    ffmpeg: str,
    threshold_db: float,
) -> StereoRedundancy:
    suppressions: list[float] = []
    for start in starts:
        common = [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{seconds:.3f}",
            "-i",
            str(source),
        ]
        main = run(common + ["-af", "astats=metadata=0:reset=0", "-f", "null", "-"])
        diff = run(
            common
            + [
                "-af",
                "pan=mono|c0=c0-c1,astats=metadata=0:reset=0",
                "-f",
                "null",
                "-",
            ]
        )
        main_rms = _last_rms(main.stderr)
        diff_rms = _last_rms(diff.stderr)
        suppressions.append(main_rms - diff_rms)

    ordered = sorted(suppressions)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else (ordered[mid - 1] + ordered[mid]) / 2
    )
    return StereoRedundancy(
        evaluated=True,
        sample_suppressions_db=suppressions,
        median_suppression_db=median,
        mono_safe=median >= threshold_db,
        threshold_db=threshold_db,
    )


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        ch
        for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def choose_consensus(texts: dict[int, str], anchor_count: int = 3) -> str:
    """Return a high-bitrate-anchored consensus transcript.

    Only the highest ``anchor_count`` bitrate candidates vote. This prevents a
    cluster of aggressively compressed candidates from becoming the reference
    merely because there are more of them. Ties are resolved toward the
    highest bitrate candidate.
    """
    normalized = {br: normalize_transcript(text) for br, text in texts.items()}
    anchors = sorted(normalized, reverse=True)[: max(1, anchor_count)]
    counts = Counter(normalized[br] for br in anchors)
    best_count = max(counts.values())
    winners = {text for text, count in counts.items() if count == best_count}
    for br in anchors:
        if normalized[br] in winners:
            return normalized[br]
    raise AssertionError("unreachable")


def recommend_bitrate(stable_descending: list[int], margin_steps: int) -> tuple[int | None, int | None]:
    if not stable_descending:
        return None, None
    stable_ascending = sorted(stable_descending)
    floor = stable_ascending[0]
    index = min(len(stable_ascending) - 1, margin_steps)
    return floor, stable_ascending[index]


def transcribe(
    wav: Path,
    out_prefix: Path,
    whisper_cli: str,
    model: Path,
    language: str,
    threads: int,
) -> str:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            whisper_cli,
            "-m",
            str(model),
            "-l",
            language,
            "-t",
            str(threads),
            "-nt",
            "-otxt",
            "-of",
            str(out_prefix),
            str(wav),
        ]
    )
    txt = out_prefix.with_suffix(".txt")
    if not txt.exists():
        raise RuntimeError(f"whisper-cli did not create expected output: {txt}")
    return txt.read_text(encoding="utf-8")


def encode_sample(
    source: Path,
    start: float,
    seconds: float,
    bitrate: int,
    ogg: Path,
    wav: Path,
    ffmpeg: str,
) -> None:
    ogg.parent.mkdir(parents=True, exist_ok=True)
    wav.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{seconds:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            f"{bitrate}k",
            "-vbr",
            "on",
            "-compression_level",
            "10",
            "-application",
            "voip",
            str(ogg),
        ]
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(ogg),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ]
    )


def encode_original_wav(
    source: Path, start: float, seconds: float, wav: Path, ffmpeg: str
) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{seconds:.3f}",
            "-i",
            str(source),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ]
    )


def actual_bitrate_kbps(files: list[Path], total_seconds: float) -> float:
    return sum(p.stat().st_size for p in files) * 8 / total_seconds / 1000


def render_markdown(advice: Advice) -> str:
    p = advice.probe
    lines = [
        "# ASR Opus Bitrate Advisor",
        "",
        f"- Source: `{advice.source}`",
        f"- Source audio: `{p.codec}`, {p.sample_rate} Hz, {p.channels} ch, "
        f"{p.average_bitrate_bps / 1000:.1f} kbps average",
        f"- Duration: {p.duration_seconds:.2f} s",
        f"- Sample plan: {len(advice.sample_starts_seconds)} × {advice.sample_seconds:.1f} s",
        f"- Whisper model: `{advice.whisper_model}`",
    ]
    sr = advice.stereo_redundancy
    if sr.evaluated:
        lines.append(
            f"- Stereo L-R suppression median: {sr.median_suppression_db:.2f} dB "
            f"(mono-safe threshold {sr.threshold_db:.1f} dB; result: "
            f"{'safe' if sr.mono_safe else 'not proven safe'})"
        )
    else:
        lines.append("- Stereo redundancy: not applicable (source is not stereo)")
    lines += [
        "",
        "## Candidate stability",
        "",
        "| Nominal | Actual sampled | Consensus CER | Max segment CER | Exact segments | Stable | Est. full size |",
        "|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for r in advice.bitrates:
        lines.append(
            f"| {r.bitrate_kbps} kbps | {r.actual_sample_bitrate_kbps:.2f} kbps | "
            f"{r.consensus_cer_percent:.3f}% | {r.max_segment_cer_percent:.3f}% | "
            f"{r.exact_consensus_segments}/{r.total_segments} | "
            f"{'yes' if r.stable else 'no'} | {r.estimated_full_size_mib:.2f} MiB |"
        )
    lines += [
        "",
        "## Recommendation",
        "",
        f"- Stable floor: `{advice.stable_floor_kbps} kbps`" if advice.stable_floor_kbps else "- Stable floor: not established",
        f"- Recommended: **{advice.recommended_kbps} kbps mono Opus**" if advice.recommended_kbps else "- Recommended: no automatic recommendation",
        f"- Rationale: {advice.recommendation_reason}",
        "",
        "> This benchmark measures transcript stability after lossy transcoding. It does not replace a human-reference WER/CER benchmark.",
    ]
    return "\n".join(lines) + "\n"


def parse_bitrates(value: str) -> tuple[int, ...]:
    vals = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}, reverse=True))
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("bitrates must be positive comma-separated integers")
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--whisper-cli", default="/opt/homebrew/bin/whisper-cli")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--sample-seconds", type=float, default=45.0)
    parser.add_argument("--bitrates", type=parse_bitrates, default=DEFAULT_BITRATES)
    parser.add_argument("--aggregate-cer-threshold", type=float, default=0.5)
    parser.add_argument("--segment-cer-threshold", type=float, default=2.0)
    parser.add_argument("--mono-suppression-threshold", type=float, default=30.0)
    parser.add_argument("--margin-steps", type=int, default=1)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.home() / "asr-opus-advisor",
        help="Benchmark workspace; existing per-source files may be overwritten.",
    )
    parser.add_argument(
        "--encode-output",
        type=Path,
        help="After benchmarking, encode the full source at the recommended bitrate.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    if not args.model.is_file():
        parser.error(f"Whisper model does not exist: {args.model}")
    if args.samples <= 0 or args.sample_seconds <= 0:
        parser.error("samples and sample-seconds must be positive")
    if args.margin_steps < 0:
        parser.error("margin-steps must be >= 0")

    ffmpeg = require_binary(args.ffmpeg)
    ffprobe = require_binary(args.ffprobe)
    whisper_cli = require_binary(args.whisper_cli)
    probe = probe_audio(source, ffprobe)
    starts = sample_starts(probe.duration_seconds, args.samples, args.sample_seconds)

    work = args.work_dir.expanduser().resolve() / re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem)[-80:]
    ogg_dir = work / "ogg"
    wav_dir = work / "wav"
    tx_dir = work / "transcripts"
    work.mkdir(parents=True, exist_ok=True)

    if probe.channels == 2:
        redundancy = measure_stereo_redundancy(
            source,
            starts,
            min(15.0, args.sample_seconds),
            ffmpeg,
            args.mono_suppression_threshold,
        )
    else:
        redundancy = StereoRedundancy(
            evaluated=False,
            sample_suppressions_db=[],
            median_suppression_db=None,
            mono_safe=(probe.channels == 1),
            threshold_db=args.mono_suppression_threshold,
        )

    transcript_by_segment: dict[str, dict[int, str]] = {}
    candidate_files: dict[int, list[Path]] = {br: [] for br in args.bitrates}

    for i, start in enumerate(starts, 1):
        tag = f"seg{i:02d}"
        print(f"[{tag}] start={start:.2f}s", flush=True)
        original_wav = wav_dir / f"{tag}-original.wav"
        encode_original_wav(source, start, args.sample_seconds, original_wav, ffmpeg)
        # Keep the source transcript as an audit artifact, but consensus is formed
        # from the lossy candidates to avoid treating one decoder boundary as truth.
        transcribe(
            original_wav,
            tx_dir / "original" / tag,
            whisper_cli,
            args.model,
            args.language,
            args.threads,
        )

        transcript_by_segment[tag] = {}
        for br in args.bitrates:
            print(f"  - {br} kbps", flush=True)
            ogg = ogg_dir / f"{tag}-{br}k.ogg"
            wav = wav_dir / f"{tag}-{br}k.wav"
            encode_sample(source, start, args.sample_seconds, br, ogg, wav, ffmpeg)
            candidate_files[br].append(ogg)
            transcript_by_segment[tag][br] = transcribe(
                wav,
                tx_dir / f"{br}k" / tag,
                whisper_cli,
                args.model,
                args.language,
                args.threads,
            )

    consensus_by_segment = {
        tag: choose_consensus(texts) for tag, texts in transcript_by_segment.items()
    }
    results: list[BitrateResult] = []
    total_sample_seconds = len(starts) * args.sample_seconds
    for br in args.bitrates:
        total_chars = 0
        total_edits = 0
        max_cer = 0.0
        exact = 0
        for tag in transcript_by_segment:
            consensus = consensus_by_segment[tag]
            hyp = normalize_transcript(transcript_by_segment[tag][br])
            edits = levenshtein(consensus, hyp)
            chars = max(1, len(consensus))
            cer = 100 * edits / chars
            total_chars += len(consensus)
            total_edits += edits
            max_cer = max(max_cer, cer)
            exact += hyp == consensus
        aggregate = 100 * total_edits / max(1, total_chars)
        actual = actual_bitrate_kbps(candidate_files[br], total_sample_seconds)
        estimate = actual * 1000 * probe.duration_seconds / 8 / 1024 / 1024
        stable = (
            aggregate <= args.aggregate_cer_threshold
            and max_cer <= args.segment_cer_threshold
        )
        results.append(
            BitrateResult(
                bitrate_kbps=br,
                actual_sample_bitrate_kbps=actual,
                estimated_full_size_mib=estimate,
                consensus_edits=total_edits,
                consensus_chars=total_chars,
                consensus_cer_percent=aggregate,
                max_segment_cer_percent=max_cer,
                exact_consensus_segments=exact,
                total_segments=len(starts),
                stable=stable,
            )
        )

    stable = [r.bitrate_kbps for r in results if r.stable]
    floor, recommended = recommend_bitrate(stable, args.margin_steps)

    if recommended is None:
        reason = "No candidate satisfied the configured stability thresholds."
    elif floor == recommended:
        reason = (
            f"{floor} kbps is the lowest stable candidate and no higher stable safety tier "
            "was available for the requested margin."
        )
    else:
        reason = (
            f"{floor} kbps is the measured stable floor; {recommended} kbps is selected "
            f"with {args.margin_steps} stable tier(s) of safety margin."
        )
    if redundancy.evaluated and redundancy.mono_safe is False:
        reason += " Stereo redundancy was not sufficient to prove that mono downmix is safe."
        recommended = None

    advice = Advice(
        source=str(source),
        probe=probe,
        stereo_redundancy=redundancy,
        sample_starts_seconds=starts,
        sample_seconds=args.sample_seconds,
        whisper_model=str(args.model),
        bitrates=results,
        stable_floor_kbps=floor,
        recommended_kbps=recommended,
        recommendation_reason=reason,
    )

    json_path = work / "report.json"
    md_path = work / "report.md"
    json_path.write_text(json.dumps(asdict(advice), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(advice), encoding="utf-8")
    print(render_markdown(advice))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    if args.encode_output:
        if recommended is None:
            raise RuntimeError("No safe automatic recommendation; refusing full encode")
        output = args.encode_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "libopus",
                "-b:a",
                f"{recommended}k",
                "-vbr",
                "on",
                "-compression_level",
                "10",
                "-application",
                "voip",
                str(output),
            ],
            capture=False,
        )
        print(f"Encoded full output: {output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed ({exc.returncode}): {' '.join(exc.cmd)}", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode)
