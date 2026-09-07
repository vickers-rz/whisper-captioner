#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/Volumes/02_HDD_unTar_NFS/播客')
PROJECT = Path('/Users/vickers/Documents/whisper-captioner')
PYTHON = Path('/Users/vickers/miniforge3/envs/whishperapp_pyside6/bin/python')
ENTRY = PROJECT / 'scripts' / 'asr_entrypoints.py'
JOBS = PROJECT / 'artifacts' / 'podcast_batch' / 'gemini_jobs'
LOG = PROJECT / 'artifacts' / 'podcast_batch' / 'gemini_batch_status.json'
MODEL = 'gemini-2.5-flash'
MEDIA_EXTS = {'.mp3', '.m4a', '.aac', '.wav', '.flac', '.ogg', '.opus', '.webm', '.mp4', '.mkv', '.mov'}


def save_status(payload: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOG.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(LOG)


def extract_transcript(md_path: Path) -> str:
    text = md_path.read_text(encoding='utf-8')
    marker = '\n## Transcript\n'
    if marker not in text:
        raise RuntimeError(f'Transcript marker not found: {md_path}')
    body = text.split(marker, 1)[1].strip()
    if not body:
        raise RuntimeError(f'Empty transcript body: {md_path}')
    return body + '\n'


def main() -> int:
    if not ROOT.is_dir():
        print(f'ROOT_NOT_FOUND: {ROOT}', file=sys.stderr, flush=True)
        return 2
    if not PYTHON.is_file() or not ENTRY.is_file():
        print('PROJECT_RUNTIME_MISSING', file=sys.stderr, flush=True)
        return 2

    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in MEDIA_EXTS)
    JOBS.mkdir(parents=True, exist_ok=True)
    status = {
        'root': str(ROOT),
        'model': MODEL,
        'total': len(files),
        'completed': [],
        'skipped': [],
        'failed': [],
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    print(f'FOUND {len(files)} media files', flush=True)

    for index, source in enumerate(files, 1):
        target = source.with_suffix('.txt')
        if target.is_file() and target.stat().st_size > 0:
            print(f'[{index}/{len(files)}] SKIP existing: {target.name}', flush=True)
            status['skipped'].append(str(target))
            save_status(status)
            continue

        digest = hashlib.sha1(str(source).encode('utf-8')).hexdigest()[:12]
        job_dir = JOBS / f'{index:03d}-{digest}'
        job_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n[{index}/{len(files)}] GEMINI: {source.name}', flush=True)
        cmd = [
            str(PYTHON), '-u', str(ENTRY),
            'gemini-local', str(source),
            '--model', MODEL,
            '--output-dir', str(job_dir),
        ]
        started = time.monotonic()
        completed = subprocess.run(cmd, cwd=str(PROJECT), check=False)
        elapsed = round(time.monotonic() - started, 1)
        md_path = job_dir / 'gemini-local-audio-asr-transcript.md'
        if completed.returncode != 0 or not md_path.is_file() or md_path.stat().st_size == 0:
            item = {'source': str(source), 'returncode': completed.returncode, 'elapsed_seconds': elapsed, 'job_dir': str(job_dir)}
            status['failed'].append(item)
            print(f'FAILED rc={completed.returncode}: {source.name}', flush=True)
            save_status(status)
            continue

        try:
            body = extract_transcript(md_path)
            tmp_target = target.with_name(target.name + '.tmp')
            tmp_target.write_text(body, encoding='utf-8')
            tmp_target.replace(target)
        except Exception as exc:
            item = {'source': str(source), 'returncode': 0, 'elapsed_seconds': elapsed, 'job_dir': str(job_dir), 'error': str(exc)}
            status['failed'].append(item)
            print(f'FAILED extracting transcript: {source.name}: {exc}', flush=True)
            save_status(status)
            continue

        status['completed'].append({'source': str(source), 'target': str(target), 'elapsed_seconds': elapsed})
        print(f'OK -> {target}', flush=True)
        save_status(status)
        shutil.rmtree(job_dir, ignore_errors=True)

    status['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save_status(status)
    print('\nBATCH_DONE', flush=True)
    print(f"completed={len(status['completed'])} skipped={len(status['skipped'])} failed={len(status['failed'])} total={status['total']}", flush=True)
    return 0 if not status['failed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
