#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

PROJECT = Path('/Users/vickers/Documents/whisper-captioner')
PYTHON = Path('/Users/vickers/miniforge3/envs/whishperapp_pyside6/bin/python')
ENTRY = PROJECT / 'scripts' / 'asr_entrypoints.py'
STATUS = PROJECT / 'artifacts' / 'podcast_batch' / 'gemini_batch_status.json'
RETRY_STATUS = PROJECT / 'artifacts' / 'podcast_batch' / 'gemini_retry_status.json'
MODEL = 'gemini-2.5-flash'
MAX_ATTEMPTS = 3


def extract_transcript(md_path: Path) -> str:
    text = md_path.read_text(encoding='utf-8')
    marker = '\n## Transcript\n'
    if marker not in text:
        raise RuntimeError(f'Transcript marker not found: {md_path}')
    body = text.split(marker, 1)[1].strip()
    if not body:
        raise RuntimeError(f'Empty transcript body: {md_path}')
    return body + '\n'


def save(payload: dict) -> None:
    RETRY_STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = RETRY_STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(RETRY_STATUS)


def main() -> int:
    original = json.loads(STATUS.read_text(encoding='utf-8'))
    failed = original.get('failed', [])
    state = {
        'model': MODEL,
        'total': len(failed),
        'completed': [],
        'failed': [],
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    save(state)
    print(f'RETRY_ONLY {len(failed)} failed items', flush=True)

    for index, item in enumerate(failed, 1):
        source = Path(item['source'])
        target = source.with_suffix('.txt')
        job_dir = Path(item['job_dir'])

        if target.is_file() and target.stat().st_size > 0:
            print(f'[{index}/{len(failed)}] ALREADY_OK: {target.name}', flush=True)
            state['completed'].append({'source': str(source), 'target': str(target), 'attempts': 0})
            save(state)
            continue

        print(f'\n[{index}/{len(failed)}] RETRY: {source.name}', flush=True)
        success = False
        last_error = ''
        last_rc = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f'  attempt {attempt}/{MAX_ATTEMPTS}', flush=True)
            cmd = [
                str(PYTHON), '-u', str(ENTRY),
                'gemini-local', str(source),
                '--model', MODEL,
                '--output-dir', str(job_dir),
            ]
            started = time.monotonic()
            proc = subprocess.run(cmd, cwd=str(PROJECT), check=False)
            elapsed = round(time.monotonic() - started, 1)
            last_rc = proc.returncode
            md_path = job_dir / 'gemini-local-audio-asr-transcript.md'

            if proc.returncode == 0 and md_path.is_file() and md_path.stat().st_size > 0:
                try:
                    body = extract_transcript(md_path)
                    tmp = target.with_name(target.name + '.tmp')
                    tmp.write_text(body, encoding='utf-8')
                    tmp.replace(target)
                    state['completed'].append({
                        'source': str(source),
                        'target': str(target),
                        'attempts': attempt,
                        'elapsed_seconds': elapsed,
                    })
                    save(state)
                    print(f'  OK -> {target}', flush=True)
                    success = True
                    break
                except Exception as exc:
                    last_error = str(exc)
            else:
                last_error = f'rc={proc.returncode}; transcript_missing={not md_path.is_file()}'

            if attempt < MAX_ATTEMPTS:
                delay = 5 * attempt
                print(f'  transient failure; retrying after {delay}s', flush=True)
                time.sleep(delay)

        if not success:
            state['failed'].append({
                'source': str(source),
                'job_dir': str(job_dir),
                'returncode': last_rc,
                'error': last_error,
            })
            save(state)
            print(f'  STILL_FAILED: {source.name}', flush=True)

    state['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save(state)
    print('\nRETRY_DONE', flush=True)
    print(f"completed={len(state['completed'])} failed={len(state['failed'])} total={state['total']}", flush=True)
    return 0 if not state['failed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
