FROM fedirz/faster-whisper-server:latest-cuda

RUN cd /root/faster-whisper-server \
  && uv pip install --python .venv/bin/python --upgrade --no-deps faster-whisper==1.2.1 ctranslate2==4.8.1

RUN python3 - <<'PY'
from pathlib import Path

path = Path("/root/faster-whisper-server/faster_whisper_server/routers/stt.py")
text = path.read_text()
text = text.replace(
    "from faster_whisper_server.config import (",
    "from faster_whisper import BatchedInferencePipeline\nfrom faster_whisper_server.config import (",
)
text = text.replace(
    "    vad_filter: Annotated[bool, Form()] = False,\n) -> Response | StreamingResponse:",
    "    vad_filter: Annotated[bool, Form()] = False,\n"
    "    batch_size: Annotated[int, Form()] = 0,\n"
    ") -> Response | StreamingResponse:",
)
old = """        segments, transcription_info = whisper.transcribe(
            file.file,
            task=Task.TRANSCRIBE,
            language=language,
            initial_prompt=prompt,
            word_timestamps=\"word\" in timestamp_granularities,
            temperature=temperature,
            vad_filter=vad_filter,
            hotwords=hotwords,
        )
"""
new = """        if batch_size and batch_size > 1:
            batched = BatchedInferencePipeline(model=whisper)
            segments, transcription_info = batched.transcribe(
                file.file,
                batch_size=batch_size,
                language=language,
                initial_prompt=prompt,
                word_timestamps=\"word\" in timestamp_granularities,
                temperature=temperature,
                vad_filter=True,
                hotwords=hotwords,
            )
        else:
            segments, transcription_info = whisper.transcribe(
                file.file,
                task=Task.TRANSCRIBE,
                language=language,
                initial_prompt=prompt,
                word_timestamps=\"word\" in timestamp_granularities,
                temperature=temperature,
                vad_filter=vad_filter,
                hotwords=hotwords,
            )
"""
if old not in text:
    raise SystemExit("Could not patch faster_whisper_server transcribe route")
path.write_text(text.replace(old, new))
PY
