FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    python-multipart

WORKDIR /app

CMD ["uvicorn", "asr_busy_proxy:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8000"]
