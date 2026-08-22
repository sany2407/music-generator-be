FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py main.py ./
COPY music_generator ./music_generator

RUN mkdir -p music_generator/generated_music

ENV PORT=8080
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT}
