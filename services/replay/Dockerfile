FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

ENV REPLAY_SCENARIO=one_person_walking \
    REPLAY_RATE=100 \
    REPLAY_TARGET=ingest:5566

CMD ["python", "-m", "replay"]
