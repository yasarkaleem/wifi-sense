FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

ENV INGEST_UDP_HOST=0.0.0.0 \
    INGEST_UDP_PORT=5566 \
    INGEST_PUB_HOST=0.0.0.0 \
    INGEST_PUB_PORT=5567 \
    INGEST_PLOT=none \
    INGEST_PLOT_HOST=0.0.0.0 \
    INGEST_PLOT_PORT=8090

EXPOSE 5566/udp 5567 8090

CMD ["python", "-m", "ingest"]
