FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# The "ml" (PyTorch) and "localize" (scikit-learn) extras are baked in so
# ML people counting and zone localization are available out of the box;
# each stays inactive unless PIPELINE_COUNTER_CHECKPOINT /
# PIPELINE_LOCALIZER_CHECKPOINT (or --counter-checkpoint /
# --localizer-checkpoint) points at a checkpoint — see cli.py and
# ../../../CLAUDE.md's "ML-based people counting" / "Zone-level
# localization" sections.
#
# Install the CPU-only torch build explicitly first: this container never
# has GPU access (there's no CUDA passthrough into a Docker container, on
# any host), but PyPI's default linux/aarch64 torch wheel bundles the full
# NVIDIA CUDA stack (cudnn/cublas/nccl/triton/...) anyway — multiple
# gigabytes of downloads for libraries that would sit unused. Once this is
# satisfied, the extras install below sees torch's version requirement
# already met and won't re-resolve it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e ".[ml,localize]"

ENV PIPELINE_SUB_HOST=ingest \
    PIPELINE_SUB_PORT=5567 \
    PIPELINE_PUB_HOST=0.0.0.0 \
    PIPELINE_PUB_PORT=5568 \
    PIPELINE_SAMPLE_RATE_HZ=100 \
    PIPELINE_WINDOW_S=2.0 \
    PIPELINE_STRIDE_S=0.5 \
    PIPELINE_CALIBRATION_S=5.0 \
    PIPELINE_N_SIGMAS=6.0 \
    PIPELINE_COUNTER_CHECKPOINT= \
    PIPELINE_LOCALIZER_CHECKPOINT=

EXPOSE 5568

CMD ["python", "-m", "pipeline"]
