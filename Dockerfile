# Ishinca inference image. Any CUDA 12.x PyTorch image works; the default is a public one.
# Validated on x86 (RTX 3090) with: docker build --build-arg BASE=<image that already has alpamayo1_5> -t ishinca:dev .
ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE}
RUN python -c "import alpamayo1_5" 2>/dev/null || \
    (apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/* && \
     pip install --no-cache-dir "git+https://github.com/NVlabs/alpamayo1.5.git" physical_ai_av)
WORKDIR /workspace/ishinca
COPY pyproject.toml README.md ./
COPY ishinca ./ishinca
COPY scripts ./scripts
COPY eval ./eval
RUN pip install --no-cache-dir --no-deps -e .
ENTRYPOINT ["python"]
