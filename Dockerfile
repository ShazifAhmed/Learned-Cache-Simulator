# Reproducible environment for learned-cache-sim.
#   docker build -t learned-cache-sim .
#   docker run --rm -v "$PWD/results:/app/results" learned-cache-sim   # runs `cachesim demo`
#
# gcc is included so `make traces` (real workload capture) works inside the container too.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run the full benchmark and write charts to /app/results.
CMD ["cachesim", "demo"]
