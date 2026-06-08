# TranscriptionSuite — NVIDIA DGX Spark (Blackwell ARM64) Deployment Guide

> Target platform: **NVIDIA DGX Spark** — GB10 (Blackwell architecture, `sm_121`), ARM64 (aarch64), Linux/Ubuntu, CUDA 13.0 (`cu130`)

This document describes how to modify the TranscriptionSuite server backend and Docker infrastructure to run natively on an NVIDIA DGX Spark. Two approaches are covered:

1. **Approach A (Recommended):** Use the NVIDIA NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.11-py3`) as the base image — batteries-included with CUDA 13.0, cuDNN, NCCL, and PyTorch pre-built for Blackwell/ARM64.
2. **Approach B:** Keep the existing `ubuntu:24.04` base but add a `cu130` PyTorch variant path (analogous to how `cu126` is handled today).

---

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Prerequisites](#prerequisites)
3. [Approach A — NGC Base Image (Recommended)](#approach-a--ngc-base-image-recommended)
4. [Approach B — Ubuntu Base with cu130 Variant](#approach-b--ubuntu-base-with-cu130-variant)
5. [Docker Compose Configuration](#docker-compose-configuration)
6. [Bootstrap Runtime Modifications](#bootstrap-runtime-modifications)
7. [Build Script Modifications](#build-script-modifications)
8. [GitHub Workflow Considerations](#github-workflow-considerations)
9. [Verification & Testing](#verification--testing)
10. [Troubleshooting](#troubleshooting)

---

## Platform Overview

| Property | Value |
|----------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| CUDA Compute Capability | `sm_121` |
| CPU Architecture | ARM64 (aarch64) |
| OS | Linux / Ubuntu |
| CUDA Toolkit | 13.0 (cu130) |
| Recommended Base Image | `nvcr.io/nvidia/pytorch:25.11-py3` |

The DGX Spark ships with an ARM64 processor paired with a Blackwell-class GPU. Current PyTorch wheels on `download.pytorch.org/whl/cu129` target x86_64 and `sm_70..sm_120`. Running on DGX Spark requires either:

- Pre-built ARM64 + Blackwell wheels from the NGC container, **or**
- Building PyTorch from source with `sm_121` support (complex, not recommended).

---

## Prerequisites

1. **NVIDIA Container Toolkit** installed and configured on the DGX Spark host:
   ```bash
   # Verify
   nvidia-ctk --version
   nvidia-smi   # Should show Blackwell GPU, CUDA 13.0 driver
   ```

2. **Docker** (≥ 24.0) with NVIDIA runtime:
   ```bash
   docker info | grep -i nvidia
   # OR verify CDI mode:
   ls /etc/cdi/nvidia.yaml
   ```

3. **Access to NGC registry** (nvcr.io):
   ```bash
   # Login (if image is gated — the PyTorch images are generally public)
   docker login nvcr.io
   # Username: $oauthtoken
   # Password: <your NGC API key>
   ```

---

## Approach A — NGC Base Image (Recommended)

This approach replaces the `ubuntu:24.04` base image with the NVIDIA PyTorch NGC container which ships pre-configured with:

- Ubuntu 24.04 (ARM64)
- Python 3.12 (NGC-managed)
- CUDA 13.0 toolkit + cuDNN + NCCL
- PyTorch (latest nightly with `sm_121` support)
- Optimized for DGX / Grace-Blackwell platforms

### A.1 — Create `server/docker/Dockerfile.dgx-spark`

Create a new Dockerfile alongside the existing one:

```dockerfile
# TranscriptionSuite — DGX Spark (Blackwell ARM64) Dockerfile
#
# Uses the NVIDIA NGC PyTorch container as the base.
# PyTorch, CUDA 13.0, cuDNN, and NCCL are pre-installed for ARM64 + sm_121.
# Application dependencies are installed at first container startup into /runtime/.venv.

FROM nvcr.io/nvidia/pytorch:25.11-py3 AS runtime

# Mark this as the DGX Spark / NGC variant
ARG PYTORCH_VARIANT=ngc

# OCI standard labels
LABEL org.opencontainers.image.title="TranscriptionSuite Server (DGX Spark)"
LABEL org.opencontainers.image.description="TranscriptionSuite server for NVIDIA DGX Spark (Blackwell ARM64, CUDA 13.0)"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"

# Global environment
ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"
# NGC image ships Python 3.12; override UV_PYTHON to match
ENV UV_PYTHON="3.12"
ENV PYTORCH_VARIANT="${PYTORCH_VARIANT}"
ENV BOOTSTRAP_RUNTIME_DIR="/runtime"
ENV BOOTSTRAP_CACHE_DIR="/runtime/cache"
ENV BOOTSTRAP_STATUS_FILE="/runtime/bootstrap-status.json"
ENV BOOTSTRAP_TIMEOUT_SECONDS="1800"
ENV BOOTSTRAP_REQUIRE_HF_TOKEN="false"
ENV BOOTSTRAP_LOG_CHANGES="true"
ENV UV_CACHE_DIR="/runtime/cache"
ENV HF_HOME="/models"
ENV TORCH_HOME="/models/torch-cache"
ENV PYTHONWARNINGS="ignore::UserWarning:pyannote.audio.core.io"

# Install additional runtime dependencies not in the NGC image.
# The NGC image already provides: Python, git, curl, build-essential, CUDA libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/0.10.8/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

# Create non-root user
RUN groupadd -g 10000 appuser && \
    useradd -u 10000 -g 10000 -m -s /bin/bash appuser

# Create app directory
WORKDIR /app

# Copy server application code and bootstrap tooling
COPY --chown=appuser:appuser server/backend/ ./server/
COPY --chown=appuser:appuser server/docker/entrypoint.py ./docker/
COPY --chown=appuser:appuser server/docker/bootstrap_runtime.py ./docker/
COPY server/docker/docker-entrypoint.sh ./docker/
RUN chmod 700 ./docker/docker-entrypoint.sh

# Copy default configuration
COPY server/config.yaml ./config.yaml

# Bake the variant identifier
RUN printf '%s\n' "${PYTORCH_VARIANT}" > /app/.pytorch_variant && \
    chown appuser:appuser /app/.pytorch_variant && \
    chmod 0644 /app/.pytorch_variant

# Create data directories
RUN mkdir -p /data/database /data/audio /data/logs /data/tokens /data/certs /models /user-config /certs /runtime /runtime/cache && \
    chown -R appuser:appuser /data /models /user-config /runtime

# Environment variables
ENV DATA_DIR="/data"
ENV SERVER_HOST="0.0.0.0"
ENV SERVER_PORT="9786"
ENV LOG_LEVEL="INFO"
# NGC image sets LD_LIBRARY_PATH for CUDA/cuDNN already; extend for runtime venv
ENV LD_LIBRARY_PATH="/runtime/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
# Blackwell-specific: enable flash attention v3 if available
ENV TORCH_CUDA_ARCH_LIST="12.1"

# Expose port
EXPOSE 9786

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT}/health || exit 1

ENTRYPOINT ["./docker/docker-entrypoint.sh"]
CMD []
```

### A.2 — Modify `server/docker/bootstrap_runtime.py`

The NGC base already ships PyTorch. The bootstrap script must detect this and **skip PyTorch installation** from the wheel index, while still installing the remaining application dependencies.

Add the `ngc` variant to the allowed variants:

```python
# In main(), around line 1504:
raw_variant = (os.environ.get("PYTORCH_VARIANT") or "").strip().lower()
if raw_variant in {"", "cu129"}:
    pytorch_variant = "cu129"
elif raw_variant in {"cu126", "cpu", "cu130", "ngc"}:
    pytorch_variant = raw_variant
else:
    log(f"Unknown PYTORCH_VARIANT={raw_variant!r}; falling back to cu129")
    pytorch_variant = "cu129"
```

Add the `ngc`/`cu130` handling to `run_dependency_sync()` (around line 425):

```python
variant_index_urls = {
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu130": "https://download.pytorch.org/whl/cu130",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# NGC variant: PyTorch is pre-installed in the base image system site-packages.
# Skip the pytorch wheel index entirely and rely on --system-site-packages
# inheritance in the venv (or exclude torch from install).
if pytorch_variant == "ngc":
    # Drop torch/torchaudio from install — they come from NGC base.
    cmd.extend([
        "--index-strategy", "unsafe-best-match",
    ])
    # Override: exclude pre-installed packages from sync
    cmd.extend(["--exclude-newer", "torch", "--exclude-newer", "torchaudio"])
```

> **Note:** The exact implementation depends on the `uv` version's support for excluding packages. An alternative is to create a separate `pyproject-dgx.toml` that omits `torch`/`torchaudio` from dependencies, or to use the `--no-install` flag if available. See [Approach A.3](#a3--alternative-pyproject-overlay) below.

### A.3 — Alternative: pyproject overlay for NGC

Create `server/backend/pyproject-dgx.toml` that inherits from the base but replaces torch dependencies with system torch:

```toml
# This is NOT a standalone pyproject.toml — it's used as documentation
# for how to strip torch from the dependency install on NGC images.
# In practice, you would modify the bootstrap to pass:
#   uv sync --no-install torch --no-install torchaudio
# Or use constraints to skip torch resolution.
```

A cleaner approach for the NGC variant in `bootstrap_runtime.py`:

```python
if pytorch_variant == "ngc":
    # NGC base has PyTorch pre-installed in /usr/local/lib/python3.12/...
    # Create the venv with --system-site-packages so it inherits torch
    # Then sync only non-torch deps.
    cmd.extend([
        "--index-strategy", "unsafe-best-match",
    ])
    # Add pip-style constraint to prevent re-installing torch:
    log("NGC variant: skipping torch/torchaudio install (pre-installed in base image)")
```

### A.4 — Modify `server/docker/docker-entrypoint.sh`

The entrypoint hardcodes `python3.13`. For NGC images (Python 3.12), update the bootstrap call:

```bash
# Replace the hardcoded python3.13 reference with a dynamic lookup:
# Before:
#   gosu appuser /usr/bin/python3.13 docker/bootstrap_runtime.py
# After:
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/usr/bin/python3.13}"
if [ ! -x "$BOOTSTRAP_PYTHON" ]; then
    # Fall back to whatever python3 is available (NGC uses python3.12)
    BOOTSTRAP_PYTHON="$(command -v python3)"
fi
gosu appuser "$BOOTSTRAP_PYTHON" docker/bootstrap_runtime.py
```

Similarly for the venv Python path detection, the existing glob fallback logic (lines 92-115) already handles alternate Python versions.

---

## Approach B — Ubuntu Base with cu130 Variant

This keeps the existing `ubuntu:24.04` Dockerfile but adds `cu130` as a recognized PyTorch variant. This is simpler but depends on ARM64 + cu130 wheels being available on `download.pytorch.org/whl/cu130`.

> **Important:** As of early 2026, PyTorch cu130 ARM64 wheels may not yet be published to the public index. Check availability with:
> ```bash
> pip index versions torch --index-url https://download.pytorch.org/whl/cu130 --platform linux_aarch64
> ```
> If unavailable, use [Approach A](#approach-a--ngc-base-image-recommended).

### B.1 — Dockerfile Modifications

Update `server/docker/Dockerfile` comments and ensure ARM64 compatibility:

```dockerfile
# PyTorch wheel-index variant (Issue #83):
#   cu129 (default) — modern GPUs, sm_70..sm_120 (Volta–Hopper, x86_64)
#   cu126           — legacy-GPU image, sm_50..sm_90 (Pascal/Maxwell)
#   cu130           — Blackwell GPUs, sm_121 (DGX Spark GB10, ARM64)
ARG PYTORCH_VARIANT=cu129
```

For ARM64, the `deadsnakes/ppa` must provide `python3.13` for `aarch64`. It does support ARM64, so the existing Dockerfile works on ARM if you build on-device or use `docker buildx` with platform targeting:

```bash
docker buildx build --platform linux/arm64 \
    --build-arg PYTORCH_VARIANT=cu130 \
    -t transcriptionsuite-server:dgx-spark \
    -f server/docker/Dockerfile .
```

### B.2 — Bootstrap Runtime Modifications

In `server/docker/bootstrap_runtime.py`:

```python
# Line ~1504 — add cu130 to allowed variants:
elif raw_variant in {"cu126", "cpu", "cu130"}:
    pytorch_variant = raw_variant

# Line ~425 — add cu130 index URL:
variant_index_urls = {
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu130": "https://download.pytorch.org/whl/cu130",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
```

---

## Docker Compose Configuration

### GPU Overlay for DGX Spark

Create `server/docker/docker-compose.gpu-dgx-spark.yml`:

```yaml
# NVIDIA GPU overlay for DGX Spark (Blackwell GB10)
# Usage:
#   docker compose -f docker-compose.yml \
#     -f docker-compose.linux-host.yml \
#     -f docker-compose.gpu-dgx-spark.yml up -d

services:
  transcriptionsuite:
    # Use the DGX Spark-specific image (Approach A)
    build:
      context: ../..
      dockerfile: server/docker/Dockerfile.dgx-spark
      args:
        PYTORCH_VARIANT: ngc
    # Or for Approach B, override the variant:
    # build:
    #   args:
    #     PYTORCH_VARIANT: cu130
    environment:
      - PYTORCH_VARIANT=ngc
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      # Blackwell-specific optimizations
      - TORCH_CUDA_ARCH_LIST=12.1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Alternatively, if the host uses CDI (common on DGX systems), use the existing `docker-compose.gpu-cdi.yml` — it already passes all GPUs via CDI.

### Start Command

```bash
# DGX Spark — Approach A (NGC image)
docker compose -f docker-compose.yml \
  -f docker-compose.linux-host.yml \
  -f docker-compose.gpu-dgx-spark.yml up -d

# DGX Spark — Approach B (ubuntu base + cu130)
PYTORCH_VARIANT=cu130 docker compose -f docker-compose.yml \
  -f docker-compose.linux-host.yml \
  -f docker-compose.gpu-cdi.yml up -d
```

---

## Build Script Modifications

To add DGX Spark as a variant in `build/docker-build-push.sh`:

### Add `dgx-spark` variant (around line 188):

```bash
# In the variant selection logic:
case "$variant" in
    default)
        IMAGE_NAME="$DEFAULT_IMAGE_NAME"
        pytorch_variant="cu129"
        dockerfile="server/docker/Dockerfile"
        ;;
    legacy)
        IMAGE_NAME="${DEFAULT_IMAGE_NAME}-legacy"
        pytorch_variant="cu126"
        dockerfile="server/docker/Dockerfile"
        ;;
    dgx-spark)
        IMAGE_NAME="${DEFAULT_IMAGE_NAME}-dgx-spark"
        pytorch_variant="ngc"
        dockerfile="server/docker/Dockerfile.dgx-spark"
        ;;
    vulkan-wsl2)
        IMAGE_NAME="${DEFAULT_IMAGE_NAME}-vulkan-wsl2"
        pytorch_variant="cu129"
        dockerfile="server/docker/whisper-cpp-vulkan-wsl2.Dockerfile"
        ;;
esac
```

### Build the DGX Spark image on-device:

```bash
# On the DGX Spark itself (native ARM64 build):
./build/docker-build-push.sh --variant dgx-spark --build v1.0.0

# Or cross-build from an x86 host (requires QEMU emulation — slow):
docker buildx build --platform linux/arm64 \
    -t ghcr.io/homelab-00/transcriptionsuite-server-dgx-spark:latest \
    -f server/docker/Dockerfile.dgx-spark .
```

---

## GitHub Workflow Considerations

The existing CI/CD pipeline (`.github/workflows/release.yml`) **does not need modification** for DGX Spark support because:

1. **Docker images are built locally** — The `docker-build-push.sh` script is the primary build mechanism (GitHub free runners lack disk space for large images).
2. **ARM64 builds require native hardware** — GitHub-hosted runners are x86_64. Building an NGC-based ARM64 image in CI would require either:
   - Self-hosted ARM64 runners (recommended if you need CI/CD for DGX Spark)
   - QEMU emulation (very slow, 30+ minutes for this image)
3. **The dashboard CI** (quality checks, CodeQL) is unaffected — it tests the Electron/Node.js frontend, not the Docker backend.

### Optional: Add a self-hosted runner job

If you have access to an ARM64 CI runner (e.g., the DGX Spark itself or a Grace CPU server), you could add:

```yaml
# .github/workflows/release.yml — optional addition
build-dgx-spark:
  runs-on: [self-hosted, linux, arm64]
  if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
  steps:
    - uses: actions/checkout@v4
    - name: Build DGX Spark image
      run: |
        docker build \
          -t ghcr.io/${{ github.repository_owner }}/transcriptionsuite-server-dgx-spark:${{ github.ref_name }} \
          -f server/docker/Dockerfile.dgx-spark .
    - name: Push to GHCR
      run: |
        echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
        docker push ghcr.io/${{ github.repository_owner }}/transcriptionsuite-server-dgx-spark:${{ github.ref_name }}
```

This is **optional** — most DGX Spark users will build locally.

---

## Verification & Testing

### 1. Verify CUDA and GPU detection inside the container

```bash
docker exec -it transcriptionsuite-container bash

# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, Arch: {torch.cuda.get_device_capability(0)}')"
# Expected: CUDA: True, Device: NVIDIA GB10, Arch: (12, 1)

# Check compute capability
python3 -c "import torch; assert torch.cuda.get_device_capability(0) >= (12, 1), 'Blackwell not detected'"
```

### 2. Health check

```bash
curl -f http://localhost:9786/health
# Expected: 200 OK with JSON status
```

### 3. Run a transcription

Upload an audio file via the dashboard or API to confirm GPU-accelerated inference works.

### 4. Verify memory allocation

```bash
docker exec -it transcriptionsuite-container bash -c \
  "python3 -c \"import torch; t = torch.zeros(1024, 1024, device='cuda'); print(f'Allocated {t.element_size() * t.nelement() / 1e6:.1f} MB on GPU')\""
```

---

## Troubleshooting

### NGC image not found / pull fails

```
Error: pull access denied for nvcr.io/nvidia/pytorch
```

**Fix:** Authenticate with NGC:
```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API key from https://ngc.nvidia.com/setup/api-key>
```

### `sm_121` not supported error

```
CUDA error: no kernel image is available for execution on the device
```

**Cause:** PyTorch was not compiled with `sm_121` support.
**Fix:** Use Approach A (NGC image) which includes Blackwell kernels. The public `cu129` wheels from PyPI do not include `sm_121`.

### ARM64 package compatibility

Some pip packages may not have ARM64 wheels. If `uv sync` fails:

```bash
# Check which package failed
docker logs transcriptionsuite-container | grep -i "error\|failed"

# Common fix: ensure build-essential is installed for source compilation
# The NGC base image includes compilers; verify with:
docker exec -it transcriptionsuite-container gcc --version
```

### Python version mismatch

The NGC image ships **Python 3.12**, while the standard Dockerfile uses **Python 3.13**. The `pyproject.toml` requires `>=3.13,<3.14`.

**Fix for Approach A:** Relax the Python version constraint for DGX Spark builds:

```toml
# server/backend/pyproject.toml — for DGX Spark compatibility
requires-python = ">=3.12,<3.14"
```

Or wait for an NGC image that ships Python 3.13+ (check future `25.xx-py3` tags).

### Bootstrap timeout

The DGX Spark has limited CPU threads compared to desktop GPUs. If bootstrap times out:

```bash
# Increase timeout (default 1800s = 30min)
BOOTSTRAP_TIMEOUT_SECONDS=3600 docker compose ... up -d
```

---

## Summary of Required Code Changes

| File | Change | Required For |
|------|--------|-------------|
| `server/docker/Dockerfile.dgx-spark` | New file — NGC-based Dockerfile | Approach A |
| `server/docker/Dockerfile` | Add `cu130` to variant comment | Approach B |
| `server/docker/bootstrap_runtime.py` | Add `cu130`/`ngc` to allowed variants + index URLs | Both |
| `server/docker/docker-entrypoint.sh` | Dynamic Python path (not hardcoded 3.13) | Approach A |
| `server/docker/docker-compose.gpu-dgx-spark.yml` | New compose overlay | Both |
| `build/docker-build-push.sh` | Add `dgx-spark` variant | Optional (convenience) |
| `server/backend/pyproject.toml` | Relax `requires-python` to `>=3.12` | Approach A (if NGC ships 3.12) |
| `.github/workflows/release.yml` | Add ARM64 self-hosted job | Optional |

---

## Quick-Start (TL;DR)

For the fastest path to running TranscriptionSuite on DGX Spark:

```bash
# 1. Clone the repo on your DGX Spark
git clone https://github.com/homelab-00/TranscriptionSuite.git
cd TranscriptionSuite

# 2. Create the DGX Spark Dockerfile (copy from Approach A.1 above)
#    or apply the patches described in this guide

# 3. Build the image natively on the DGX Spark
docker build -t transcriptionsuite-server:dgx-spark \
    -f server/docker/Dockerfile.dgx-spark .

# 4. Start with GPU access
cd server/docker
docker compose -f docker-compose.yml \
    -f docker-compose.linux-host.yml \
    -f docker-compose.gpu-dgx-spark.yml up -d

# 5. Verify
curl http://localhost:9786/health
docker logs transcriptionsuite-container | tail -20
```
