# DGX Spark Docker Integration — Implementation Plan

> **Status:** Planning  
> **Target platform:** NVIDIA DGX Spark — Blackwell GB10 (sm_121), aarch64, CUDA 12.x  
> **Issue:** [#1](https://github.com/homelab-00/TranscriptionSuite/issues/1)

---

## 1. Problem Statement

TranscriptionSuite currently supports these Docker platforms:

| Platform | Base Image | CUDA / Arch | Dockerfile |
|----------|-----------|-------------|------------|
| Linux x86_64 (modern GPU) | `ubuntu:24.04` | cu129, sm_70–sm_120 | `Dockerfile` |
| Linux x86_64 (legacy GPU) | `ubuntu:24.04` | cu126, sm_50–sm_90 | `Dockerfile` (build-arg) |
| CPU-only | `ubuntu:24.04` | none | `Dockerfile` (build-arg) |

The **DGX Spark** cannot use any of these because:

1. **Architecture mismatch:** DGX Spark is `aarch64`; the `deadsnakes/ppa` used for Python 3.13 does not publish aarch64 packages.
2. **CUDA compute capability gap:** The Blackwell GB10 SoC requires `sm_121` kernels. PyTorch `cu129` wheels ship `sm_70`–`sm_120` only — no `sm_121`.
3. **Solution:** Use `nvcr.io/nvidia/pytorch:25.09-py3` (or later) as the base image. This NGC container ships CUDA 12.x with sm_121 support on aarch64, pre-installed PyTorch, and Python 3.12.

---

## 2. Architecture Overview

The integration follows the existing **compose overlay** pattern:

```
docker-compose.yml                     ← base (unchanged)
  + docker-compose.linux-host.yml      ← host networking (unchanged)
  + docker-compose.gpu.yml             ← NVIDIA GPU reservation (unchanged)
  + docker-compose.dgx-spark.yml       ← NEW: overrides Dockerfile + image tag + env
```

The DGX Spark overlay swaps the build to `Dockerfile.dgx-spark` (NGC base) and
sets `PYTORCH_VARIANT=dgx-spark`.

### File Inventory

| File | Action | Purpose |
|------|--------|---------|
| `server/docker/Dockerfile.dgx-spark` | **New** | NGC-based Dockerfile for aarch64 sm_121 |
| `server/docker/docker-compose.dgx-spark.yml` | **New** | Compose overlay: build + image + env overrides |
| `server/docker/bootstrap_runtime.py` | **Modify** | Recognize `dgx-spark` variant, skip PyTorch index swap |
| `dashboard/src/types/runtime.ts` | **Modify** | Add `'dgx-spark'` to `RUNTIME_PROFILES` |
| `dashboard/electron/dockerManager.ts` | **Modify** | Add `dgx-spark` branch in `composeFileArgs()` |
| `dashboard/src/services/instanceMatrix.ts` | **Modify** | Map `dgx-spark` profile to model/family choices |
| `docs/deployment-guide.md` | **Modify** | Add DGX Spark setup instructions |

---

## 3. Detailed Changes

### 3.1 Dockerfile.dgx-spark (New)

**Already created** — see `server/docker/Dockerfile.dgx-spark`.

Key differences from the standard `Dockerfile`:

| Aspect | Standard | DGX Spark |
|--------|----------|-----------|
| Base image | `ubuntu:24.04` | `nvcr.io/nvidia/pytorch:25.09-py3` |
| Python version | 3.13 (deadsnakes PPA) | 3.12 (NGC pre-installed) |
| CUDA / cuDNN | Installed at bootstrap via PyTorch wheels | Pre-installed in base image |
| `UV_PYTHON` | `3.13` | `3.12` |
| `LD_LIBRARY_PATH` | Points to `python3.13/…` | Points to `python3.12/…`, extends NGC paths |
| System packages | Full install (build-essential, python, etc.) | Minimal delta (libsndfile1, portaudio, gosu) |
| `PYTORCH_VARIANT` | `cu129` / `cu126` / `cpu` | `dgx-spark` |

### 3.2 docker-compose.dgx-spark.yml (New)

**Already created** — see `server/docker/docker-compose.dgx-spark.yml`.

Overrides:
- `build.dockerfile` → `Dockerfile.dgx-spark`
- `build.args.BASE_IMAGE` → configurable via `DGX_BASE_IMAGE` env
- `build.args.PYTORCH_VARIANT` → `dgx-spark`
- `image` tag → `dgx-spark`
- `environment.PYTORCH_VARIANT` → `dgx-spark`

### 3.3 bootstrap_runtime.py Changes

The bootstrap script's variant parser (around line 1684) must recognize `dgx-spark`:

```python
# Current:
elif raw_variant in {"cu126", "cpu"}:
    pytorch_variant = raw_variant

# After:
elif raw_variant in {"cu126", "cpu", "dgx-spark"}:
    pytorch_variant = raw_variant
```

The `_sync_deps()` function (around line 494) must handle `dgx-spark`:

```python
# In variant_index_urls dict and the conditional below:
# dgx-spark should NOT swap the PyTorch index — PyTorch is pre-installed
# in the NGC base image. The bootstrap should use --frozen (like cu129)
# OR skip torch entirely and only install non-torch dependencies.
```

**Option A (recommended):** Treat `dgx-spark` like `cu129` (default path with `--frozen`). The NGC base already has PyTorch; uv sync will see it satisfied and skip it.

**Option B:** Add `dgx-spark` to `variant_index_urls` pointing at `https://download.pytorch.org/whl/cu129` (since NGC's CUDA 12.x is compatible). This is functionally equivalent since PyTorch is already installed.

### 3.4 Dashboard Changes

#### 3.4.1 `dashboard/src/types/runtime.ts`

```typescript
// Add 'dgx-spark' to the profiles array:
const RUNTIME_PROFILES = ['gpu', 'cpu', 'vulkan', 'vulkan-wsl2', 'metal', 'dgx-spark'] as const;
```

#### 3.4.2 `dashboard/electron/dockerManager.ts` — `composeFileArgs()`

```typescript
// After the vulkan block (around line 1377):
if (runtimeProfile === 'dgx-spark') {
  files.push('docker-compose.dgx-spark.yml');
}
```

The `dgx-spark` profile implicitly includes the GPU overlay since the compose
overlay already sets the correct Dockerfile. The `docker-compose.gpu.yml` overlay
is still needed for the NVIDIA device reservation.

#### 3.4.3 `dashboard/src/services/instanceMatrix.ts`

Map `dgx-spark` to the same model/family choices as `gpu` (it has the same CUDA capabilities, plus NeMo support is natural on DGX):

```typescript
// In familyChoicesFor():
case 'dgx-spark':
  return GPU_FAMILIES;  // same as 'gpu'

// In defaultMainModelFor():
case 'dgx-spark':
  return DEFAULT_GPU_MODEL;
```

### 3.5 Documentation Updates

Add a section to `docs/deployment-guide.md`:

```markdown
### DGX Spark

The DGX Spark uses the NVIDIA Blackwell GB10 SoC (sm_121, aarch64) which
requires a specialized Docker image based on the NGC PyTorch container.

**Quick start:**
​```bash
docker compose -f docker-compose.yml \
               -f docker-compose.linux-host.yml \
               -f docker-compose.gpu.yml \
               -f docker-compose.dgx-spark.yml up -d
​```

**Build locally:**
​```bash
docker compose -f docker-compose.yml \
               -f docker-compose.dgx-spark.yml build
​```

To pin a specific NGC base image version:
​```bash
DGX_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3 \
  docker compose -f docker-compose.yml \
                 -f docker-compose.dgx-spark.yml build
​```
```

---

## 4. Testing Strategy

### 4.1 Unit Tests (No DGX Hardware Needed)

| Test | What It Validates |
|------|-------------------|
| `bootstrap_runtime.py` variant parsing | `dgx-spark` is accepted, not falling back to `cu129` |
| `composeFileArgs('dgx-spark')` | Returns correct compose overlay list |
| `instanceMatrix` model mapping | `dgx-spark` maps to GPU-tier models |
| `isRuntimeProfile('dgx-spark')` | Type guard accepts the new profile |

### 4.2 Integration Tests (DGX Spark Hardware)

| Test | What It Validates |
|------|-------------------|
| `docker compose build` with DGX overlay | Image builds successfully on aarch64 |
| Container startup + health check | Server starts and responds to `/health` |
| `nvidia-smi` inside container | GPU is visible with sm_121 |
| `python -c "import torch; print(torch.cuda.get_device_name())"` | PyTorch sees the GB10 |
| Transcription end-to-end | A sample audio file transcribes correctly |

### 4.3 Regression Tests

- Existing `cu129` and `cu126` builds must be unaffected.
- The `cpu` downgrade path must still work.
- Dashboard compose file selection for non-DGX profiles must be unchanged.

---

## 5. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| NGC base image ships different Python minor version | Medium | `UV_PYTHON` is set to `3.12`; verify at build time |
| NGC base image is very large (~15 GB) | Certain | Document the size tradeoff; this is inherent to NGC |
| PyTorch version in NGC conflicts with `uv.lock` pins | Medium | Use `--no-deps` for torch or exclude it from sync |
| `gosu` not available for aarch64 on NGC base | Low | NGC base is Ubuntu-based; gosu installs normally |
| NeMo extra works differently on NGC (NeMo may be pre-installed) | Medium | Detect pre-installed NeMo and skip re-install |

---

## 6. Future Considerations

1. **Multi-arch image manifest:** Publish a single `transcriptionsuite-server:latest` that resolves to the correct image per architecture (amd64 → ubuntu base, arm64 → NGC base). Requires a CI matrix build.

2. **NGC base image version pinning:** The `25.09-py3` tag will age. Add a CI job that tests against the latest NGC monthly release. When upgrading, check the Python version shipped in the new tag and update `UV_PYTHON` in the Dockerfile if it changes. NGC tags follow a `YY.MM-py3` convention; prefer LTS-aligned quarterly updates (e.g. `25.09`, `26.01`, `26.05`).

3. **NeMo integration:** The DGX Spark is a natural fit for NVIDIA Parakeet ASR models via NeMo. The `INSTALL_NEMO=true` flag should work out of the box since NeMo's dependencies are partially satisfied by the NGC base.

4. **Shared entrypoint:** Both Dockerfiles use the same `docker-entrypoint.sh`. The Python version difference (`3.13` vs `3.12`) only affects the `LD_LIBRARY_PATH` glob — the entrypoint's glob-based fallback (lines 118–141 of `docker-entrypoint.sh`) already handles this.
