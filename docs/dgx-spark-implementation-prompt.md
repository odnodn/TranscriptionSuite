# DGX Spark Docker Integration — Coding Agent Prompt

> **Purpose:** Full-featured prompt for delegating the DGX Spark Docker integration
> to a coding agent in TranscriptionSuite (or a fork/another repository).
> Copy this entire document as the prompt input.

---

## Task

Integrate NVIDIA DGX Spark support into TranscriptionSuite's Docker infrastructure.
The DGX Spark uses a Blackwell GB10 SoC (CUDA compute capability sm_121, aarch64
architecture) that is not supported by the current `ubuntu:24.04`-based Dockerfile
or by standard PyTorch cu129/cu126 wheels (which ship sm_70–sm_120 only).

**The solution** uses `nvcr.io/nvidia/pytorch:25.09-py3` as the base image (the NGC
PyTorch container ships pre-built CUDA 12.x with sm_121 on aarch64) and follows
TranscriptionSuite's existing compose overlay pattern to integrate cleanly.

---

## Repository Context

TranscriptionSuite is a multi-platform transcription suite with:

- **Backend:** Python 3.13 (FastAPI + WebSocket), runtime dependencies bootstrapped
  at first container start via `server/docker/bootstrap_runtime.py` + `uv sync`.
- **Frontend:** Electron + React dashboard that manages Docker containers.
- **Docker architecture:** Layered compose files with a base (`docker-compose.yml`)
  and platform/GPU overlays (`docker-compose.linux-host.yml`, `docker-compose.gpu.yml`,
  `docker-compose.desktop-vm.yml`, etc.).
- **PyTorch variant system:** Build-arg `PYTORCH_VARIANT` (cu129 | cu126 | cpu) selects
  the PyTorch wheel index at bootstrap time. The variant is baked into
  `/app/.pytorch_variant` at build time for runtime cross-checking.

### Key Files to Understand Before Editing

| File | Role |
|------|------|
| `server/docker/Dockerfile` | Standard Ubuntu-based image (DO NOT MODIFY) |
| `server/docker/docker-compose.yml` | Base compose (DO NOT MODIFY) |
| `server/docker/docker-compose.gpu.yml` | NVIDIA GPU overlay |
| `server/docker/bootstrap_runtime.py` | Runtime dependency bootstrapper — parses `PYTORCH_VARIANT` |
| `server/docker/docker-entrypoint.sh` | Container entrypoint — handles TLS, CA certs, bootstrap, privilege drop |
| `dashboard/src/types/runtime.ts` | `RuntimeProfile` type union |
| `dashboard/electron/dockerManager.ts` | `composeFileArgs()` — selects compose overlays per profile |
| `dashboard/src/services/instanceMatrix.ts` | Maps runtime profiles to model/family choices |

---

## Deliverables

### 1. `server/docker/Dockerfile.dgx-spark` (NEW)

Create a Dockerfile based on `nvcr.io/nvidia/pytorch:25.09-py3` that:

- Uses `ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3` for configurability.
- Sets `ARG PYTORCH_VARIANT=dgx-spark`.
- Sets `ENV UV_PYTHON="3.12"` (NGC ships Python 3.12, not 3.13).
- Installs only the delta system packages not in the NGC base: `libsndfile1`,
  `libportaudio2`, `portaudio19-dev`, `gosu`.
- Installs `uv` (same version: 0.10.8).
- Creates the `appuser` (UID 10000, GID 10000) — same as the standard Dockerfile.
- Copies the same application code, config, and entrypoint scripts.
- Bakes `PYTORCH_VARIANT` into `/app/.pytorch_variant`.
- Sets `LD_LIBRARY_PATH` to include both `python3.12` site-packages paths AND
  preserves the NGC base image's existing `LD_LIBRARY_PATH` with `${LD_LIBRARY_PATH}`.
- Uses the same `ENTRYPOINT`, `HEALTHCHECK`, and `EXPOSE` as the standard Dockerfile.
- Add an attribution comment: `# Adapted from nemotron-asr-server (https://github.com/briancaffey/nemotron-asr-server) — base image selection pattern for DGX Spark sm_121 aarch64 support.`

### 2. `server/docker/docker-compose.dgx-spark.yml` (NEW)

Create a compose overlay that:

- Overrides `build.dockerfile` to `server/docker/Dockerfile.dgx-spark`.
- Passes `BASE_IMAGE` and `PYTORCH_VARIANT` as build args.
- Sets the image tag to `dgx-spark`.
- Sets `PYTORCH_VARIANT=dgx-spark` in the environment.

### 3. `server/docker/bootstrap_runtime.py` (MODIFY)

**Variant parsing** (around line 1684–1691):

```python
# Change this:
elif raw_variant in {"cu126", "cpu"}:
    pytorch_variant = raw_variant

# To this:
elif raw_variant in {"cu126", "cpu", "dgx-spark"}:
    pytorch_variant = raw_variant
```

**Dependency sync** (`_sync_deps()`, around line 494–510):

The `dgx-spark` variant should follow the **default cu129 code path** (with `--frozen`)
because PyTorch is pre-installed in the NGC base image. The uv sync will see the
existing torch installation as satisfied and skip it. Do NOT add `dgx-spark` to
`variant_index_urls` — it does not need an index swap.

**Fingerprint** (`_compute_bootstrap_fingerprint()`, around line 219):

No changes needed — `pytorch_variant` is already included in the fingerprint.

### 4. `dashboard/src/types/runtime.ts` (MODIFY)

Add `'dgx-spark'` to the `RUNTIME_PROFILES` const array:

```typescript
const RUNTIME_PROFILES = ['gpu', 'cpu', 'vulkan', 'vulkan-wsl2', 'metal', 'dgx-spark'] as const;
```

### 5. `dashboard/electron/dockerManager.ts` (MODIFY)

In `composeFileArgs()` (around line 1366–1387), add a block for `dgx-spark`:

```typescript
// DGX Spark overlay (NVIDIA Blackwell GB10 aarch64)
if (runtimeProfile === 'dgx-spark') {
  files.push('docker-compose.dgx-spark.yml');
}
```

The GPU overlay (`docker-compose.gpu.yml`) should also be included for DGX Spark
since it needs the NVIDIA device reservation. Either:
- Treat `dgx-spark` like `gpu` in the GPU overlay block, OR
- Include `docker-compose.gpu.yml` inside the dgx-spark block.

### 6. `dashboard/src/services/instanceMatrix.ts` (MODIFY)

Map `dgx-spark` to the same model/family capabilities as `gpu`:

- `familyChoicesFor('dgx-spark')` → same as `gpu`
- `defaultMainModelFor('dgx-spark')` → same as `gpu`
- `liveModelsFor('dgx-spark')` → same as `gpu`
- `isFamilyChoiceEnabledFor(id, 'dgx-spark')` → same as `gpu`

### 7. Documentation Updates

Add a "DGX Spark" subsection to `docs/deployment-guide.md` with:
- Quick start commands (compose up with overlays)
- Build instructions (local build with compose)
- NGC base image version pinning via `DGX_BASE_IMAGE` env var
- Note about the large image size (~15 GB for NGC base)

---

## Constraints

1. **DO NOT modify** `server/docker/Dockerfile` or `server/docker/docker-compose.yml` — the DGX Spark support is entirely additive via new files and compose overlays.
2. **Follow existing patterns** — look at how `cu126` and `vulkan` are integrated for precedent.
3. **Keep the same entrypoint** — `docker-entrypoint.sh` must work unmodified. Its glob-based `LD_LIBRARY_PATH` discovery (lines 118–141) already handles different Python versions.
4. **Test without DGX hardware** — unit tests for variant parsing, compose file selection, and runtime profile type guards can run anywhere.
5. **Use `uv`, never `pip`** — project convention.
6. **No AI attribution** — do not add `Co-Authored-By` trailers or assistant mentions.

---

## Testing Checklist

### Unit Tests (run anywhere)

- [ ] `bootstrap_runtime.py`: `PYTORCH_VARIANT=dgx-spark` is parsed correctly (not falling back to cu129)
- [ ] `bootstrap_runtime.py`: `dgx-spark` uses the `--frozen` code path (not index swap)
- [ ] `composeFileArgs('dgx-spark')`: returns `['-f', 'docker-compose.yml', '-f', 'docker-compose.linux-host.yml', '-f', 'docker-compose.gpu.yml', '-f', 'docker-compose.dgx-spark.yml']` (on Linux)
- [ ] `isRuntimeProfile('dgx-spark')`: returns `true`
- [ ] Existing profiles (`gpu`, `cpu`, `vulkan`, `vulkan-wsl2`, `metal`): unchanged behavior

### Integration Tests (DGX Spark hardware)

- [ ] `docker compose -f ... build` succeeds
- [ ] Container starts and `/health` responds 200
- [ ] `nvidia-smi` inside container shows GB10
- [ ] `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"` prints `True NVIDIA ...`
- [ ] Sample transcription completes successfully

### Regression Tests

- [ ] Standard `cu129` build unchanged
- [ ] `cu126` build unchanged
- [ ] `cpu` build unchanged
- [ ] Dashboard compose file selection for all existing profiles unchanged

---

## Commit Style

```
feat(docker): add DGX Spark Docker integration (sm_121, aarch64)

* feat(docker): add Dockerfile.dgx-spark using NGC PyTorch base image
  * nvcr.io/nvidia/pytorch:25.09-py3 for Blackwell GB10 sm_121 support
  * Python 3.12 (NGC-shipped), delta system packages only

* feat(docker): add docker-compose.dgx-spark.yml overlay
  * overrides Dockerfile, image tag, and PYTORCH_VARIANT

* fix(bootstrap): recognize dgx-spark as valid PYTORCH_VARIANT
  * uses default --frozen sync path (PyTorch pre-installed in NGC base)

* feat(dashboard): add dgx-spark runtime profile
  * types, compose file selection, model/family mapping

* docs(deployment): add DGX Spark setup instructions
```

---

## Reference: Existing Variant Flow

```
Dockerfile ARG PYTORCH_VARIANT=cu129
  → ENV PYTORCH_VARIANT="${PYTORCH_VARIANT}"
  → /app/.pytorch_variant (baked at build)
  → bootstrap_runtime.py reads PYTORCH_VARIANT env
    → if cu129: uv sync --frozen (default)
    → if cu126: uv sync --index pytorch-cu129=https://…/cu126
    → if cpu:   uv sync --index pytorch-cu129=https://…/cpu
    → if dgx-spark: uv sync --frozen (NEW — same as cu129, PyTorch is in NGC base)
```

## Reference: Compose Overlay Pattern

```
docker compose \
  -f docker-compose.yml              # base: service, env, volumes
  -f docker-compose.linux-host.yml   # platform: host networking
  -f docker-compose.gpu.yml          # GPU: NVIDIA device reservation
  -f docker-compose.dgx-spark.yml   # DGX: Dockerfile + image + env override
  up -d
```
