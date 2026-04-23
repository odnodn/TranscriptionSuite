# Prompt: Standalone Parakeet/Sortformer OpenAI-Compatible Server

This document provides a **standalone prompt** that can be handed to an AI coding
assistant (or used as a starting point for a developer) to build a minimal,
self-contained FastAPI server that:

- Accepts `POST /v1/audio/transcriptions` (OpenAI-compatible)
- Runs ASR via **MLX Parakeet** on Apple Silicon (Metal)
- Runs speaker diarization via **Sortformer** (also Metal, no HuggingFace token)
- Returns `diarized_json` (speaker-attributed segments + words)

The resulting server can be run standalone — no Electron, no Docker, no CUDA — on
any Apple Silicon Mac with `uv` and `ffmpeg` installed.

---

## The Prompt

> Copy everything between the `---begin prompt---` and `---end prompt---` markers
> and paste it into your AI assistant of choice.

---begin prompt---

**Task**: Build a minimal, self-contained Python FastAPI server that provides an
OpenAI-compatible audio transcription endpoint with speaker diarization, running
entirely on Apple Silicon / Metal via the `mlx` package ecosystem.

---

### Requirements

1. **Runtime**: Python 3.13, `uv` for dependency management.  macOS arm64 only.

2. **Dependencies** (all pip-installable, no CUDA required):
   ```
   fastapi>=0.135.0
   uvicorn[standard]>=0.41.0
   python-multipart>=0.0.22
   parakeet-mlx>=0.2.0          # ASR
   mlx-audio>=0.4.1             # Sortformer diarization
   soundfile>=0.13.1
   numpy>=2.4.0
   scipy>=1.17.0
   ffmpeg-python>=0.2.0
   ```

3. **Endpoint**: `POST /v1/audio/transcriptions` (OpenAI Audio API compatible).
   - Accept `multipart/form-data` with fields: `file` (audio), `model` (ignored),
     `language` (ignored — Parakeet auto-detects), `response_format`, `diarization`
     (bool, default false), `expected_speakers` (int, optional).
   - Supported `response_format` values: `json`, `text`, `verbose_json`, `srt`,
     `vtt`, `diarized_json`.

4. **ASR**: Use `parakeet-mlx` with model `mlx-community/parakeet-tdt-0.6b-v3`.
   - Load on first request (lazy load).
   - Produce word-level timestamps (needed for diarization merge).
   - Use local attention for audio longer than 120 s to bound Metal memory.
   - Split audio into 120 s chunks with 15 s overlap for very long files.

5. **Diarization**: Use `mlx-audio` Sortformer with model
   `mlx-community/diar_sortformer_4spk-v1-fp32`.
   - Load on first diarization request (lazy load).
   - Use streaming inference with 5-second chunks (`generate_stream`).
   - Supports up to 4 speakers; no HuggingFace token required.
   - Run **sequentially after ASR** (not in parallel) to avoid Metal GPU deadlock
     when both models compete for the same Metal command queue.

6. **Speaker merge**: After obtaining ASR words and diarization segments, assign
   a speaker label to each word:
   - For each word, find the diarization segment with maximum temporal overlap
     (inflate the word interval by ±40 ms to account for boundary jitter).
   - Fallback chain: max-overlap → midpoint containment → nearest turn (≤120 ms)
     → previous word's speaker (gap ≤200 ms) → "UNKNOWN".
   - Smooth micro-turns: if a single-word run is sandwiched between the same
     speaker on both sides, relabel it to the surrounding speaker.
   - Group speaker-labelled words into contiguous segments.

7. **Audio loading**: Use `ffmpeg-python` to decode any audio format (mp3, m4a,
   wav, ogg, flac, …) to 16 kHz mono float32 PCM.  Fall back to `soundfile` for
   WAV/FLAC if ffmpeg is unavailable.

8. **Memory management**: After each inference call, run `mlx.core.mx.clear_cache()`
   to release intermediate Metal buffers.

9. **Health endpoint**: `GET /health` returning `{"status": "ok", "backend": "mlx_parakeet"}`.

10. **OpenAPI docs**: Available at `/docs` (Swagger UI).

---

### File layout

Produce a single directory `parakeet_server/` with the following structure:

```
parakeet_server/
├── pyproject.toml          # uv project file
├── server.py               # FastAPI application (single file)
└── README.md               # Quick-start instructions
```

---

### `pyproject.toml`

```toml
[project]
name = "parakeet-server"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
dependencies = [
    "fastapi>=0.135.0",
    "uvicorn[standard]>=0.41.0",
    "python-multipart>=0.0.22",
    "parakeet-mlx>=0.2.0",
    "mlx-audio>=0.4.1",
    "soundfile>=0.13.1",
    "numpy>=2.4.0",
    "scipy>=1.17.0",
    "ffmpeg-python>=0.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

### `server.py` — expected structure

The single-file server should follow this structure (implement each section fully):

```python
"""
Minimal standalone Parakeet + Sortformer OpenAI-compatible transcription server.
Apple Silicon (Metal) only.  No HuggingFace token required.

Usage:
    uv run uvicorn server:app --host 127.0.0.1 --port 5167 --workers 1
"""

from __future__ import annotations
import asyncio
import functools
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="Parakeet+Sortformer STT", version="0.1.0")

# ─── Model state (module-level singletons, loaded lazily) ────────────────────

_asr_model = None       # parakeet-mlx model
_diar_model = None      # mlx-audio Sortformer model

# ─── Audio loading ───────────────────────────────────────────────────────────

def load_audio_ffmpeg(path: str, target_sr: int = 16000) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 PCM using ffmpeg."""
    ...

# ─── ASR (MLX Parakeet) ──────────────────────────────────────────────────────

def _get_asr_model():
    """Lazy-load the Parakeet model on first call."""
    global _asr_model
    if _asr_model is None:
        from parakeet_mlx import from_pretrained
        _asr_model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    return _asr_model

def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Run Parakeet-TDT inference and return:
        {"text": str, "segments": list[dict], "words": list[dict]}
    Each word dict: {"word": str, "start": float, "end": float, "probability": float}
    Each segment:   {"text": str, "start": float, "end": float, "words": list[dict]}

    - Use local attention for audio > 120 s.
    - Use 120 s chunks with 15 s overlap for audio > 120 s.
    - Call mx.clear_cache() after inference.
    """
    ...

def _tokens_to_words(tokens) -> list[dict]:
    """
    Convert parakeet-mlx AlignedToken objects to word-level dicts.
    Tokens whose .text starts with a space mark a new word boundary.
    """
    ...

# ─── Diarization (MLX Sortformer) ────────────────────────────────────────────

def _get_diar_model():
    """Lazy-load the Sortformer model on first call."""
    global _diar_model
    if _diar_model is None:
        from mlx_audio.vad import load as load_sortformer
        _diar_model = load_sortformer("mlx-community/diar_sortformer_4spk-v1-fp32")
    return _diar_model

def diarize(audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
    """
    Run Sortformer diarization and return a list of speaker segments:
        [{"start": float, "end": float, "speaker": str}, ...]

    - Write audio to a temp WAV (Sortformer expects a file path).
    - Use streaming inference: generate_stream(chunk_duration=5.0, threshold=0.5).
    - Call mx.clear_cache() after inference.
    """
    ...

# ─── Speaker merge ───────────────────────────────────────────────────────────

def assign_speakers(
    words: list[dict],
    diar_segments: list[dict],
    *,
    padding_s: float = 0.040,
    nearest_tol_s: float = 0.120,
) -> list[dict]:
    """
    Assign a 'speaker' key to each word dict.

    Fallback chain per word:
    1. Max-overlap diarization segment (word interval ± padding_s)
    2. Midpoint containment
    3. Nearest diarization turn within nearest_tol_s
    4. Previous word's speaker (gap ≤ 200 ms)
    5. "UNKNOWN"
    Returns new list of word dicts with 'speaker' added.
    """
    ...

def smooth_micro_turns(words: list[dict], max_run: int = 1) -> list[dict]:
    """
    Relabel isolated micro-turns (≤ max_run words) sandwiched between the
    same speaker on both sides to that surrounding speaker.
    """
    ...

def build_diarized_segments(words: list[dict]) -> list[dict]:
    """
    Group speaker-labelled words into contiguous segments.
    Returns [{"start", "end", "text", "speaker", "words": [...]}, ...]
    """
    ...

# ─── Response formatters ─────────────────────────────────────────────────────

def fmt_json(text: str) -> dict:
    return {"text": text}

def fmt_text(text: str) -> str:
    return text

def fmt_verbose_json(segments: list[dict], words: list[dict], duration: float) -> dict:
    """OpenAI verbose_json format with segment + word timestamps."""
    ...

def fmt_srt(segments: list[dict]) -> str:
    """Convert segments to SubRip (.srt) format."""
    ...

def fmt_vtt(segments: list[dict]) -> str:
    """Convert segments to WebVTT (.vtt) format."""
    ...

def fmt_diarized_json(segments: list[dict], words: list[dict], duration: float) -> dict:
    """verbose_json extended with 'speaker' on every segment and word."""
    ...

# ─── Endpoint ────────────────────────────────────────────────────────────────

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),          # ignored — Parakeet is always used
    language: str | None = Form(None),       # ignored — Parakeet auto-detects
    response_format: str = Form("json"),
    diarization: bool = Form(False),
    expected_speakers: int | None = Form(None),  # hint, currently unused by Sortformer
):
    """
    OpenAI-compatible transcription endpoint.

    1. Save upload to a temp file.
    2. Decode audio to 16 kHz mono float32 PCM.
    3. Run Parakeet ASR → text + word timestamps.
    4. If diarization=true:
         a. Run Sortformer diarization (sequential, after ASR, to avoid Metal deadlock).
         b. Merge speaker labels onto ASR words.
         c. Build diarized segments.
    5. Format and return the response.
    """
    ...

@app.get("/health")
async def health():
    return {"status": "ok", "backend": "mlx_parakeet"}
```

---

### `README.md`

Include:

1. **Prerequisites**: Apple Silicon Mac, macOS 12+, `uv` (`brew install uv`),
   `ffmpeg` (`brew install ffmpeg`).

2. **Install**:
   ```bash
   cd parakeet_server
   uv sync
   ```

3. **Run**:
   ```bash
   uv run uvicorn server:app --host 127.0.0.1 --port 5167 --workers 1
   ```

4. **Test** with curl:
   ```bash
   curl -X POST http://127.0.0.1:5167/v1/audio/transcriptions \
     -F "file=@/path/to/audio.mp3" \
     -F "response_format=diarized_json" \
     -F "diarization=true"
   ```

5. **Test** with Open-WebUI or any OpenAI-compatible client by setting
   `base_url = "http://127.0.0.1:5167"` and `api_key = "none"`.

6. **Expected `diarized_json` output**:
   ```json
   {
     "task": "transcribe",
     "language": "en",
     "duration": 12.34,
     "text": "Good morning everyone. Let us get started.",
     "segments": [
       {
         "start": 0.32,
         "end": 4.18,
         "text": "Good morning everyone.",
         "speaker": "SPEAKER_00",
         "words": [
           {"word": "Good",     "start": 0.32, "end": 0.58, "probability": 0.99, "speaker": "SPEAKER_00"},
           {"word": "morning",  "start": 0.61, "end": 1.10, "probability": 0.98, "speaker": "SPEAKER_00"},
           {"word": "everyone.","start": 1.20, "end": 1.88, "probability": 0.97, "speaker": "SPEAKER_00"}
         ]
       },
       {
         "start": 5.01,
         "end": 8.92,
         "text": "Let us get started.",
         "speaker": "SPEAKER_01",
         "words": [...]
       }
     ]
   }
   ```

---

### Implementation notes for the assistant

- **Do not import** `torch`, `pyannote`, `whisper`, `faster-whisper`, or any
  CUDA library.  This server is Metal/MLX only.
- The `diarization` form field is a string `"true"` / `"false"` when sent via
  `multipart/form-data`; FastAPI's `Form(False)` with type `bool` handles this
  transparently.
- `Sortformer.generate_stream()` returns an iterator of result objects.  Each
  result has a `.segments` attribute that is a list of segment objects with
  `.start`, `.end`, `.speaker` attributes (floats and string).
- `parakeet-mlx`'s `model.transcribe(path)` returns an `AlignedResult` with a
  `.sentences` list.  Each sentence has `.text`, `.start`, `.end`, and `.tokens`
  (list of `AlignedToken` with `.text`, `.start`, `.end`, `.confidence`).
- Use `asyncio.to_thread()` to run the CPU/GPU-bound inference off the event loop.
- Keep both models as module-level globals; do not re-load them per request.
- After every `mlx` inference call, run:
  ```python
  import mlx.core as mx
  mx.clear_cache()
  ```
  to release intermediate Metal buffers.
- A single uvicorn worker is correct — the Metal GPU is single-threaded and the
  job tracker in the full TranscriptionSuite enforces single-job concurrency.

---end prompt---

---

## How This Relates to TranscriptionSuite

The prompt above describes a **simplified extraction** of the Metal diarization
stack that already exists in TranscriptionSuite.  The key modules to read for
reference implementations are:

| Task | Source file |
|------|-------------|
| Parakeet ASR | `server/backend/core/stt/backends/mlx_parakeet_backend.py` |
| Sortformer diarization | `server/backend/core/sortformer_engine.py` |
| Sequential orchestration | `server/backend/core/parallel_diarize.py` → `transcribe_then_diarize` |
| Speaker merge | `server/backend/core/speaker_merge.py` |
| OpenAI endpoint | `server/backend/api/routes/openai_audio.py` |
| Response formatters | `server/backend/core/formatters.py` |
| Audio decoding | `server/backend/core/audio_utils.py` → `load_audio` |
| Startup / app | `server/backend/api/main.py` |

For a full standalone server that reuses TranscriptionSuite's existing code,
see `scripts/start-backend-macos-metal.sh` which starts the complete backend
(with database, Live Mode, notebook, and all routes) directly from either the
installed DMG bundle or the source repository.
