# Diarization Pipeline — In-Depth Technical Reference

This document traces the complete diarization pipeline in TranscriptionSuite,
starting from an HTTP request to `POST /v1/audio/transcriptions` and ending at
the final speaker-attributed transcript.  Code samples are taken directly from
the repository; line numbers are approximate and may shift as the codebase evolves.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Entry Point — `openai_audio.py`](#2-entry-point--openai_audiopy)
3. [Orchestration — `_run_transcription()`](#3-orchestration--_run_transcription)
   - 3a. [Path 1 — Integrated single-pass (WhisperX / VibeVoice-ASR)](#3a-path-1--integrated-single-pass-whisperx--vibevoice-asr)
   - 3b. [Path 2 — Parallel / sequential transcription + diarization](#3b-path-2--parallel--sequential-transcription--diarization)
   - 3c. [Path 3 — Plain transcription (fallback)](#3c-path-3--plain-transcription-fallback)
4. [STT Engine — `engine.py`](#4-stt-engine--enginepy)
5. [Diarization Engine Selection — `diarization_engine.py`](#5-diarization-engine-selection--diarization_enginepy)
   - 5a. [PyAnnote (`DiarizationEngine`)](#5a-pyannote-diarizationengine)
   - 5b. [Sortformer / Metal (`SortformerEngine`)](#5b-sortformer--metal-sortformerengine)
6. [Parallel Orchestration — `parallel_diarize.py`](#6-parallel-orchestration--parallel_diarizepy)
7. [MLX Parakeet Backend — `mlx_parakeet_backend.py`](#7-mlx-parakeet-backend--mlx_parakeet_backendpy)
8. [Speaker Merge — `speaker_merge.py`](#8-speaker-merge--speaker_mergepy)
9. [Response Formatting](#9-response-formatting)
10. [Apple Silicon (Metal) Specifics](#10-apple-silicon-metal-specifics)
11. [Configuration Reference](#11-configuration-reference)
12. [Data-Flow Diagram](#12-data-flow-diagram)

---

## 1. Overview

TranscriptionSuite combines automatic speech recognition (ASR / STT) with speaker
diarization to answer: *who said what and when?*

Two orthogonal concerns are pipelined together:

| Concern | Output | Engine |
|---------|--------|--------|
| **Transcription** | Text + word timestamps | Parakeet, Whisper, Canary, … |
| **Diarization**  | Speaker segments (start, end, speaker ID) | Sortformer (Metal), PyAnnote |

Their outputs are then **merged** at word granularity: every ASR word is
matched to the diarization segment with the highest temporal overlap, giving
speaker-attributed sentences like:

```json
{
  "speaker": "SPEAKER_00",
  "start": 0.32,
  "end": 4.18,
  "text": "Good morning, everyone."
}
```

---

## 2. Entry Point — `openai_audio.py`

The public surface is the `POST /v1/audio/transcriptions` endpoint defined in
`server/backend/api/routes/openai_audio.py`.  It accepts an uploaded audio file
plus optional form fields:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `diarization` | bool | `false` | Enable speaker labelling |
| `expected_speakers` | int \| null | `null` | Known speaker count hint |
| `parallel_diarization` | bool \| null | `null` | Override the config flag |
| `response_format` | str | `"json"` | Output format |
| `timestamp_granularities[]` | list | `[]` | Include `"word"` for word timestamps |

```python
# server/backend/api/routes/openai_audio.py
@router.post("/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    diarization: bool = Form(False),
    expected_speakers: int | None = Form(None),
    parallel_diarization: bool | None = Form(None),
):
    ...
    result = await _run_transcription(
        request=request,
        tmp_path=tmp_path,
        task="transcribe",
        language=language,
        translation_target_language=None,
        initial_prompt=prompt,
        word_timestamps=client_requested_word_timestamps,
        diarization=diarization,
        expected_speakers=expected_speakers,
        parallel_diarization=parallel_diarization,
    )
```

Before dispatching to `_run_transcription`, the handler:

1. **Validates** `response_format` and `expected_speakers`.
2. **Self-heals** the backend via `model_manager.ensure_transcription_loaded()` — if the
   model was evicted or never loaded, it is reloaded here rather than surfacing a 503.
3. **Acquires a job slot** via `job_tracker.try_start_job()` — only one transcription job
   runs at a time; a second request gets a 429.
4. **Streams the upload** to a temp file so the audio decoder (ffmpeg) can work on
   a complete file.

---

## 3. Orchestration — `_run_transcription()`

`_run_transcription` implements a **three-path dispatch** based on what the active
backend supports and what the request asked for.

```python
async def _run_transcription(
    *,
    request: Request,
    tmp_path: str,
    task: str,
    language: str | None,
    ...
    diarization: bool,
    expected_speakers: int | None,
    parallel_diarization: bool | None,
) -> Any:
```

### 3a. Path 1 — Integrated single-pass (WhisperX / VibeVoice-ASR)

Some backends run ASR and diarization in a single pass, avoiding the need to run
two separate models.  The check is:

```python
backend = getattr(engine, "_backend", None)
use_integrated_diarization = (
    diarization
    and backend is not None
    and type(backend).transcribe_with_diarization is not STTBackend.transcribe_with_diarization
)
```

`STTBackend.transcribe_with_diarization` is a default no-op method.  If a backend
**overrides** it (currently WhisperX and VibeVoice-ASR), the integrated path is taken.

```python
if use_integrated_diarization:
    audio_data, audio_sample_rate = await asyncio.to_thread(
        load_audio, tmp_path, target_sample_rate=preferred_rate
    )
    diar_result = await asyncio.to_thread(
        functools.partial(
            backend.transcribe_with_diarization,
            audio_data,
            audio_sample_rate=audio_sample_rate,
            language=language,
            task=task,
            beam_size=engine.beam_size,
            initial_prompt=initial_prompt or engine.initial_prompt,
            suppress_tokens=engine.suppress_tokens,
            vad_filter=engine.faster_whisper_vad_filter,
            num_speakers=expected_speakers,
        )
    )
    return TranscriptionResult(
        text=" ".join(seg.get("text", "") for seg in diar_result.segments).strip(),
        segments=diar_result.segments,
        words=diar_result.words,
        language=diar_result.language,
        language_probability=diar_result.language_probability,
        duration=len(audio_data) / audio_sample_rate,
        num_speakers=diar_result.num_speakers,
    )
```

If this throws any non-cancellation exception, `diarization = False` is set and
execution falls through to Path 3 (plain transcription).

### 3b. Path 2 — Parallel / sequential transcription + diarization

When diarization is requested and the backend does *not* override
`transcribe_with_diarization`, the code picks between parallel or sequential
execution:

```python
if diarization:
    config = request.app.state.config
    use_parallel = (
        parallel_diarization
        if parallel_diarization is not None
        else config.get("diarization", "parallel", default=True)
    )

    if use_parallel:
        from server.core.parallel_diarize import transcribe_and_diarize as diarize_fn
    else:
        from server.core.parallel_diarize import transcribe_then_diarize as diarize_fn

    result, diar_result = await asyncio.to_thread(
        functools.partial(
            diarize_fn,
            engine=engine,
            model_manager=model_manager,
            file_path=tmp_path,
            language=language,
            task=task,
            word_timestamps=need_word_timestamps,
            expected_speakers=expected_speakers,
            cancellation_check=model_manager.job_tracker.is_cancelled,
        )
    )
```

After obtaining both results, the **speaker merge** step assigns speaker labels to
individual ASR words:

```python
from server.core.speaker_merge import build_speaker_segments

diar_dicts = [seg.to_dict() for seg in diar_result.segments]
merged_segments, merged_words, num_speakers = build_speaker_segments(
    result.words, diar_dicts
)
if merged_segments:
    result.segments = merged_segments
    result.words = merged_words
    result.num_speakers = num_speakers
```

A fallback path handles the case where word timestamps were not produced
(rare, backend-dependent):

```python
elif not result.words and result.segments:
    from server.core.speaker_merge import build_speaker_segments_nowords

    fallback = build_speaker_segments_nowords(result.segments, diar_dicts)
    if fallback:
        speakers = {s["speaker"] for s in fallback} - {"UNKNOWN"}
        result.segments = fallback
        result.num_speakers = len(speakers)
```

### 3c. Path 3 — Plain transcription (fallback)

Used when `diarization=False` or every diarization branch failed gracefully:

```python
return await asyncio.to_thread(
    functools.partial(
        engine.transcribe_file,
        tmp_path,
        language=language,
        task=task,
        translation_target_language=translation_target_language,
        word_timestamps=need_word_timestamps,
        initial_prompt=initial_prompt,
    )
)
```

---

## 4. STT Engine — `engine.py`

`AudioToTextRecorder` in `server/backend/core/stt/engine.py` is the central
transcription facade.  It holds a reference to the active `STTBackend`
(e.g. `MLXParakeetBackend`), manages VAD pre-processing for live mode, and
exposes `transcribe_file()` for file-based jobs.

```
AudioToTextRecorder
  ├── _backend: STTBackend          # active backend (Parakeet, Whisper, …)
  ├── transcribe_file(path, ...)    # file-based transcription → TranscriptionResult
  └── feed_audio(chunk, ...)        # live-mode streaming path (WebSocket)
```

`transcribe_file` routes internally to `_backend.transcribe()`, normalising the
result into a backend-agnostic `TranscriptionResult`:

```python
@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None
    language_probability: float = 0.0
    duration: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)
    words: list[dict[str, Any]] = field(default_factory=list)
    num_speakers: int = 0
```

---

## 5. Diarization Engine Selection — `diarization_engine.py`

`create_diarization_engine(config)` in `server/backend/core/diarization_engine.py`
chooses between the two available engines:

```python
def create_diarization_engine(config: dict[str, Any]) -> DiarizationEngine | Any:
    diar_config = config.get("diarization", {})

    from server.core.sortformer_engine import sortformer_available

    explicit_model = diar_config.get("model")
    max_speakers = diar_config.get("max_speakers")
    resolved_device = _resolve_device(diar_config.get("device", "cuda"))

    use_sortformer = (
        sortformer_available()          # mlx-audio is installed
        and resolved_device in ("mps", "cpu")   # Apple Silicon path
        and not (explicit_model and "pyannote" in explicit_model)
        and (max_speakers is None or max_speakers <= 4)
    )

    if use_sortformer:
        from server.core.sortformer_engine import SortformerEngine
        return SortformerEngine(
            num_speakers=diar_config.get("num_speakers"),
            min_speakers=diar_config.get("min_speakers"),
            max_speakers=max_speakers,
        )

    return DiarizationEngine(
        model=diar_config.get("model", "pyannote/speaker-diarization-community-1"),
        hf_token=diar_config.get("hf_token") or os.environ.get("HF_TOKEN"),
        device=resolved_device,
        ...
    )
```

### 5a. PyAnnote (`DiarizationEngine`)

Used on Linux/Windows (CUDA) or when a PyAnnote model is explicitly configured.

```python
class DiarizationEngine:
    def load(self) -> None:
        self._pipeline = Pipeline.from_pretrained(
            self.model,
            token=self.hf_token,        # HuggingFace token required
        )
        self._pipeline = self._pipeline.to(torch.device(self.device))

    def diarize_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: int | None = None,
    ) -> DiarizationResult:
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = self._pipeline(
            audio_input,
            num_speakers=n_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))
        return DiarizationResult(segments=segments, num_speakers=len(speakers))
```

**Requirements**: `pyannote.audio`, PyTorch, a HuggingFace token, and acceptance of
the model license on HuggingFace.

### 5b. Sortformer / Metal (`SortformerEngine`)

Used automatically on Apple Silicon when `mlx-audio` is installed and no explicit
PyAnnote model is configured.  No HuggingFace token is required.

```python
class SortformerEngine:
    # Default: mlx-community/diar_sortformer_4spk-v1-fp32  (up to 4 speakers)

    def load(self) -> None:
        self._model = _load_sortformer(self.model_name)  # mlx_audio.vad.load()

    def diarize_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: int | None = None,
    ) -> DiarizationResult:
        # Write to temp WAV (Sortformer expects a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, sample_rate)
            tmp_path = tmp.name

        # Streaming inference: small chunks to bound Metal memory.
        all_segments_raw = []
        for result in self._model.generate_stream(
            tmp_path,
            chunk_duration=chunk_duration_s,   # default 5.0 s
            threshold=self.threshold,           # default 0.5
        ):
            all_segments_raw.extend(result.segments)

        segments = [
            DiarizationSegment(
                start=float(seg.start),
                end=float(seg.end),
                speaker=str(seg.speaker),
            )
            for seg in all_segments_raw
        ]
        return DiarizationResult(segments=segments, num_speakers=len(speakers))
```

Streaming inference (`generate_stream`) processes the audio in 5-second chunks by
default.  This keeps Metal GPU memory bounded because Sortformer's attention is
quadratic with input length — processing a 60-second file at once would use 144×
the memory of a 5-second chunk.  Speaker context is carried across chunks so
accuracy is maintained.

---

## 6. Parallel Orchestration — `parallel_diarize.py`

`server/backend/core/parallel_diarize.py` provides two orchestration modes.

### `transcribe_and_diarize` (parallel, default)

Runs ASR and diarization **concurrently** in a two-thread pool.  Ideal for
CUDA backends where transcription and PyAnnote diarization release the GIL
during their respective GPU computations.

```python
with ThreadPoolExecutor(max_workers=2) as pool:
    transcribe_future = pool.submit(_do_transcribe)
    diarize_future   = pool.submit(_do_diarize)

    result     = transcribe_future.result()   # critical path
    diar_result = diarize_future.result()
```

**Important exception**: On Apple Silicon, running MLX Whisper (ASR) and
Sortformer (diarization) in two threads simultaneously **deadlocks** the Metal
GPU because both compete for the same command queue.  The code detects this
and falls back to the sequential mode automatically:

```python
from server.core.sortformer_engine import SortformerEngine

if isinstance(diar_engine, SortformerEngine):
    logger.info(
        "Sortformer + MLX detected — switching to sequential mode to avoid Metal deadlock"
    )
    return transcribe_then_diarize(...)
```

### `transcribe_then_diarize` (sequential)

Runs ASR first, **unloads** the ASR model to free GPU/Metal memory, then runs
diarization.  Used when VRAM is tight (<16 GB) or when Sortformer is the
diarization engine.

```python
# Phase 1 — Transcribe
result = engine.transcribe_file(file_path, ...)
model_manager.unload_transcription_model()   # frees ~1–10 GB VRAM

# Phase 2 — Diarize
model_manager.load_diarization_model()
diar_result = diar_engine.diarize_audio(audio_data, audio_sample_rate)
return result, diar_result
```

The `finally` block always restores the pre-job model state: diarization model
unloaded, ASR model reloaded.

---

## 7. MLX Parakeet Backend — `mlx_parakeet_backend.py`

`MLXParakeetBackend` wraps the `parakeet-mlx` library for Metal-accelerated
Parakeet-TDT inference on Apple Silicon.

**Model**: `mlx-community/parakeet-tdt-0.6b-v3` (default) — 660K hours of
training data, 25 European languages, native punctuation and capitalisation.

```python
class MLXParakeetBackend(STTBackend):
    def load(self, model_name: str, device: str, **kwargs: Any) -> None:
        from parakeet_mlx import from_pretrained
        self._model = from_pretrained(model_name)

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        audio_sample_rate: int = 16000,
        word_timestamps: bool = True,
        ...
    ) -> tuple[list[BackendSegment], BackendTranscriptionInfo]:
        # Toggle attention model based on audio duration
        if use_local and audio_duration_s > threshold_s:
            self._model.encoder.set_attention_model("rel_pos_local_attn", tuple(window))
        else:
            self._model.encoder.set_attention_model("rel_pos")

        # Optionally chunk long audio
        result = self._model.transcribe(tmp_path, **transcribe_kwargs)

        # Convert sentences → BackendSegment list with word timestamps
        segments = []
        for sentence in result.sentences:
            words = _tokens_to_words(sentence.tokens)
            segments.append(BackendSegment(
                text=str(sentence.text).strip(),
                start=float(sentence.start),
                end=float(sentence.end),
                words=words,
            ))
        return segments, info
```

**Word timestamps** are produced by `_tokens_to_words()`, which groups SentencePiece
tokens (each with `start`, `end`, `confidence`) into word-level dicts:

```python
def _tokens_to_words(tokens: list[Any]) -> list[dict[str, Any]]:
    # Tokens whose .text starts with a space mark a new word boundary.
    # Example: ["▁Good", "▁morning", ",", "▁everyone", "."]
    #          → [{"word": "Good", ...}, {"word": "morning,", ...}, {"word": "everyone.", ...}]
    ...
    return [{"word": text, "start": start, "end": end, "probability": conf}, ...]
```

These word timestamps are what the speaker merge step uses to assign a speaker
label to each spoken word.

---

## 8. Speaker Merge — `speaker_merge.py`

`build_speaker_segments` in `server/backend/core/speaker_merge.py` is the final
assembly step.  It takes ASR words + diarization segments and produces
speaker-attributed transcript segments.

### Step 1 — Word-level speaker assignment

`assign_speakers_to_words` matches each ASR word to the diarization segment with
the highest temporal overlap using a four-level fallback chain:

```
1. Max-overlap diarization segment (word interval padded ±40 ms)
2. Midpoint containment (word midpoint inside a diarization segment)
3. Nearest diarization turn within 120 ms
4. Previous word's speaker (gap ≤ 200 ms)
5. "UNKNOWN"
```

```python
for w in words:
    padded_start = w_start - word_padding_s   # ±40 ms
    padded_end   = w_end   + word_padding_s

    best_speaker = None
    best_overlap = 0.0
    for seg in diar:
        overlap = max(0.0, min(padded_end, seg_end) - max(padded_start, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]
    ...
```

### Step 2 — Micro-turn smoothing

`smooth_micro_turns` removes speaker ping-pong artefacts where isolated short
function words (e.g. "uh", "yeah") are erroneously assigned to the wrong speaker:

```python
# If a run of ≤1 word is sandwiched between the same speaker on both sides,
# relabel it to the surrounding speaker.
for r_idx in range(1, len(runs) - 1):
    spk, start_idx, length = runs[r_idx]
    if length > max_run_length:
        continue
    prev_spk = runs[r_idx - 1][0]
    next_spk = runs[r_idx + 1][0]
    if prev_spk == next_spk and prev_spk != spk:
        for wi in range(start_idx, start_idx + length):
            relabel[wi] = prev_spk
```

### Step 3 — Segment grouping

Speaker-labelled words are grouped into contiguous segments where the speaker does
not change.  Each output segment looks like:

```json
{
  "start": 0.32,
  "end": 4.18,
  "text": "Good morning, everyone.",
  "speaker": "SPEAKER_00",
  "words": [
    {"word": "Good",     "start": 0.32, "end": 0.58, "speaker": "SPEAKER_00"},
    {"word": "morning,", "start": 0.61, "end": 1.10, "speaker": "SPEAKER_00"},
    {"word": "everyone.","start": 1.20, "end": 1.88, "speaker": "SPEAKER_00"}
  ]
}
```

---

## 9. Response Formatting

The result is serialised by `_build_response` in `openai_audio.py` using the
formatters in `server/backend/core/formatters.py`:

| `response_format` | Content-Type | Description |
|-------------------|-------------|-------------|
| `json` | application/json | `{"text": "..."}` |
| `text` | text/plain | Raw transcript text |
| `verbose_json` | application/json | Full segments + word timestamps |
| `srt` | text/plain | SubRip subtitle format |
| `vtt` | text/plain | WebVTT subtitle format |
| `diarized_json` | application/json | `verbose_json` + speaker labels |

When diarization is enabled, `diarized_json` is the richest format — it includes
per-segment and per-word speaker labels alongside timestamps.

---

## 10. Apple Silicon (Metal) Specifics

On Apple Silicon, the full diarization pipeline operates without any NVIDIA GPU or
HuggingFace token:

| Component | macOS / Metal | Linux / CUDA |
|-----------|--------------|--------------|
| ASR engine | MLX Parakeet (`parakeet-mlx`) | NeMo Parakeet / Whisper |
| Diarization | Sortformer (`mlx-audio`) | PyAnnote |
| Attention | Local attention for >2 min | Full attention (CUDA parallelism) |
| Parallelism | **Sequential** (Metal deadlock prevention) | Parallel threads |
| HF token | Not required | Required for PyAnnote |

**Memory management**: MLX buffers are released between phases using
`mlx.core.mx.clear_cache()`.  The buffer pool is capped at 1 GB by default
(`mlx.metal_cache_limit_mb: 1024` in `config.yaml`).

---

## 11. Configuration Reference

The relevant `config.yaml` sections for diarization are:

```yaml
diarization:
    # PyAnnote model (null → Sortformer on Apple Silicon)
    model: null
    hf_token: null               # HuggingFace token (PyAnnote only)
    device: "auto"               # auto → MPS on Apple Silicon
    min_speakers: null           # null → auto-detect
    max_speakers: null           # null → auto-detect (Sortformer caps at 4)
    parallel: false              # false → sequential (recommended for Metal)

sortformer:
    chunk_duration_s: 5.0        # streaming chunk size (seconds)

parakeet:
    mlx_local_attention: true
    mlx_local_attention_threshold_s: 120   # use local attn for audio > 2 min
    mlx_chunk_duration_s: 120              # split audio into 2-min chunks
    mlx_overlap_duration_s: 15            # 15 s overlap between chunks

mlx:
    metal_cache_limit_mb: 1024   # Metal GPU buffer pool cap
```

---

## 12. Data-Flow Diagram

```
HTTP POST /v1/audio/transcriptions
    │
    ▼
create_transcription()              openai_audio.py
    │  validate params
    │  acquire job slot
    │  save upload to temp file
    ▼
_run_transcription()                openai_audio.py
    │
    ├─► [backend.transcribe_with_diarization()]  ─── PATH 1 (WhisperX / VibeVoice)
    │       single-pass ASR + diarization
    │
    ├─► transcribe_and_diarize()    parallel_diarize.py  ─── PATH 2a (parallel)
    │   │  or
    │   └─► transcribe_then_diarize()             ─── PATH 2b (sequential, Metal)
    │           │
    │           ├─► engine.transcribe_file()      stt/engine.py
    │           │       │
    │           │       └─► MLXParakeetBackend.transcribe()   mlx_parakeet_backend.py
    │           │               parakeet_mlx.from_pretrained()
    │           │               → AlignedResult(sentences, tokens)
    │           │               → BackendSegment list + word timestamps
    │           │
    │           └─► diar_engine.diarize_audio()
    │                   │
    │                   ├─► SortformerEngine     (Apple Silicon)
    │                   │       mlx_audio.vad.load()
    │                   │       generate_stream() → DiarizationResult
    │                   │
    │                   └─► DiarizationEngine    (CUDA / PyAnnote)
    │                           pyannote.Pipeline()
    │                           itertracks() → DiarizationResult
    │
    ├─► build_speaker_segments()    speaker_merge.py
    │       assign_speakers_to_words()
    │       smooth_micro_turns()
    │       → merged segments + words with speaker labels
    │
    └─► engine.transcribe_file()    stt/engine.py  ─── PATH 3 (fallback)
            plain transcription, no speaker labels

    │
    ▼
_build_response()                   openai_audio.py
    │  format_json / format_verbose_json / format_diarized_json / …
    ▼
HTTP Response (JSON / text / SRT / VTT)
```
