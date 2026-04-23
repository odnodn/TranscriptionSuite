# Parakeet Server

A minimal, self-contained OpenAI-compatible audio transcription server powered by
[parakeet-mlx](https://github.com/senstella/parakeet-mlx) (ASR) and
[mlx-audio](https://github.com/Blaizzy/mlx-audio) Sortformer (speaker diarization).
Runs entirely on Apple Silicon via the Metal/MLX stack — no CUDA, no HuggingFace
token, no Docker required.

---

## Prerequisites

| Requirement | Install |
|-------------|---------|
| Apple Silicon Mac (M1/M2/M3/M4) | — |
| macOS 12 Monterey or later | — |
| `uv` package manager | `brew install uv` |
| `ffmpeg` (optional but recommended) | `brew install ffmpeg` |

---

## Install

```bash
cd parakeet_server
uv sync
```

Model weights are downloaded automatically on the first request:
- **ASR**: `mlx-community/parakeet-tdt-0.6b-v3` (~1.2 GB)
- **Diarization**: `mlx-community/diar_sortformer_4spk-v1-fp32` (loaded only when `diarization=true`)

---

## Run

```bash
uv run uvicorn server:app --host 127.0.0.1 --port 5167 --workers 1
```

> **Important**: use exactly `--workers 1`.  The Metal GPU is single-threaded and
> both models share the same command queue — multiple workers would deadlock.

Interactive API docs are available at <http://127.0.0.1:5167/docs>.

---

## Test with curl

### Plain JSON (default)
```bash
curl -X POST http://127.0.0.1:5167/v1/audio/transcriptions \
  -F "file=@/path/to/audio.mp3"
```

### Plain text
```bash
curl -X POST http://127.0.0.1:5167/v1/audio/transcriptions \
  -F "file=@/path/to/audio.mp3" \
  -F "response_format=text"
```

### Diarized JSON (speaker labels)
```bash
curl -X POST http://127.0.0.1:5167/v1/audio/transcriptions \
  -F "file=@/path/to/audio.mp3" \
  -F "response_format=diarized_json" \
  -F "diarization=true"
```

### SRT subtitles
```bash
curl -X POST http://127.0.0.1:5167/v1/audio/transcriptions \
  -F "file=@/path/to/audio.mp3" \
  -F "response_format=srt"
```

### Health check
```bash
curl http://127.0.0.1:5167/health
# {"status":"ok","backend":"mlx_parakeet"}
```

---

## Use with Open-WebUI or any OpenAI-compatible client

Set the STT base URL and a dummy API key:

```
base_url = "http://127.0.0.1:5167"
api_key  = "none"
```

The endpoint `/v1/audio/transcriptions` is drop-in compatible with the
[OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

---

## Supported `response_format` values

| Value | Description |
|-------|-------------|
| `json` | `{"text": "…"}` (default) |
| `text` | Plain text transcript |
| `verbose_json` | Segments + word timestamps (OpenAI format) |
| `srt` | SubRip subtitles |
| `vtt` | WebVTT subtitles |
| `diarized_json` | `verbose_json` extended with `speaker` on every segment/word |

---

## Example `diarized_json` output

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 12.34,
  "text": "Good morning everyone. Let us get started.",
  "segments": [
    {
      "id": 0,
      "start": 0.32,
      "end": 4.18,
      "text": "Good morning everyone.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Good",      "start": 0.32, "end": 0.58, "probability": 0.99, "speaker": "SPEAKER_00"},
        {"word": "morning",   "start": 0.61, "end": 1.10, "probability": 0.98, "speaker": "SPEAKER_00"},
        {"word": "everyone.", "start": 1.20, "end": 1.88, "probability": 0.97, "speaker": "SPEAKER_00"}
      ]
    },
    {
      "id": 1,
      "start": 5.01,
      "end": 8.92,
      "text": "Let us get started.",
      "speaker": "SPEAKER_01",
      "words": [
        {"word": "Let",      "start": 5.01, "end": 5.28, "probability": 0.99, "speaker": "SPEAKER_01"},
        {"word": "us",       "start": 5.31, "end": 5.48, "probability": 0.99, "speaker": "SPEAKER_01"},
        {"word": "get",      "start": 5.52, "end": 5.74, "probability": 0.98, "speaker": "SPEAKER_01"},
        {"word": "started.", "start": 5.80, "end": 8.92, "probability": 0.97, "speaker": "SPEAKER_01"}
      ]
    }
  ]
}
```

---

## Architecture notes

- Both models are loaded lazily on the first request that needs them and kept
  in memory for the lifetime of the process.
- ASR and diarization always run **sequentially** (never in parallel) to prevent
  Metal GPU deadlock when both models compete for the same command queue.
- `mlx.core.mx.clear_cache()` is called after every inference call to release
  intermediate Metal buffers.
- Audio is decoded to 16 kHz mono float32 PCM via `ffmpeg` (all formats) with
  a `soundfile` fallback for WAV/FLAC when `ffmpeg` is unavailable.
- Long audio files (> 120 s) are processed in 120 s chunks with 15 s overlap.
- Sortformer supports up to 4 speakers.
