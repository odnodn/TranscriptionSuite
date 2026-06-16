"""
Minimal standalone Parakeet + Sortformer OpenAI-compatible transcription server.
Apple Silicon (Metal) only.  No HuggingFace token required.

Usage:
    uv run uvicorn server:app --host 127.0.0.1 --port 5167 --workers 1
"""

from __future__ import annotations

import asyncio
import io
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

_asr_model = None   # parakeet-mlx model
_diar_model = None  # mlx-audio Sortformer model

# ─── Audio loading ───────────────────────────────────────────────────────────

_FFMPEG_AVAILABLE: bool | None = None  # None = not yet checked


def _check_ffmpeg() -> bool:
    """Return True if ffmpeg-python and the ffmpeg binary are usable."""
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is not None:
        return _FFMPEG_AVAILABLE
    try:
        import ffmpeg  # noqa: F401 — import-only check

        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        _FFMPEG_AVAILABLE = result.returncode == 0
    except Exception:
        _FFMPEG_AVAILABLE = False
    return _FFMPEG_AVAILABLE


def load_audio_ffmpeg(path: str, target_sr: int = 16000) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 PCM using ffmpeg."""
    import ffmpeg

    try:
        out, _ = (
            ffmpeg.input(path)
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=target_sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        raise RuntimeError(
            f"ffmpeg failed to decode {path}: {exc.stderr.decode(errors='replace')}"
        ) from exc

    audio = np.frombuffer(out, dtype=np.float32)
    return audio


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio from *path* as 16 kHz mono float32 PCM.

    Tries ffmpeg first (supports mp3, m4a, ogg, …).  Falls back to soundfile
    for WAV/FLAC when ffmpeg is unavailable.
    """
    if _check_ffmpeg():
        return load_audio_ffmpeg(path, target_sr)

    # soundfile fallback — handles WAV and FLAC only
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    # always_2d=True → shape is always (samples, channels)
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    # Mix to mono: average across the channel dimension
    audio = audio.mean(axis=1)
    # Resample if necessary
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


# ─── ASR (MLX Parakeet) ──────────────────────────────────────────────────────

_ASR_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
_CHUNK_SECONDS = 120
_OVERLAP_SECONDS = 15


def _get_asr_model():
    """Lazy-load the Parakeet model on first call."""
    global _asr_model
    if _asr_model is None:
        from parakeet_mlx import from_pretrained

        logger.info("Loading Parakeet ASR model %s …", _ASR_MODEL_ID)
        _asr_model = from_pretrained(_ASR_MODEL_ID)
        logger.info("Parakeet ASR model loaded.")
    return _asr_model


def _tokens_to_words(tokens) -> list[dict]:
    """
    Convert parakeet-mlx AlignedToken objects to word-level dicts.

    Tokens whose .text starts with a space (or whose index is 0) mark a new
    word boundary.  We accumulate subword pieces and emit a word dict whenever
    the next boundary token arrives.
    """
    words: list[dict] = []
    current_text = ""
    current_start: float | None = None
    current_end: float = 0.0
    current_conf: list[float] = []

    for tok in tokens:
        text: str = tok.text
        start: float = float(tok.start)
        end: float = float(tok.end)
        conf: float = float(tok.confidence) if hasattr(tok, "confidence") else 1.0

        is_word_start = text.startswith(" ") or not current_text

        if is_word_start and current_text:
            # Flush accumulated word
            words.append(
                {
                    "word": current_text.strip(),
                    "start": current_start,
                    "end": current_end,
                    "probability": float(np.mean(current_conf)) if current_conf else 1.0,
                }
            )
            current_text = ""
            current_start = None
            current_conf = []

        if current_start is None:
            current_start = start
        current_text += text
        current_end = end
        current_conf.append(conf)

    # Flush final word
    if current_text.strip():
        words.append(
            {
                "word": current_text.strip(),
                "start": current_start,
                "end": current_end,
                "probability": float(np.mean(current_conf)) if current_conf else 1.0,
            }
        )

    return words


def _merge_overlapping_results(results: list[dict], overlap_s: float) -> dict:
    """
    Merge multiple chunked transcription results, de-duplicating words that
    fall inside the overlap region.
    """
    if not results:
        return {"text": "", "segments": [], "words": []}
    if len(results) == 1:
        return results[0]

    merged_words: list[dict] = []
    merged_segments: list[dict] = []

    for i, res in enumerate(results):
        chunk_words = res["words"]
        chunk_segments = res["segments"]

        if i == 0:
            # Keep everything from the first chunk
            merged_words.extend(chunk_words)
            merged_segments.extend(chunk_segments)
        else:
            # The previous chunk ends at offset (i * chunk_s - overlap_s)
            # Words from this chunk that start before or at the previous chunk end
            # are duplicates — skip them.
            cutoff = merged_words[-1]["end"] if merged_words else 0.0
            for w in chunk_words:
                if w["start"] > cutoff:
                    merged_words.append(w)
            for seg in chunk_segments:
                if seg["start"] > cutoff:
                    merged_segments.append(seg)

    merged_text = " ".join(w["word"] for w in merged_words)
    return {"text": merged_text, "segments": merged_segments, "words": merged_words}


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Run Parakeet-TDT inference and return:
        {"text": str, "segments": list[dict], "words": list[dict]}

    Each word dict:    {"word": str, "start": float, "end": float, "probability": float}
    Each segment dict: {"text": str, "start": float, "end": float, "words": list[dict]}

    Strategy:
    - For audio ≤ 120 s: single inference pass.
    - For audio > 120 s: chunk into 120 s segments with 15 s overlap, then merge.
    - Always call mx.clear_cache() after inference.
    """
    import mlx.core as mx

    model = _get_asr_model()
    duration_s = len(audio) / sample_rate

    try:
        if duration_s <= _CHUNK_SECONDS:
            return _transcribe_chunk(model, audio, sample_rate, offset=0.0)

        # Long-form: chunk with overlap
        chunk_samples = _CHUNK_SECONDS * sample_rate
        step_samples = (_CHUNK_SECONDS - _OVERLAP_SECONDS) * sample_rate
        results: list[dict] = []
        start_sample = 0
        while start_sample < len(audio):
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]
            offset_s = start_sample / sample_rate
            chunk_result = _transcribe_chunk(model, chunk, sample_rate, offset=offset_s)
            results.append(chunk_result)
            if end_sample >= len(audio):
                break
            start_sample += step_samples

        return _merge_overlapping_results(results, _OVERLAP_SECONDS)
    finally:
        mx.clear_cache()


def _transcribe_chunk(model, audio: np.ndarray, sample_rate: int, offset: float = 0.0) -> dict:
    """
    Transcribe a single audio chunk and return normalised result dict.
    Applies a time *offset* to all timestamps (used when chunking long audio).
    """
    import mlx.core as mx
    import soundfile as sf

    # Parakeet expects a file path — write to a temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")

        # Use local attention for long chunks to bound Metal memory.
        # The flag is silently skipped if the installed version doesn't support it.
        duration_s = len(audio) / sample_rate
        kwargs: dict[str, Any] = {}
        if duration_s > 60:
            kwargs["local_attention"] = True

        result = model.transcribe(tmp_path, **kwargs)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Normalise AlignedResult → our internal dict
    segments: list[dict] = []
    all_words: list[dict] = []

    for sentence in result.sentences:
        words = _tokens_to_words(sentence.tokens)
        # Apply chunk offset
        for w in words:
            w["start"] = round(w["start"] + offset, 4)
            w["end"] = round(w["end"] + offset, 4)
        all_words.extend(words)

        seg_start = round(float(sentence.start) + offset, 4)
        seg_end = round(float(sentence.end) + offset, 4)
        segments.append(
            {
                "text": sentence.text.strip(),
                "start": seg_start,
                "end": seg_end,
                "words": words,
            }
        )

    full_text = " ".join(s["text"] for s in segments)
    return {"text": full_text, "segments": segments, "words": all_words}


# ─── Diarization (MLX Sortformer) ────────────────────────────────────────────

_DIAR_MODEL_ID = "mlx-community/diar_sortformer_4spk-v1-fp32"


def _get_diar_model():
    """Lazy-load the Sortformer model on first call."""
    global _diar_model
    if _diar_model is None:
        from mlx_audio.vad import load as load_sortformer

        logger.info("Loading Sortformer diarization model %s …", _DIAR_MODEL_ID)
        _diar_model = load_sortformer(_DIAR_MODEL_ID)
        logger.info("Sortformer diarization model loaded.")
    return _diar_model


def diarize(audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
    """
    Run Sortformer diarization and return a list of speaker segments:
        [{"start": float, "end": float, "speaker": str}, ...]

    Writes audio to a temp WAV (Sortformer expects a file path) and uses
    streaming inference (generate_stream) with 5-second chunks.
    Calls mx.clear_cache() after inference.
    """
    import mlx.core as mx
    import soundfile as sf

    model = _get_diar_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")

        segments: list[dict] = []
        for result in model.generate_stream(tmp_path, chunk_duration=5.0, threshold=0.5):
            for seg in result.segments:
                segments.append(
                    {
                        "start": round(float(seg.start), 4),
                        "end": round(float(seg.end), 4),
                        "speaker": str(seg.speaker),
                    }
                )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        mx.clear_cache()

    return segments


# ─── Speaker merge ───────────────────────────────────────────────────────────


def assign_speakers(
    words: list[dict],
    diar_segments: list[dict],
    *,
    padding_s: float = 0.040,
    nearest_tol_s: float = 0.120,
    gap_inherit_s: float = 0.200,
) -> list[dict]:
    """
    Assign a 'speaker' key to each word dict.

    Fallback chain per word:
    1. Max-overlap diarization segment (word interval ± padding_s)
    2. Midpoint containment
    3. Nearest diarization turn within nearest_tol_s
    4. Previous word's speaker if gap ≤ gap_inherit_s
    5. "UNKNOWN"

    Returns a new list of word dicts with 'speaker' added.
    """
    if not diar_segments:
        return [{**w, "speaker": "UNKNOWN"} for w in words]

    result: list[dict] = []
    prev_speaker: str | None = None
    prev_end: float = 0.0

    for word in words:
        w_start = word["start"] - padding_s
        w_end = word["end"] + padding_s
        w_mid = (word["start"] + word["end"]) / 2.0

        # 1. Max-overlap
        best_overlap = 0.0
        best_speaker: str | None = None
        for seg in diar_segments:
            overlap = min(w_end, seg["end"]) - max(w_start, seg["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]

        if best_speaker is not None:
            speaker = best_speaker
        else:
            # 2. Midpoint containment
            mid_speaker: str | None = None
            for seg in diar_segments:
                if seg["start"] <= w_mid <= seg["end"]:
                    mid_speaker = seg["speaker"]
                    break
            if mid_speaker is not None:
                speaker = mid_speaker
            else:
                # 3. Nearest turn within tolerance
                nearest_dist = float("inf")
                nearest_speaker: str | None = None
                for seg in diar_segments:
                    dist = min(
                        abs(word["start"] - seg["end"]),
                        abs(word["end"] - seg["start"]),
                    )
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_speaker = seg["speaker"]
                if nearest_dist <= nearest_tol_s and nearest_speaker is not None:
                    speaker = nearest_speaker
                elif (
                    prev_speaker is not None
                    and (word["start"] - prev_end) <= gap_inherit_s
                ):
                    # 4. Inherit from previous word
                    speaker = prev_speaker
                else:
                    speaker = "UNKNOWN"

        result.append({**word, "speaker": speaker})
        prev_speaker = speaker
        prev_end = word["end"]

    return result


def smooth_micro_turns(words: list[dict], max_run: int = 1) -> list[dict]:
    """
    Relabel isolated micro-turns (runs of ≤ max_run words with the same speaker)
    that are sandwiched between the same speaker on both sides.
    """
    if len(words) < 3:
        return list(words)

    words = list(words)  # copy
    n = len(words)

    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            current = words[i]["speaker"]
            # Find the end of this run
            j = i
            while j < n and words[j]["speaker"] == current:
                j += 1
            run_len = j - i

            if run_len <= max_run and i > 0 and j < n:
                left = words[i - 1]["speaker"]
                right = words[j]["speaker"]
                if left == right and left != current:
                    for k in range(i, j):
                        words[k] = {**words[k], "speaker": left}
                    changed = True
            i = j

    return words


def build_diarized_segments(words: list[dict]) -> list[dict]:
    """
    Group speaker-labelled words into contiguous segments.
    Returns [{"start", "end", "text", "speaker", "words": [...]}, ...]
    """
    if not words:
        return []

    segments: list[dict] = []
    current_speaker = words[0]["speaker"]
    current_words: list[dict] = [words[0]]

    for word in words[1:]:
        if word["speaker"] == current_speaker:
            current_words.append(word)
        else:
            segments.append(_flush_segment(current_speaker, current_words))
            current_speaker = word["speaker"]
            current_words = [word]

    segments.append(_flush_segment(current_speaker, current_words))
    return segments


def _flush_segment(speaker: str, words: list[dict]) -> dict:
    text = " ".join(w["word"] for w in words)
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "speaker": speaker,
        "words": words,
    }


# ─── Response formatters ─────────────────────────────────────────────────────


def fmt_json(text: str) -> dict:
    return {"text": text}


def fmt_text(text: str) -> str:
    return text


def fmt_verbose_json(
    segments: list[dict], words: list[dict], duration: float, task: str = "transcribe"
) -> dict:
    """OpenAI verbose_json format with segment + word timestamps."""
    return {
        "task": task,
        "language": "en",
        "duration": round(duration, 3),
        "text": " ".join(s["text"] for s in segments),
        "words": [
            {
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "probability": w.get("probability", 1.0),
            }
            for w in words
        ],
        "segments": [
            {
                "id": idx,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "words": [
                    {
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "probability": w.get("probability", 1.0),
                    }
                    for w in seg.get("words", [])
                ],
            }
            for idx, seg in enumerate(segments)
        ],
    }


def _ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _ts_vtt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm (VTT) timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def fmt_srt(segments: list[dict]) -> str:
    """Convert segments to SubRip (.srt) format."""
    lines: list[str] = []
    for idx, seg in enumerate(segments, 1):
        lines.append(str(idx))
        lines.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
        speaker_prefix = f"[{seg['speaker']}] " if "speaker" in seg else ""
        lines.append(f"{speaker_prefix}{seg['text']}")
        lines.append("")
    return "\n".join(lines)


def fmt_vtt(segments: list[dict]) -> str:
    """Convert segments to WebVTT (.vtt) format."""
    lines = ["WEBVTT", ""]
    for idx, seg in enumerate(segments, 1):
        lines.append(str(idx))
        lines.append(f"{_ts_vtt(seg['start'])} --> {_ts_vtt(seg['end'])}")
        speaker_prefix = f"<v {seg['speaker']}>" if "speaker" in seg else ""
        lines.append(f"{speaker_prefix}{seg['text']}")
        lines.append("")
    return "\n".join(lines)


def fmt_diarized_json(
    segments: list[dict], words: list[dict], duration: float, task: str = "transcribe"
) -> dict:
    """verbose_json extended with 'speaker' on every segment and word."""
    base = fmt_verbose_json(segments, words, duration, task=task)
    # Overlay speaker information
    base["segments"] = [
        {
            **seg,
            "speaker": segments[idx].get("speaker", "UNKNOWN"),
            "words": [
                {
                    **w,
                    "speaker": words_in_seg[i].get("speaker", "UNKNOWN"),
                }
                for i, w in enumerate(seg["words"])
            ],
        }
        for idx, (seg, words_in_seg) in enumerate(
            zip(base["segments"], [s.get("words", []) for s in segments])
        )
    ]
    # Overlay speaker on top-level words list
    base["words"] = [
        {**w, "speaker": words[i].get("speaker", "UNKNOWN")}
        for i, w in enumerate(base["words"])
    ]
    return base


# ─── Endpoint ────────────────────────────────────────────────────────────────

_VALID_FORMATS = {"json", "text", "verbose_json", "srt", "vtt", "diarized_json"}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),           # ignored — Parakeet is always used
    language: str | None = Form(None),        # ignored — Parakeet auto-detects
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
    if response_format not in _VALID_FORMATS:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": (
                        f"Invalid response_format '{response_format}'. "
                        f"Must be one of: {', '.join(sorted(_VALID_FORMATS))}"
                    ),
                    "type": "invalid_request_error",
                    "param": "response_format",
                    "code": None,
                }
            },
        )

    # 1. Persist upload to a temp file.
    # Validate the extension against a known-safe allowlist to prevent path
    # traversal or shell-injection via a crafted filename.
    _ALLOWED_AUDIO_EXTS = {
        ".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".opus",
        ".aac", ".wma", ".webm", ".aiff", ".aif",
    }
    raw_suffix = Path(file.filename or "audio.bin").suffix.lower()
    suffix = raw_suffix if raw_suffix in _ALLOWED_AUDIO_EXTS else ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        # 2. Decode audio
        audio = await asyncio.to_thread(load_audio, tmp_path)
        duration_s = len(audio) / 16000

        # 3. ASR
        asr_result = await asyncio.to_thread(transcribe, audio)
        asr_words = asr_result["words"]
        asr_segments = asr_result["segments"]

        # 4. Diarization (sequential — must not run concurrently with ASR on Metal)
        final_segments = asr_segments
        final_words = asr_words

        if diarization:
            try:
                diar_segments = await asyncio.to_thread(diarize, audio)
                labelled_words = assign_speakers(asr_words, diar_segments)
                labelled_words = smooth_micro_turns(labelled_words)
                final_words = labelled_words
                final_segments = build_diarized_segments(labelled_words)
            except Exception:
                logger.exception(
                    "Diarization failed — falling back to plain transcript"
                )
                # Non-fatal: return plain transcript

        # 5. Format response
        text = asr_result["text"]

        if response_format == "text":
            return PlainTextResponse(fmt_text(text))
        if response_format == "srt":
            return PlainTextResponse(fmt_srt(final_segments))
        if response_format == "vtt":
            return PlainTextResponse(fmt_vtt(final_segments))
        if response_format == "verbose_json":
            return JSONResponse(fmt_verbose_json(final_segments, final_words, duration_s))
        if response_format == "diarized_json":
            return JSONResponse(
                fmt_diarized_json(final_segments, final_words, duration_s)
            )
        # default: json
        return JSONResponse(fmt_json(text))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "mlx_parakeet"}
