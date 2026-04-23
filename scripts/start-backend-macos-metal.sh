#!/bin/bash
# start-backend-macos-metal.sh — Start the TranscriptionSuite Python backend
#                                 on Apple Silicon without the Electron frontend.
#
# This script supports two runtime modes:
#
#   MODE A — Installed DMG / app bundle
#     The .venv and Python interpreter are bundled inside the .app at:
#       <app>/Contents/Resources/backend/.venv
#     Run:  bash start-backend-macos-metal.sh [--app /path/to/TranscriptionSuite.app]
#
#   MODE B — Source repository (development / contributors)
#     uv is used directly.  The --extra mlx dependencies must already be synced.
#     Run:  bash start-backend-macos-metal.sh --source
#
# Options
#   --app <path>      Path to TranscriptionSuite.app  (default: /Applications/TranscriptionSuite.app)
#   --source          Force source-repository mode even when the .app exists
#   --host <addr>     Bind address  (default: 127.0.0.1)
#   --port <n>        Listen port   (default: 5167)
#   --config <file>   Path to config.yaml overriding the bundled one
#   --data <dir>      Data directory for audio + database  (default: ~/Library/Application Support/TranscriptionSuite)
#   --workers <n>     Number of uvicorn workers  (default: 1)
#   --reload          Enable uvicorn hot-reload (source mode only, for development)
#   -h, --help        Show this help
#
# Standalone use does not require Node.js, Electron, or any npm packages.
# Only ffmpeg must be installed (brew install ffmpeg) for audio decoding.
#
# Config and model re-use from a previous DMG install
# ─────────────────────────────────────────────────────
# Configuration (config.yaml) and downloaded model weights are stored under
# the data directory and are shared between the bundled Electron app and this
# standalone backend.  If you have already launched the app from the DMG and
# configured a model there, this script will pick up those settings without
# any extra steps:
#
#   • config.yaml   — ~/Library/Application Support/TranscriptionSuite/config.yaml
#   • SQLite DB     — ~/Library/Application Support/TranscriptionSuite/database/
#   • Audio files   — ~/Library/Application Support/TranscriptionSuite/audio/
#   • MLX models    — cached by the mlx-community libraries under ~/.cache/huggingface/
#
# If no config.yaml is found in the data directory, a minimal Metal-tuned
# default is written automatically (see _write_default_config below).
#
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_APP="/Applications/TranscriptionSuite.app"
DEFAULT_DATA_DIR="$HOME/Library/Application Support/TranscriptionSuite"
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="5167"
DEFAULT_WORKERS="1"

APP_PATH="$DEFAULT_APP"
DATA_DIR="$DEFAULT_DATA_DIR"
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
WORKERS="$DEFAULT_WORKERS"
SOURCE_MODE=false
RELOAD=false
CUSTOM_CONFIG=""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
usage() {
  sed -n '/^# Options/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# //'
  exit 0
}

die() { echo "❌  $*" >&2; exit 1; }
info() { echo "→  $*"; }
ok()   { echo "✓  $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)     APP_PATH="$2"; shift 2 ;;
    --source)  SOURCE_MODE=true; shift ;;
    --host)    HOST="$2"; shift 2 ;;
    --port)    PORT="$2"; shift 2 ;;
    --config)  CUSTOM_CONFIG="$2"; shift 2 ;;
    --data)    DATA_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --reload)  RELOAD=true; shift ;;
    -h|--help) usage ;;
    *) die "Unknown option: $1.  Use --help for usage." ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Platform guard
# ─────────────────────────────────────────────────────────────────────────────
[[ "$(uname -s)" == "Darwin" ]] || die "This script is macOS-only."
[[ "$(uname -m)" == "arm64"  ]] || die "This script requires Apple Silicon (arm64)."

# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg check
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
  die "ffmpeg is not installed.  Run: brew install ffmpeg"
fi
ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"

# ─────────────────────────────────────────────────────────────────────────────
# Resolve runtime mode and Python interpreter
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_SRC="$REPO_ROOT/server/backend"

if [[ "$SOURCE_MODE" == false && -d "$APP_PATH" ]]; then
  # ── MODE A: app bundle ──────────────────────────────────────────────────────
  info "Using installed app bundle: $APP_PATH"
  RESOURCES="$APP_PATH/Contents/Resources"
  VENV_DIR="$RESOURCES/backend/.venv"
  PYTHON_DIR="$RESOURCES/python"

  [[ -d "$VENV_DIR" ]] || die \
    "Backend venv not found at $VENV_DIR.  Re-install from the DMG."
  [[ -d "$PYTHON_DIR" ]] || die \
    "Bundled Python not found at $PYTHON_DIR.  Re-install from the DMG."

  # Locate the versioned python binary (e.g. python3.13).
  PYTHON_BIN=""
  for bin in "$VENV_DIR/bin/python3" "$VENV_DIR/bin/python"; do
    if [[ -f "$bin" || -L "$bin" ]]; then
      # Resolve through relative symlink — the venv symlinks point to
      # ../../../python/bin/python3.13 (relative, bundle-portable).
      PYTHON_BIN="$bin"
      break
    fi
  done
  [[ -n "$PYTHON_BIN" ]] || die \
    "No python binary found in $VENV_DIR/bin.  The bundle may be incomplete."

  ok "Python: $PYTHON_BIN"

  # The server package is baked into site-packages (--no-editable at build
  # time), so PYTHONPATH is not needed.  We just activate the venv.
  export PATH="$VENV_DIR/bin:$PATH"
  export VIRTUAL_ENV="$VENV_DIR"

  UVICORN_CMD=("$PYTHON_BIN" -m uvicorn)
  APP_MODULE="server.api.main:app"

else
  # ── MODE B: source repository ───────────────────────────────────────────────
  info "Using source repository: $BACKEND_SRC"
  SOURCE_MODE=true

  [[ -d "$BACKEND_SRC" ]] || die \
    "server/backend directory not found under $REPO_ROOT.  " \
    "Run from within the TranscriptionSuite repository."

  command -v uv &>/dev/null || die \
    "uv is not installed.  Run: brew install uv"
  ok "uv: $(uv --version)"

  # Ensure the server symlink that hatchling needs exists.
  if [[ ! -L "$BACKEND_SRC/server" ]]; then
    ln -sf . "$BACKEND_SRC/server"
    ok "Created server/backend/server symlink."
  fi

  # Ensure the MLX venv is up to date.
  info "Syncing uv environment (mlx extras)…"
  (cd "$BACKEND_SRC" && uv sync --extra mlx --quiet)
  ok "uv sync complete."

  VENV_DIR="$BACKEND_SRC/.venv"
  PYTHON_BIN="$VENV_DIR/bin/python"
  export PATH="$VENV_DIR/bin:$PATH"
  export VIRTUAL_ENV="$VENV_DIR"
  export PYTHONPATH="$BACKEND_SRC"

  UVICORN_CMD=("$PYTHON_BIN" -m uvicorn)
  APP_MODULE="server.api.main:app"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Data directory and config
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR/audio" "$DATA_DIR/database" "$DATA_DIR/tokens"

_write_default_config() {
  local cfg="$1"
  info "Writing minimal Metal-tuned config.yaml to $cfg"
  cat > "$cfg" <<'YAML'
# TranscriptionSuite — standalone Metal backend config
# Generated by start-backend-macos-metal.sh
# Edit as needed; changes take effect on the next server restart.

longform_recording:
    language: null
    translation_enabled: false
    translation_target_language: "en"
    auto_add_to_audio_notebook: false

main_transcriber:
    # Apple Silicon MLX Parakeet model (25 EU languages, no GPU token needed).
    # For English-optimised: mlx-community/parakeet-tdt-1.1b
    model: "mlx-community/parakeet-tdt-0.6b-v3"
    compute_type: "default"
    device: "mps"
    gpu_device_index: 0
    batch_size: 1
    beam_size: 5
    initial_prompt: null
    faster_whisper_vad_filter: true
    no_log_file: true

parakeet:
    mlx_local_attention: true
    mlx_local_attention_window: [256, 256]
    mlx_local_attention_threshold_s: 120
    mlx_chunk_duration_s: 120
    mlx_overlap_duration_s: 15

sortformer:
    chunk_duration_s: 5.0

mlx:
    metal_cache_limit_mb: 1024

diarization:
    # On Apple Silicon, Sortformer (no HF token) is used automatically instead
    # of PyAnnote when diarization is requested via the API.
    model: null
    hf_token: null
    device: "auto"
    min_speakers: null
    max_speakers: null
    min_duration_on: 0.0
    min_duration_off: 0.0
    merge_gap_threshold: 0.5
    embedding_batch_size: 32
    parallel: false

static_transcription:
    max_segment_chars: 500
    silero_vad_preprocessing: true
    silero_vad_sensitivity: 0.5

live_transcriber:
    enabled: false
    live_language: "en"
    model: "Systran/faster-whisper-medium"
    compute_type: "default"
    device: "mps"
    gpu_device_index: 0
    batch_size: 1
    beam_size: 5
    silero_sensitivity: 0.4
    silero_use_onnx: false
    post_speech_silence_duration: 3.0
    min_length_of_recording: 1.0
    early_transcription_on_silence: 0.5
    no_log_file: true

storage:
    audio_dir: "__DATA_DIR__/audio"
    database_dir: "__DATA_DIR__/database"
    audio_format: "mp3"
    audio_bitrate: 160

backup:
    enabled: true
    max_age_hours: 1
    max_backups: 3

processing:
    temp_dir: "/tmp/transcriptionsuite"
    keep_temp_files: false
    sample_rate: 16000

audio_processing:
    backend: "ffmpeg"
    resampler: "soxr"
    normalization_method: "dynaudnorm"

remote_server:
    enabled: true
    host: "127.0.0.1"
    token_store: "__DATA_DIR__/tokens/tokens.json"
    tls:
        enabled: false
        cert_file: ""
        key_file: ""

local_llm:
    enabled: false
    base_url: "http://127.0.0.1:1234"
    api_key: ""
    auto_title_enabled: true
YAML
  # Substitute the actual data dir path.
  sed -i '' "s|__DATA_DIR__|$DATA_DIR|g" "$cfg"
}

# Resolve config path: CLI flag → data-dir copy → default
if [[ -n "$CUSTOM_CONFIG" ]]; then
  [[ -f "$CUSTOM_CONFIG" ]] || die "Config file not found: $CUSTOM_CONFIG"
  CONFIG_PATH="$CUSTOM_CONFIG"
  ok "Using custom config: $CONFIG_PATH"
elif [[ -f "$DATA_DIR/config.yaml" ]]; then
  CONFIG_PATH="$DATA_DIR/config.yaml"
  ok "Using existing config: $CONFIG_PATH"
else
  CONFIG_PATH="$DATA_DIR/config.yaml"
  _write_default_config "$CONFIG_PATH"
  ok "Default config written: $CONFIG_PATH"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Environment variables consumed by the server
# ─────────────────────────────────────────────────────────────────────────────
export CONFIG_PATH
export DATA_DIR
# Tell MLX to cap the Metal buffer pool (mirrors config mlx.metal_cache_limit_mb).
export MLX_METAL_CACHE_LIMIT_MB="${MLX_METAL_CACHE_LIMIT_MB:-1024}"

echo ""
echo "=================================================="
echo "  TranscriptionSuite — Metal backend"
echo "  Host   : http://${HOST}:${PORT}"
echo "  Config : $CONFIG_PATH"
echo "  Data   : $DATA_DIR"
echo "=================================================="
echo ""
echo "  OpenAI-compatible endpoint:"
echo "    POST http://${HOST}:${PORT}/v1/audio/transcriptions"
echo "    POST http://${HOST}:${PORT}/v1/audio/translations"
echo ""
echo "  Health check:"
echo "    GET  http://${HOST}:${PORT}/health"
echo ""
echo "  API docs (Swagger UI):"
echo "    http://${HOST}:${PORT}/docs"
echo ""
echo "  Press Ctrl-C to stop."
echo "=================================================="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Launch uvicorn
# ─────────────────────────────────────────────────────────────────────────────
UVICORN_ARGS=(
  "$APP_MODULE"
  --host "$HOST"
  --port "$PORT"
  --workers "$WORKERS"
  --log-level info
)

if [[ "$RELOAD" == true ]]; then
  if [[ "$SOURCE_MODE" == false ]]; then
    echo "⚠️  --reload is only useful in source mode and will be ignored."
  else
    UVICORN_ARGS+=(--reload)
  fi
fi

exec "${UVICORN_CMD[@]}" "${UVICORN_ARGS[@]}"
