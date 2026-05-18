#!/usr/bin/env bash
# Download models needed by the Oracle.
# Usage: ./scripts/download_models.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
MODELS_DIR="models"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Would download the following:"
fi

mkdir -p "$MODELS_DIR"

# Whisper small.en model for whisper.cpp
WHISPER_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
WHISPER_FILE="$MODELS_DIR/whisper-small.en.bin"

if [[ "$DRY_RUN" == true ]]; then
    echo "  Whisper: $WHISPER_URL -> $WHISPER_FILE (~460MB)"
else
    if [[ ! -f "$WHISPER_FILE" ]]; then
        echo "Downloading Whisper small.en model..."
        wget -q --show-progress -O "$WHISPER_FILE" "$WHISPER_URL"
    else
        echo "Whisper model already exists: $WHISPER_FILE"
    fi
fi

# Kokoro TTS model + voices
KOKORO_MODEL_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
KOKORO_MODEL="$MODELS_DIR/kokoro-v1.0.onnx"
KOKORO_VOICES="$MODELS_DIR/voices-v1.0.bin"

if [[ "$DRY_RUN" == true ]]; then
    echo "  Kokoro TTS model: $KOKORO_MODEL_URL -> $KOKORO_MODEL (~300MB)"
    echo "  Kokoro voices:    $KOKORO_VOICES_URL -> $KOKORO_VOICES (~50MB)"
else
    if [[ ! -f "$KOKORO_MODEL" ]]; then
        echo "Downloading Kokoro TTS model..."
        wget -q --show-progress -O "$KOKORO_MODEL" "$KOKORO_MODEL_URL"
    else
        echo "Kokoro model already exists: $KOKORO_MODEL"
    fi
    if [[ ! -f "$KOKORO_VOICES" ]]; then
        echo "Downloading Kokoro voices..."
        wget -q --show-progress -O "$KOKORO_VOICES" "$KOKORO_VOICES_URL"
    else
        echo "Kokoro voices already exist: $KOKORO_VOICES"
    fi
fi

# Embedding model is downloaded by sentence-transformers on first use
echo ""
echo "Note: The embedding model (all-MiniLM-L6-v2, ~80MB) will be downloaded"
echo "automatically by sentence-transformers on first use."

echo ""
echo "Done. Pull the Ollama model separately:"
echo "  ollama pull llama3.2:3b"
