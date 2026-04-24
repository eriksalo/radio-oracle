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

# Piper TTS model (lessac medium)
PIPER_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_JSON_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
PIPER_FILE="$MODELS_DIR/en_US-lessac-medium.onnx"
PIPER_JSON="$MODELS_DIR/en_US-lessac-medium.onnx.json"

if [[ "$DRY_RUN" == true ]]; then
    echo "  Piper TTS: $PIPER_URL -> $PIPER_FILE (~75MB)"
else
    if [[ ! -f "$PIPER_FILE" ]]; then
        echo "Downloading Piper TTS model..."
        wget -q --show-progress -O "$PIPER_FILE" "$PIPER_URL"
        wget -q --show-progress -O "$PIPER_JSON" "$PIPER_JSON_URL"
    else
        echo "Piper model already exists: $PIPER_FILE"
    fi
fi

# Embedding model is downloaded by sentence-transformers on first use
echo ""
echo "Note: The embedding model (all-MiniLM-L6-v2, ~80MB) will be downloaded"
echo "automatically by sentence-transformers on first use."

echo ""
echo "Done. Pull the Ollama model separately:"
echo "  ollama pull llama3.2:3b"
