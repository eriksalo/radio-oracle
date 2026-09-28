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

# Parakeet-TDT-0.6B v2 int8 (sherpa-onnx bundle) — used when
# ORACLE_STT_BACKEND=parakeet
PARAKEET_NAME="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
PARAKEET_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$PARAKEET_NAME.tar.bz2"
PARAKEET_DIR="$MODELS_DIR/$PARAKEET_NAME"

if [[ "$DRY_RUN" == true ]]; then
    echo "  Parakeet STT: $PARAKEET_URL -> $PARAKEET_DIR (~700MB)"
else
    if [[ ! -d "$PARAKEET_DIR" ]]; then
        echo "Downloading Parakeet-TDT-0.6B v2 int8..."
        wget -q --show-progress -O "$MODELS_DIR/$PARAKEET_NAME.tar.bz2" "$PARAKEET_URL"
        tar -xjf "$MODELS_DIR/$PARAKEET_NAME.tar.bz2" -C "$MODELS_DIR"
        rm "$MODELS_DIR/$PARAKEET_NAME.tar.bz2"
    else
        echo "Parakeet model already exists: $PARAKEET_DIR"
    fi
fi

# End-of-utterance models (ORACLE_VAD_BACKEND=silero / silero+smartturn):
# Silero VAD (sherpa-onnx export, ~640KB) and Pipecat Smart Turn v3.2
# int8 (BSD-2, ~8.7MB). See oracle/endpoint.py.
SILERO_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
SILERO_FILE="$MODELS_DIR/silero_vad.onnx"
SMARTTURN_URL="https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
SMARTTURN_FILE="$MODELS_DIR/smart-turn-v3.2-cpu.onnx"

if [[ "$DRY_RUN" == true ]]; then
    echo "  Silero VAD:  $SILERO_URL -> $SILERO_FILE (~640KB)"
    echo "  Smart Turn:  $SMARTTURN_URL -> $SMARTTURN_FILE (~8.7MB)"
else
    for pair in "$SILERO_URL|$SILERO_FILE" "$SMARTTURN_URL|$SMARTTURN_FILE"; do
        url="${pair%%|*}"; file="${pair##*|}"
        if [[ ! -f "$file" ]]; then
            echo "Downloading $(basename "$file")..."
            wget -q --show-progress -O "$file" "$url"
        else
            echo "Already exists: $file"
        fi
    done
fi

# Streaming STT (ORACLE_STT_BACKEND=nemotron-streaming): NVIDIA
# Nemotron-speech-streaming-en-0.6b, sherpa-onnx int8 export (~460MB).
NEMOTRON_NAME="sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
NEMOTRON_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$NEMOTRON_NAME.tar.bz2"
NEMOTRON_DIR="$MODELS_DIR/$NEMOTRON_NAME"

if [[ "$DRY_RUN" == true ]]; then
    echo "  Nemotron streaming STT: $NEMOTRON_URL -> $NEMOTRON_DIR (~460MB)"
else
    if [[ ! -d "$NEMOTRON_DIR" ]]; then
        echo "Downloading Nemotron streaming STT..."
        wget -q --show-progress -O "$MODELS_DIR/$NEMOTRON_NAME.tar.bz2" "$NEMOTRON_URL"
        tar -xjf "$MODELS_DIR/$NEMOTRON_NAME.tar.bz2" -C "$MODELS_DIR"
        rm "$MODELS_DIR/$NEMOTRON_NAME.tar.bz2"
    else
        echo "Nemotron streaming model already exists: $NEMOTRON_DIR"
    fi
fi

# Query-side embedder for ORACLE_EMBEDDING_RUNTIME=onnx: nomic-embed-text
# v1.5 fp32 ONNX (~520MB; identical vectors to the sentence-transformers
# build used for the FAISS indices) + tokenizer. model_int8.onnx (131MB)
# is optional: 3x faster but cos 0.97 to the index vectors.
NOMIC_DIR="$MODELS_DIR/nomic-embed-text-v1.5-onnx"
NOMIC_BASE="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main"
if [[ "$DRY_RUN" == true ]]; then
    echo "  nomic ONNX embedder: $NOMIC_BASE/onnx/model.onnx (+tokenizer) -> $NOMIC_DIR"
else
    mkdir -p "$NOMIC_DIR"
    for f in onnx/model.onnx tokenizer.json tokenizer_config.json config.json special_tokens_map.json; do
        if [[ ! -f "$NOMIC_DIR/$(basename "$f")" ]]; then
            echo "Downloading nomic $(basename "$f")..."
            wget -q --show-progress -O "$NOMIC_DIR/$(basename "$f")" "$NOMIC_BASE/$f"
        fi
    done
fi

# Embedding model is downloaded by sentence-transformers on first use
echo ""
echo "Note: The embedding model (all-MiniLM-L6-v2, ~80MB) will be downloaded"
echo "automatically by sentence-transformers on first use."

echo ""
echo "Done. Pull the Ollama model separately:"
echo "  ollama pull qwen3:4b-instruct-2507-q4_K_M"
