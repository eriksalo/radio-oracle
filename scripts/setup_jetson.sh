#!/usr/bin/env bash
# One-time setup for the Jetson Orin Nano Super.
# Usage: sudo ./scripts/setup_jetson.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
INSTALL_DIR="/opt/radio-oracle"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Would perform the following:"
    echo ""
fi

run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "  $ $*"
    else
        "$@"
    fi
}

echo "=== Radio Oracle Jetson Setup ==="
echo ""

# 1. System dependencies
echo "1. Installing system dependencies..."
run_cmd apt-get update -qq
run_cmd apt-get install -y -qq python3-venv python3-dev portaudio19-dev \
    alsa-utils libasound2-dev wget curl git

# 2. Create oracle user
echo "2. Creating oracle user..."
if ! id -u oracle &>/dev/null; then
    run_cmd useradd -r -m -s /bin/bash -G gpio,audio,video oracle
else
    echo "  User 'oracle' already exists"
fi

# 3. Clone/update repo
echo "3. Setting up application directory..."
if [[ ! -d "$INSTALL_DIR" ]]; then
    run_cmd mkdir -p "$INSTALL_DIR"
    echo "  Clone the repo to $INSTALL_DIR or rsync from workstation"
fi

# 4. Python venv (uv-managed Python 3.11)
echo "4. Creating Python virtual environment..."
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "/home/erik/.local/bin/uv" ]]; then
    UV_BIN="/home/erik/.local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
    echo "  ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
run_cmd "$UV_BIN" venv --seed --python 3.11 "$INSTALL_DIR/.venv"
run_cmd "$INSTALL_DIR/.venv/bin/pip" install -e "$INSTALL_DIR[all,dev]"

# 5. Install Ollama
echo "5. Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "  $ curl -fsSL https://ollama.com/install.sh | sh"
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
else
    echo "  Ollama already installed"
fi

# 6. Pull LLM model
echo "6. Pulling LLM model..."
run_cmd ollama pull llama3.2:3b

# 7. Download voice models
echo "7. Downloading voice models..."
run_cmd bash "$INSTALL_DIR/scripts/download_models.sh"

# 8. Configure ALSA
echo "8. Configuring ALSA for USB audio..."
if [[ "$DRY_RUN" == true ]]; then
    echo "  Would configure /etc/asound.conf for USB audio"
else
    if [[ ! -f /etc/asound.conf ]]; then
        cat > /etc/asound.conf << 'ASOUND'
# Radio Oracle USB audio config
pcm.!default {
    type asym
    playback.pcm "plughw:1,0"
    capture.pcm "plughw:2,0"
}
ctl.!default {
    type hw
    card 1
}
ASOUND
        echo "  Created /etc/asound.conf (adjust card numbers for your hardware)"
    else
        echo "  /etc/asound.conf already exists"
    fi
fi

# 9. Install systemd service
echo "9. Installing systemd service..."
run_cmd cp "$INSTALL_DIR/systemd/radio-oracle.service" /etc/systemd/system/
run_cmd systemctl daemon-reload
run_cmd systemctl enable radio-oracle

# 10. Headless optimization
echo "10. Optimizing for headless operation..."
if [[ "$DRY_RUN" == true ]]; then
    echo "  $ systemctl set-default multi-user.target"
    echo "  Would set OLLAMA_MAX_LOADED_MODELS=1"
else
    systemctl set-default multi-user.target
    mkdir -p /etc/systemd/system/ollama.service.d
    cat > /etc/systemd/system/ollama.service.d/override.conf << 'OLLAMA'
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
OLLAMA
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Adjust /etc/asound.conf for your USB audio devices (arecord -l / aplay -l)"
echo "  2. rsync ChromaDB data from workstation: rsync -av workstation:radio-oracle/data/chroma/ $INSTALL_DIR/data/chroma/"
echo "  3. Test: sudo systemctl start radio-oracle && journalctl -fu radio-oracle"
echo "  4. Reboot and verify auto-start"
