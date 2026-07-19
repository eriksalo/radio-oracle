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
# mpg123 is the music decoder (Player spawns it per track); pulseaudio +
# pulseaudio-utils carry the AEC mic path and pactl volume bridge.
echo "1. Installing system dependencies..."
run_cmd apt-get update -qq
run_cmd apt-get install -y -qq python3-venv python3-dev portaudio19-dev \
    alsa-utils libasound2-dev wget curl git mpg123 pulseaudio pulseaudio-utils

# 2. Create oracle user
# uid is pinned to 999: radio-oracle.service hard-depends on
# user@999.service and points PULSE_SERVER at /run/user/999.
echo "2. Creating oracle user..."
if ! id -u oracle &>/dev/null; then
    run_cmd useradd -r -m -s /bin/bash -u 999 -G gpio,audio,video,i2c oracle
else
    echo "  User 'oracle' already exists (uid $(id -u oracle))"
    if [[ "$(id -u oracle)" != "999" ]]; then
        echo "  WARNING: radio-oracle.service assumes uid 999 (user@999.service," \
             "/run/user/999). Fix the unit or the uid."
    fi
    run_cmd usermod -aG gpio,audio,video,i2c oracle
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
# Install Python under /opt so it's readable by the oracle service user, not in
# /root/.local/share/uv/ where sudo would put it by default.
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
run_cmd mkdir -p "$UV_PYTHON_INSTALL_DIR"
run_cmd chmod 0755 "$UV_PYTHON_INSTALL_DIR"
run_cmd "$UV_BIN" python install 3.11
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
if [[ "$DRY_RUN" != true ]]; then
    # Ollama's installer enables the systemd unit, but the daemon may still be
    # binding to 127.0.0.1:11434 when we try to pull. Wait up to 30s.
    for i in {1..30}; do
        if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi
run_cmd ollama pull qwen3:4b-instruct-2507-q4_K_M

# 7. Download voice models
echo "7. Downloading voice models..."
run_cmd bash "$INSTALL_DIR/scripts/download_models.sh"

# 8. Configure PulseAudio (asymmetric AEC routing — see docs/SETUP.md §1.6)
# NOTE: the app talks to PulseAudio, NOT raw ALSA. An /etc/asound.conf that
# redefines pcm.!default (an earlier version of this script wrote one)
# conflicts with Pulse's device handling — remove it if present.
echo "8. Configuring PulseAudio for the oracle user..."
if [[ -f /etc/asound.conf ]] && grep -q "Radio Oracle USB audio config" /etc/asound.conf; then
    run_cmd rm /etc/asound.conf
    echo "  Removed stale /etc/asound.conf (superseded by PulseAudio routing)"
fi
run_cmd mkdir -p /home/oracle/.config/pulse
run_cmd cp "$INSTALL_DIR/systemd/pulse-default.pa" /home/oracle/.config/pulse/default.pa
run_cmd chown -R oracle:oracle /home/oracle/.config/pulse
# The service needs the oracle user's Pulse daemon at /run/user/999.
run_cmd loginctl enable-linger oracle

# 9. Install systemd services (main app + diagnostics page)
echo "9. Installing systemd services..."
run_cmd cp "$INSTALL_DIR/systemd/radio-oracle.service" /etc/systemd/system/
run_cmd cp "$INSTALL_DIR/systemd/radio-oracle-diag.service" /etc/systemd/system/
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
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
OLLAMA
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Check device names in systemd/pulse-default.pa match yours (pactl list short sinks/sources)"
echo "  2. rsync FAISS indices from workstation: rsync -av workstation:radio-oracle/data/faiss/ $INSTALL_DIR/data/faiss/"
echo "     (ChromaDB stays on the workstation — FAISS cutover 2026-05-19)"
echo "  3. Index music + books: .venv/bin/python scripts/index_music.py <dir>; scripts/index_books.py <dir>"
echo "  4. Test: sudo systemctl start radio-oracle && journalctl -fu radio-oracle"
echo "  5. Reboot and verify auto-start"
