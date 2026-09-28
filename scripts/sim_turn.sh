#!/usr/bin/env bash
# Run the latency harness on the Jetson with the service stopped, and
# always bring the service back — even if the harness crashes or the ssh
# session that launched this goes away. Detach it:
#
#   nohup sudo /opt/radio-oracle/scripts/sim_turn.sh [sim_turn.py args] \
#       > /dev/null 2>&1 &
#   tail -f /tmp/sim_turn.log
#
# Must run as root (systemctl + sudo -u oracle).
set -uo pipefail

ROOT=/opt/radio-oracle
LOG=${SIM_LOG:-/tmp/sim_turn.log}
# Any script that needs the service stopped can ride the same wrapper,
# optionally from another interpreter (e.g. the cp310 TTS sidecar venv).
SCRIPT=${SIM_SCRIPT:-scripts/sim_turn.py}
PYTHON=${SIM_PYTHON:-$ROOT/.venv/bin/python}

restart() {
    systemctl start radio-oracle
    echo "# radio-oracle restarted: $(systemctl is-active radio-oracle)" >> "$LOG"
}
trap restart EXIT

: > "$LOG"
echo "# $(date -Is) stopping radio-oracle" >> "$LOG"
systemctl stop radio-oracle
sleep 3
free -h | sed -n 2p >> "$LOG"

cd "$ROOT"
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a
sudo -E -u oracle -H "$PYTHON" "$ROOT/$SCRIPT" "$@" >> "$LOG" 2>&1
echo "# harness exit=$?" >> "$LOG"
free -h | sed -n 2p >> "$LOG"
