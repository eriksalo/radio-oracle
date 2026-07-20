#!/usr/bin/env bash
# Idempotently bring up the echo-cancel audio topology.
#
# module-echo-cancel names its master devices explicitly, so loading it
# from default.pa races USB enumeration and fails on most boots
# ("Source aec_source does not exist" in the journal — the AEC was
# silently absent). This script waits for the USB devices, then loads
# the module and sets defaults. Run as the oracle user with
# PULSE_SERVER pointing at its daemon (radio-oracle.service does this
# via ExecStartPre).
#
# Usage: ensure_aec.sh [direct|aec]
#   direct — music/TTS to the raw USB sink (pre-2026-07 behavior)
#   aec    — music/TTS through aec_sink so the canceller has a
#            reference signal ("mono gambit"); wake word + STT hear
#            music-cancelled audio. Default: $ORACLE_AUDIO_TOPOLOGY or aec.

set -u

MODE="${1:-${ORACLE_AUDIO_TOPOLOGY:-aec}}"
MIC_SRC="alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"
SPK_SINK_BASE="alsa_output.usb-Jieli_Technology_UACDemoV1.0_415035313136340C-00"

log() { echo "ensure_aec: $*"; }

# 1. Wait for the USB mic and speaker to appear in Pulse (udev race).
for i in $(seq 1 30); do
    have_src=$(pactl list short sources 2>/dev/null | grep -c "$MIC_SRC" || true)
    spk_sink=$(pactl list short sinks 2>/dev/null | grep -o "${SPK_SINK_BASE}[^ 	]*" | head -1 || true)
    if [[ "$have_src" -ge 1 && -n "$spk_sink" ]]; then
        break
    fi
    sleep 1
done
if [[ "${have_src:-0}" -lt 1 || -z "${spk_sink:-}" ]]; then
    log "USB audio devices never appeared; leaving defaults alone"
    exit 0   # never block the service on audio topology
fi
log "devices ready (speaker sink: $spk_sink)"

# 2. Load echo-cancel if its source isn't live yet.
if ! pactl list short sources | grep -q '^[0-9]*[[:space:]]aec_source'; then
    pactl load-module module-echo-cancel \
        source_master="$MIC_SRC" \
        sink_master="$spk_sink" \
        aec_method=webrtc \
        'aec_args=analog_gain_control=0 digital_gain_control=0 noise_suppression=1 extended_filter=1 high_pass_filter=1' \
        source_name=aec_source sink_name=aec_sink \
        && log "module-echo-cancel loaded" \
        || { log "module-echo-cancel failed to load; leaving defaults alone"; exit 0; }
fi

# 3. Defaults per topology.
pactl set-default-source aec_source || true
if [[ "$MODE" == "aec" ]]; then
    # Mono gambit: playback through the canceller so it has a reference —
    # wake word and STT then hear music-cancelled audio. The AEC path is
    # ~32kHz mono, which is what a one-speaker vintage radio is anyway.
    pactl set-default-sink aec_sink || true
else
    pactl set-default-sink "$spk_sink" || true
fi
log "topology=$MODE default-sink=$(pactl get-default-sink) default-source=$(pactl get-default-source)"
