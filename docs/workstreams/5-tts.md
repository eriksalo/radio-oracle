# Workstream 5: Text-to-voice (TTS + audio I/O)

The "voice" of the Oracle. Owns Kokoro TTS, speaker playback, microphone
capture, voice activity detection, volume control, and the AM-radio filter
that gives the output its vintage character.

## Status

Working end-to-end. Kokoro TTS (82M param ONNX, CPU-only, 24kHz) with
`am_michael` voice. Volume knob (potentiometer via ADS1115) controls
playback gain with quadratic scaling. Recording and playback both support
abort callbacks for immediate cutoff on power-off.

## Scope

- Kokoro TTS wrapper (CPU, ONNX) — `synthesize(text) → np.ndarray`
- Sentence-level streaming so the first words come out fast
- Speaker playback via `sounddevice` with volume control
- AM-radio DSP filter (bandpass) for vintage speaker feel
- Microphone capture with VAD-driven silence detection
- Abort callbacks on recording and playback (power switch integration)
- Sample-rate handling between TTS output (24kHz) and speaker (48kHz)

## File ownership

```
oracle/
  tts.py                   # KokoroTTS wrapper (kokoro-onnx)
  audio.py                 # play_audio, record_until_silence, apply_radio_filter,
                           # volume control integration, abort support
oracle/hardware/
  volume.py                # VolumeControl — pot % → quadratic gain 0.0–1.0
tests/
  (no dedicated tests yet — see TODO)
```

`oracle/audio.py` is shared with Workstream 7 (mic capture for STT) but its
ownership and design live here because it's all DSP/audio-I/O code.

## Settings

```bash
ORACLE_TTS_MODEL_PATH=models/kokoro-v1.0.onnx
ORACLE_TTS_VOICES_PATH=models/voices-v1.0.bin
ORACLE_TTS_VOICE=am_michael                    # American male, natural
ORACLE_TTS_SPEED=1.0
ORACLE_AUDIO_SAMPLE_RATE=16000                 # rate Whisper sees (after resample)
ORACLE_AUDIO_CHANNELS=1
ORACLE_VAD_ENERGY_THRESHOLD=0.004
ORACLE_VAD_SILENCE_DURATION=1.5
# On the Jetson, audio I/O routes through oracle's per-user PulseAudio,
# which loads module-echo-cancel to provide AEC for the music-during-wake-word
# case. See docs/SETUP.md §1.6 for the AEC stack rationale and config.
# ORACLE_AUDIO_*_DEVICE=pulse resolves to PulseAudio's default source/sink,
# which are the AEC'd virtual devices (aec_source / aec_sink).
ORACLE_AUDIO_INPUT_DEVICE=pulse
ORACLE_AUDIO_OUTPUT_DEVICE=pulse
# On dev machines without the PulseAudio AEC stack, pin devices directly:
# ORACLE_AUDIO_INPUT_DEVICE=ReSpeaker
# ORACLE_AUDIO_OUTPUT_DEVICE=UACDemoV1.0
ORACLE_AUDIO_CAPTURE_SAMPLE_RATE=16000
ORACLE_AUDIO_PLAYBACK_SAMPLE_RATE=48000
```

The AM-radio filter (`apply_radio_filter`) is implemented in `oracle/audio.py`
but is **not** applied to TTS by default — the persona uses a clean Librarian
voice. The filter is available for future use by the music player (Workstream 3).

## Dependencies

```bash
pip install -e ".[tts,voice]"
./scripts/download_models.sh      # fetches Kokoro model + voices
```

`[tts]` brings `kokoro-onnx`; `[voice]` brings `sounddevice` + `scipy`.

## Interface contract

**Provides** (consumed by Workstreams 3, 4, 7):
- `KokoroTTS.synthesize(text: str) → np.ndarray` (float32 PCM at 24kHz)
- `KokoroTTS.sample_rate: int` (24000)
- `play_audio(samples, sample_rate, should_abort)` — blocking playback with volume + abort
- `apply_radio_filter(samples, sample_rate) → np.ndarray`
- `record_until_silence(should_abort) → np.ndarray` (mic capture with VAD + abort)
- `VolumeControl.gain: float` — 0.0–1.0, read from pot at playback time

**Consumes**: nothing. This is a leaf workstream.

**Concurrency rule**: only one caller may hold the speaker at a time. The
orchestration layer (Workstream 7) coordinates between TTS, music, and
book-reader by pausing the active producer before starting another.

## Standalone exercise

```bash
# Synthesize one phrase to a file (no playback dependency)
python -c "
from oracle.tts import KokoroTTS
import scipy.io.wavfile as wav
t = KokoroTTS()
samples = t.synthesize('Good day, citizen. The Oracle is online.')
wav.write('/tmp/oracle.wav', t.sample_rate, samples)
print('wrote /tmp/oracle.wav')
"

# Round-trip mic → speaker (verifies audio devices)
python -c "
from oracle.audio import record_until_silence, play_audio
audio = record_until_silence()
play_audio(audio, 16000)
"

# Synthesize + play with radio filter
python -c "
from oracle.tts import KokoroTTS
from oracle.audio import play_audio, apply_radio_filter
t = KokoroTTS()
s = t.synthesize('Static and steel.')
play_audio(apply_radio_filter(s, t.sample_rate), t.sample_rate)
"
```

## TODO

- [ ] Better sentence boundary detection (abbreviations, decimals, "Mr.")
- [ ] `tests/test_audio.py` — radio filter spectral check
- [ ] Configurable AM-filter intensity per producer (TTS gentle, music heavier)
- [ ] Multiple voices selectable from `persona.toml`
- [ ] Streaming TTS (sample-by-sample) so playback can start before
      synthesis finishes (currently sentence-buffered)
