# Workstream 5: Text-to-voice (TTS + audio I/O)

The "voice" of the Oracle. Owns Piper TTS, speaker playback, microphone
capture, voice activity detection, and the AM-radio filter that gives the
output its vintage character.

## Status

Working end-to-end. Piper streams sentence-by-sentence; the radio filter is
applied; mic capture has VAD-based silence detection. STT lives in
Workstream 7 (it's part of the input pipeline rather than the audio
subsystem).

## Scope

- Piper TTS wrapper (CPU, ONNX) — `synthesize(text) → np.ndarray`
- Sentence-level streaming so the first words come out fast
- Speaker playback via `sounddevice`
- AM-radio DSP filter (bandpass + saturation) applied to TTS output
- Microphone capture with VAD-driven silence detection
- Sample-rate handling between Piper output and mic input

## File ownership

```
oracle/
  tts.py                   # PiperTTS wrapper
  audio.py                 # play_audio, record_until_silence, apply_radio_filter
config/
  persona.toml             # voice config lives in [tts] here
tests/
  (no dedicated tests yet — see TODO)
```

`oracle/audio.py` is shared with Workstream 7 (mic capture for STT) but its
ownership and design live here because it's all DSP/audio-I/O code.

## Settings

```bash
ORACLE_PIPER_MODEL_PATH=models/en_US-lessac-medium.onnx
ORACLE_PIPER_SAMPLE_RATE=22050
ORACLE_AUDIO_SAMPLE_RATE=16000              # rate Whisper sees (after resample)
ORACLE_AUDIO_CHANNELS=1
ORACLE_VAD_ENERGY_THRESHOLD=0.004
ORACLE_VAD_SILENCE_DURATION=1.5
# PortAudio can't address ALSA `plughw`/`asym` PCMs, so devices are pinned
# by name and opened at their native rates. Capture is resampled in software
# down to ORACLE_AUDIO_SAMPLE_RATE before STT.
ORACLE_AUDIO_INPUT_DEVICE=ReSpeaker
ORACLE_AUDIO_OUTPUT_DEVICE=UACDemoV1.0
ORACLE_AUDIO_CAPTURE_SAMPLE_RATE=16000
ORACLE_AUDIO_PLAYBACK_SAMPLE_RATE=48000
ORACLE_VOICE_PLAY_GREETING=true             # off-switch for the boot greeting
```

The AM-radio filter (`apply_radio_filter`) is still implemented in
`oracle/audio.py` but is **not** applied to TTS by default — the persona
rework switched the Oracle to a clean Librarian voice. The filter is left
in place for future use by the music player (Workstream 3).

## Dependencies

```bash
pip install -e ".[tts,voice]"
./scripts/download_models.sh      # fetches Piper voices
```

`[tts]` brings `piper-tts`; `[voice]` brings `sounddevice` + `scipy`.

## Interface contract

**Provides** (consumed by Workstreams 3, 4, 7):
- `PiperTTS.synthesize(text: str) → np.ndarray` (float32 PCM at `sample_rate`)
- `PiperTTS.sample_rate: int`
- `play_audio(samples, sample_rate)` — blocking playback
- `apply_radio_filter(samples, sample_rate) → np.ndarray`
- `record_until_silence() → np.ndarray` (mic capture with VAD)

**Consumes**: nothing. This is a leaf workstream.

**Concurrency rule**: only one caller may hold the speaker at a time. The
orchestration layer (Workstream 7) coordinates between TTS, music, and
book-reader by pausing the active producer before starting another.

## Standalone exercise

```bash
# Synthesize one phrase to a file (no playback dependency)
python -c "
from oracle.tts import PiperTTS
import scipy.io.wavfile as wav
t = PiperTTS()
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
from oracle.tts import PiperTTS
from oracle.audio import play_audio, apply_radio_filter
t = PiperTTS()
s = t.synthesize('Static and steel.')
play_audio(apply_radio_filter(s, t.sample_rate), t.sample_rate)
"
```

## TODO

- [ ] Pre-warm Piper at startup so the first response isn't slow
- [ ] Better sentence boundary detection (abbreviations, decimals, "Mr.")
- [ ] `tests/test_audio.py` — radio filter spectral check
- [ ] Configurable AM-filter intensity per producer (TTS gentle, music heavier)
- [ ] Streaming Piper (sample-by-sample) so playback can start before
      synthesis finishes (currently sentence-buffered)
- [ ] Multiple voices selectable from `persona.toml`
