# Retraining the "librarian" wake word with livekit-wakeword

The current detector is a custom openWakeWord model (`models/librarian.onnx`,
threshold 0.7). livekit-wakeword (Apache-2.0, Feb 2026) is its successor:
same ONNX inference contract — **backward compatible with openWakeWord
models and the openwakeword library**, so `oracle/wakeword.py` needs no
code change, only a model file swap — with vendor-reported ~100× fewer
false positives (0.08 vs 8.50 FP/hour) and +17pt recall. Those numbers are
vendor-run; the A/B below is the real gate.

Training is fully synthetic (TTS-generated positives + adversarial
negatives) — the old `training/` feature dumps aren't needed. Run this on
the GPU workstation.

## 1. Train (workstation)

```bash
uv tool install "livekit-wakeword[train,eval,export]"

cat > librarian.yaml << 'YAML'
model_name: librarian
target_phrases:
  - "librarian"
  # near-miss pronunciations help recall; the pipeline generates
  # adversarial negatives (library, libertarian, ...) automatically
n_samples: 10000
model:
  model_type: conv_attention
  model_size: small
steps: 50000
target_fp_per_hour: 0.2
YAML

livekit-wakeword run librarian.yaml   # generate → augment → train → export → eval
```

Output: `librarian.onnx` + an eval report (FP/hour, recall). Keep the eval
numbers.

## 2. A/B on the Jetson

Ship the new model NEXT TO the old one and switch via env, so rollback is
one line:

```bash
rsync -avz -e ssh --rsync-path="sudo rsync" \
    librarian.onnx erik@radio-oracle.local:/opt/radio-oracle/models/librarian-lk.onnx

# /opt/radio-oracle/.env
#   ORACLE_WAKEWORD_MODEL=models/librarian-lk.onnx
#   ORACLE_WAKEWORD_THRESHOLD=0.5   # re-tune: livekit scores aren't
#                                   # calibrated like the old model's
sudo systemctl restart radio-oracle
```

Test protocol (repeat for old model if you never measured it):
1. **Recall**: say "librarian" 20× at conversation distance, 10× from
   across the room, 10× with music playing. Record hit rates separately —
   the music case is expected to be weak for BOTH models (AEC has no music
   reference; the button remains the wake during playback).
2. **False positives**: leave the radio playing music overnight
   (~8 hours); count wake events in the journal
   (`journalctl -u radio-oracle | grep "Wake word detected"`).
3. Adjust `ORACLE_WAKEWORD_THRESHOLD` and re-run 1–2 until recall ≥ old
   model AND overnight FPs < old model.

Promote: rename to `models/librarian.onnx`, drop the env overrides, commit
the model + the measured numbers here.

## Results

| Metric | openWakeWord (old) | livekit (new) |
|---|---|---|
| Recall @ conversation distance | | |
| Recall @ across room | | |
| Recall during music | | |
| False positives / night | | |
| Threshold used | 0.7 | |
