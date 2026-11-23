# Firmware artifacts

Binary firmware for the ReSpeaker Lite USB mic (XMOS XU316). Kept in-repo
so a re-flash doesn't require pulling from the vendor's GitHub at recovery
time.

## Files

| File | Notes |
|------|-------|
| `ffva_ua_v2.0.8_test.bin` | **Installed since 2026-11-23.** Unofficial Seeed test build (forum thread "ReSpeaker Lite AEC support", post 27, 2025-04-01, by their firmware developer; never released on GitHub). sha256 `c04635ca…a75a94`. The only build whose echo cancellation works — see below. |
| `respeaker_lite_usb_dfu_firmware_v2.0.7.bin` | Last official release; the rollback image. Flashed 2026-05-22 → 2026-11-23. |

## Checking the installed version

```bash
sudo lsusb -v -d 2886:0019 | grep bcdDevice
# bcdDevice            2.08   ← installed version
```

`bcdDevice` matches the firmware version (e.g. `2.07` = v2.0.7).

## Flashing

The XU316 exposes three DFU partitions: FACTORY (alt 0), UPGRADE (alt 1),
DATAPARTITION (alt 2). We only ever write to UPGRADE; FACTORY stays
untouched as the recovery image — a bad flash to alt 1 is not fatal.

```bash
sudo apt install -y dfu-util
sudo systemctl stop radio-oracle           # release the mic
sudo dfu-util -R -a 1 -D ffva_ua_v2.0.8_test.bin      # or the v2.0.7 file to roll back
# wait for device to re-enumerate (a few seconds)
sudo systemctl start radio-oracle
```

`dfu-util` will print "Invalid DFU suffix signature" — that's benign;
XMOS firmware binaries don't include the standard DFU suffix.

## What the firmware provides

On-chip DSP runs before audio reaches the host:

- **IC** — two-mic spatial interference cancellation
- **NS** — noise suppression
- **AGC** — automatic gain control (whisper-to-shout normalization)
- **VNR** — voice-to-noise-ratio estimation

**On-chip AEC** (speaker on the board's own amplifier since 2026-11-22),
measured 2026-11-23 with 50 s of speech at 30 % sink volume, mic level in
dBFS while the speaker plays:

| Firmware | 0–10 s | 20–50 s | Quiet room |
|---|---|---|---|
| v2.0.7 | −15…−20 | −15…−20 (never converges) | −65 |
| v2.0.8 test | −21 | **−48** | −55 |

On v2.0.8 the canceller needs ~15–20 s of playback the first time after
power-up, then stays locked across pauses (3 s, 20 s, sink idle) and
re-locks in ~4 s. It holds fully up to ~55 % sink volume and degrades by
a few dB up to 80 %. **A sudden jump UP in level (30 → 55 % in one step)
unlocks it** — echo back at full level until it re-learns; the same change
as small steps, or a sudden drop, does not. `oracle/volume_bridge.py`
therefore ramps the knob upward. Not measured: whether a person talking
during playback comes through, and wake-word / transcription quality on a
real voice with this build.

Roll back with the Flashing command above and the v2.0.7 file.

## Upstream

Firmware source: https://github.com/respeaker/ReSpeaker_Lite/tree/master/xmos_firmwares
