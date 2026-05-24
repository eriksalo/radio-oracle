# Firmware artifacts

Binary firmware for the ReSpeaker Lite USB mic (XMOS XU316). Kept in-repo
so a re-flash doesn't require pulling from the vendor's GitHub at recovery
time.

## Files

| File | Notes |
|------|-------|
| `respeaker_lite_usb_dfu_firmware_v2.0.7.bin` | Current shipping firmware. Flashed to all units 2026-05-22. |

## Checking the installed version

```bash
sudo lsusb -v -d 2886:0019 | grep bcdDevice
# bcdDevice            2.07   ← installed version
```

`bcdDevice` matches the firmware version (e.g. `2.07` = v2.0.7).

## Flashing

The XU316 exposes three DFU partitions: FACTORY (alt 0), UPGRADE (alt 1),
DATAPARTITION (alt 2). We only ever write to UPGRADE; FACTORY stays
untouched as the recovery image — a bad flash to alt 1 is not fatal.

```bash
sudo apt install -y dfu-util
sudo systemctl stop radio-oracle           # release the mic
sudo dfu-util -R -a 1 -D respeaker_lite_usb_dfu_firmware_v2.0.7.bin
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

**On-chip AEC is also a feature of this firmware but doesn't apply in our
architecture** — the XU316 has no reference signal for our separate USB
speaker (UACDemoV1.0), so it can't cancel that echo. Software AEC via
PulseAudio (`module-echo-cancel`) handles echo cancellation instead. See
`docs/SETUP.md` §1.6.

## Upstream

Firmware source: https://github.com/respeaker/ReSpeaker_Lite/tree/master/xmos_firmwares
