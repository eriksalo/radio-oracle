"""Kokoro TTS worker — one-shot or persistent.

One-shot mode (default): read UTF-8 text from stdin, write WAV to stdout, exit.
The ONNX session is reclaimed on process exit.

Persistent mode (``--persistent``): load Kokoro once, then service
length-prefixed requests on stdin and write length-prefixed responses on
stdout until stdin closes. Eliminates the ~2-4 s cold-start the diag server
otherwise pays on every /api/speak call.

Persistent framing:

    request  = b"<flag> <text_len>\\n" + <text_len bytes UTF-8>
               flag is 0 or 1 (radio filter off/on)
    response = b"OK <wav_len>\\n"   + <wav_len bytes WAV>
             | b"ERR <msg_len>\\n"  + <msg_len bytes UTF-8>

Stderr is inherited from the parent so worker logs land in journalctl.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radio-filter", action="store_true", help="(one-shot only) apply radio filter to output"
    )
    parser.add_argument(
        "--persistent", action="store_true", help="serve repeated requests over stdin/stdout"
    )
    args = parser.parse_args()

    if args.persistent:
        return _persistent_loop()
    return _one_shot(args.radio_filter)


def _one_shot(radio_filter: bool) -> int:
    text = sys.stdin.read()
    if not text.strip():
        return 2

    from oracle.audio import apply_radio_filter, audio_to_wav_bytes
    from oracle.tts import KokoroTTS  # noqa: E402

    tts = KokoroTTS()
    tts.load()
    audio = tts.synthesize(text)
    if radio_filter:
        audio = apply_radio_filter(audio, tts.sample_rate)
    wav = audio_to_wav_bytes(audio, tts.sample_rate)
    sys.stdout.buffer.write(wav)
    sys.stdout.flush()
    return 0


def _persistent_loop() -> int:
    from oracle.audio import apply_radio_filter, audio_to_wav_bytes
    from oracle.tts import KokoroTTS  # noqa: E402

    tts = KokoroTTS()
    tts.load()

    out = sys.stdout.buffer
    inp = sys.stdin.buffer

    out.write(b"READY\n")
    out.flush()

    while True:
        header = inp.readline()
        if not header:
            return 0  # parent closed stdin
        try:
            flag_str, len_str = header.decode("ascii").strip().split()
            radio_filter = flag_str == "1"
            text_len = int(len_str)
        except (UnicodeDecodeError, ValueError):
            _write_err(out, f"bad header: {header!r}")
            continue

        text_bytes = _read_exact(inp, text_len)
        if text_bytes is None:
            return 0  # EOF mid-message

        try:
            text = text_bytes.decode("utf-8")
            audio = tts.synthesize(text)
            if radio_filter:
                audio = apply_radio_filter(audio, tts.sample_rate)
            wav = audio_to_wav_bytes(audio, tts.sample_rate)
            out.write(f"OK {len(wav)}\n".encode("ascii"))
            out.write(wav)
            out.flush()
        except Exception as e:  # noqa: BLE001
            _write_err(out, f"{type(e).__name__}: {e}")


def _read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _write_err(out, msg: str) -> None:
    data = msg.encode("utf-8")
    out.write(f"ERR {len(data)}\n".encode("ascii"))
    out.write(data)
    out.flush()


if __name__ == "__main__":
    sys.exit(main())
