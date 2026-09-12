#!/usr/bin/env python3
"""End-to-end tests for the omatts binary: real inference, CLI, and HTTP.

Unlike tests/test_omatts.cpp (pure logic), this drives the built binary with
the real models and a committed voice. Exit codes:
    0  all checks passed
    1  a check failed
    77 models are not present -> skipped (ctest SKIP_RETURN_CODE)

Run:  python3 tests/integration.py [--binary PATH] [--models DIR] [--voices DIR]
"""

import argparse
import json
import math
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKS = FAILED = 0


def check(cond, label):
    global CHECKS, FAILED
    CHECKS += 1
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED += 1
        print(f"  FAIL {label}", file=sys.stderr)


def wav_info(data):
    """Return (fmt, channels, rate, bits, data_bytes) or None if not a WAV."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    fmt, ch = struct.unpack_from("<HH", data, 20)
    (rate,) = struct.unpack_from("<I", data, 24)
    (bits,) = struct.unpack_from("<H", data, 34)
    (dsz,) = struct.unpack_from("<I", data, 40)
    if dsz == 0xFFFFFFFF:  # streaming WAV header: size unknown up front
        dsz = len(data) - 44
    if len(data) < 44 + dsz:
        return None
    return fmt, ch, rate, bits, dsz


def rms(data):
    dsz = wav_info(data)[4]
    n = dsz // 4
    if n == 0:
        return 0.0
    s = struct.unpack_from(f"<{n}f", data, 44)
    return math.sqrt(sum(x * x for x in s) / n)


def run(args, stdin=None):
    return subprocess.run(args, input=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=180)


def cli(common, *args, stdin=None):
    return run([common["binary"], "--quiet", "--no-daemon",
                "--models-dir", common["models"], "--voices-dir", common["voices"],
                *args], stdin=stdin)


def gen(common, text, out, *extra):
    r = cli(common, "-v", "dhh", "-o", out, *extra, text)
    if r.returncode != 0:
        print(r.stderr.decode(errors="replace"), file=sys.stderr)
    return r


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def http_test(common, tmp):
    port = free_port()
    proc = subprocess.Popen(
        [common["binary"], "serve", "--quiet", "--port", str(port),
         "--models-dir", common["models"], "--voices-dir", common["voices"]],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120
        while True:
            try:
                body = urllib.request.urlopen(base + "/health", timeout=5).read()
                break
            except Exception:
                if proc.poll() is not None or time.time() > deadline:
                    check(False, "http: server came up")
                    return
                time.sleep(0.25)
        check(json.loads(body) == {"status": "ok"}, "http: /health returns ok")

        voices = json.loads(urllib.request.urlopen(base + "/v1/audio/voices",
                                                   timeout=5).read())
        ids = [v["voice_id"] for v in voices.get("data", [])]
        check("dhh" in ids and "de/klaus" in ids and "de" in ids,
              "http: /v1/audio/voices lists top-level, pack and default voices")
        defaults = [v for v in voices.get("data", []) if v.get("default")]
        check(all(v["language"] and v["voice_id"] == v["language"] for v in defaults)
              and len(defaults) == 1, "http: exactly one marked pack default (de)")

        payload = json.dumps({"input": "Hello there.", "voice": "dhh",
                              "response_format": "wav"}).encode()
        req = urllib.request.Request(base + "/v1/audio/speech", data=payload,
                                     headers={"Content-Type": "application/json"})
        info = wav_info(urllib.request.urlopen(req, timeout=120).read())
        check(info is not None and info[2] == 24000 and info[4] > 0,
              "http: /v1/audio/speech returns 24kHz WAV audio")

        try:
            bad = urllib.request.Request(base + "/v1/audio/speech",
                                         data=json.dumps({"voice": "dhh"}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(bad, timeout=30)
            code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        check(code == 400, "http: missing 'input' is a 400, not a crash")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default=os.path.join(ROOT, "omatts"))
    p.add_argument("--models", default=os.path.join(ROOT, "models"))
    p.add_argument("--voices", default=os.path.join(ROOT, "voices"))
    args = p.parse_args()

    if not os.path.exists(os.path.join(args.models, "flow_lm_main_int8.onnx")):
        print(f"integration: skipped, no models in {args.models}")
        return 77
    if not os.path.exists(args.binary):
        print(f"integration: binary not built: {args.binary}", file=sys.stderr)
        return 1

    common = {"binary": args.binary, "models": args.models, "voices": args.voices}
    tmp = os.path.join(ROOT, ".build", "it")
    os.makedirs(tmp, exist_ok=True)
    a, b, c = (os.path.join(tmp, n) for n in ("a.wav", "b.wav", "c.wav"))

    # CLI: valid audio out, not silence.
    r = gen(common, "Hello world.", a)
    data = open(a, "rb").read() if r.returncode == 0 else b""
    info = wav_info(data)
    check(r.returncode == 0 and info is not None and info[0] == 3
          and info[1] == 1 and info[2] == 24000 and info[3] == 32,
          "cli: -o writes a 24kHz mono float32 WAV")
    check(info is not None and info[4] > 0 and rms(data) > 1e-4,
          "cli: output is non-empty and not silent")

    # Same --seed -> byte-identical audio.
    gen(common, "Hello world.", a, "--seed", "42")
    gen(common, "Hello world.", b, "--seed", "42")
    check(open(a, "rb").read() == open(b, "rb").read(),
          "cli: --seed makes generation reproducible")

    # [[pause 1]] adds ~1s of audio.
    gen(common, "Hello world.", a, "--seed", "42")
    gen(common, "Hello world. [[pause 1]]", c, "--seed", "42")
    d0, d1 = wav_info(open(a, "rb").read())[4], wav_info(open(c, "rb").read())[4]
    check(abs((d1 - d0) / 4 / 24000.0 - 1.0) < 0.2,
          "cli: [[pause 1]] adds ~1s of silence")

    # stdin -> stdout.
    r = cli(common, "-v", "dhh", "-o", "-", stdin=b"Hi from stdin.")
    check(r.returncode == 0 and wav_info(r.stdout) is not None, 
          "cli: reads text from stdin and streams WAV to stdout")

    # --volume scales the gain; 0.1 is the floor (not silence).
    gen(common, "Hello world.", a, "--seed", "42", "--volume", "0.1")
    gen(common, "Hello world.", b, "--seed", "42", "--volume", "1.0")
    ratio = rms(open(a, "rb").read()) / rms(open(b, "rb").read())
    check(0.05 < ratio < 0.15, "cli: --volume scales output gain (0.1 vs 1.0)")

    # Missing models fail loudly instead of crashing.
    r = run([args.binary, "-q", "--no-daemon", "--models-dir", os.path.join(tmp, "nope"),
             "-v", "dhh", "-o", a, "Hello."])
    check(r.returncode != 0 and b"model" in r.stderr.lower(),
          "cli: missing models exit non-zero with an error")

    http_test(common, tmp)

    print(f"{CHECKS - FAILED}/{CHECKS} integration checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
