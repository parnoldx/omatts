# <img src="site/assets/favicon.svg" alt="" width="28" height="28"> Omatts

**Website:** [parnoldx.github.io/omatts](https://parnoldx.github.io/omatts/)

Fast, fully local text-to-speech with voice cloning. One C++ file drives the
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) model through ONNX
Runtime — no Python, no cloud, 9.2x realtime with 30ms to first audio.

- **Zero-shot voice cloning** — point it at any WAV/MP3/FLAC clip and it speaks in that voice
- **Runs on CPU** — INT8 models, mmap'd weights, a daemon that keeps everything warm
- **CLI and HTTP server** — including an OpenAI-compatible `/v1/audio/speech` endpoint, so SillyTavern, Open WebUI, and friends work as drop-in clients
- **Linux, macOS, Windows** — no frameworks, one binary plus one data blob

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/parnoldx/omatts/master/install.sh | sh
```

Binary to `~/.local/bin`, models and voices to `~/.local/share/omatts`.
Requires `curl` and `zstd`. Override with `PREFIX`, `OMATTS_VERSION`, or
`OMATTS_URL` — see the script header.

## Usage

```bash
omatts Hello world                        # speaks out loud
omatts -v dhh Hello world                 # pick a voice from voices/
omatts -v ~/clips/me.mp3 Hello world      # ...or clone from any audio file
omatts -o out.wav Hello world             # write a WAV instead of playing
omatts -o - Hello world | ffplay -nodisp -autoexit -
echo "Task finished" | omatts             # read from stdin
omatts serve --port 8080                  # OpenAI-compatible HTTP server
```

The first call warms up a background daemon that keeps the model loaded;
later calls start in milliseconds. It exits after 5 idle minutes.

More:

| Ask | Run |
|---|---|
| what it can do | `omatts help` |
| what voices exist | `omatts voices` |
| where to drop a voice sample | `omatts voices open` |
| set a default voice | `export OMATTS_VOICE=george` |

A voice is a name from `voices/` or any WAV/MP3/FLAC file. Text that starts
with `-`: `omatts -- -foo bar`.

## HTTP API

| Endpoint | What it is |
|---|---|
| `POST /v1/audio/speech` | OpenAI-compatible TTS — a drop-in base-URL swap. `model` and `speed` accepted but ignored; `response_format` is `wav` (default) or `pcm` |
| `POST /tts` | Streams chunked PCM (`audio/pcm;rate=24000;encoding=float;bits=32`) for low-latency clients |
| `GET /health` | Health check |

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "dhh"}' \
  --output speech.wav
```

## Options

| Flag | Default | What it does |
|---|---|---|
| `-v`, `--voice <name\|file>` | `alba` / `$OMATTS_VOICE` | Voice name or audio file |
| `-o`, `--output <file>` | play | Write WAV; `-` streams to stdout |
| `-q`, `--quiet` | — | No status output |
| `--precision <int8\|fp32>` | `int8` | The flow model always uses fp32 when available — int8 flow sounds robotic |
| `--temperature <f>` | `0.3` | Sampling temperature |
| `--lsd-steps <n>` | `2` | Flow matching ODE solver steps |
| `--seed <n>` | `0` | Fixed seed for reproducible output (`0` = random) |
| `--threads <n>` | `0` | Thread budget (`0` = half the cores) |
| `--no-cache` | — | Disable both disk caches |
| `--no-daemon` | — | Generate in-process, never start the daemon |
| `--idle-exit <sec>` | `300` | Daemon exits after this much idle time (`0` = never) |
| `--models-dir <dir>` | `~/.local/share/omatts/models` | Or `$OMATTS_MODELS_DIR` |
| `--voices-dir <dir>` | `~/.local/share/omatts/voices` | Or `$OMATTS_VOICES_DIR` |
| `--port <n>` | `8080` | Port for `omatts serve` |

## Advanced

### Caching

Two disk caches under `voices/.cache/`, both generated on first use:

- **`.emb`** — the Mimi encoder output for a voice sample, so the WAV is not re-encoded on every run.
- **`.kv`** — the transformer's KV state after voice conditioning. Cold conditioning costs hundreds of milliseconds; the snapshot restores in ~4ms and is also held in memory so only the first sentence pays for it.

Both invalidate when the source audio changes. `rm -rf voices/.cache/` clears
them; `--no-cache` disables them entirely.

### Build from source

```bash
make dist    # exports models, builds, packs dist/omatts-linux-x86_64.tar.zst
make deploy  # ...and publishes it as a GitHub release
```

`make dist` runs `export_onnx.py`, which exports, quantizes, and validates all
ONNX models from the upstream weights (requires
[uv](https://docs.astral.sh/uv/); it pins a pocket-tts commit because newer
releases dropped the `conditioners` module the script imports). Weights land in
`models/` as small `.onnx` graphs with `.onnx.data` sidecars that ONNX Runtime
mmaps at load — keep each `.data` next to its `.onnx`. All dependencies (ONNX
Runtime, SentencePiece, dr_wav) are fetched by CMake; you need CMake 3.28+ and
a C++17 compiler.

## Licenses

- Code: MIT — see [LICENSE](LICENSE).
- Model weights: the ONNX files in `models/` are derived from Kyutai Labs'
  [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) and are licensed
  [CC-BY-4.0](models/LICENSE-WEIGHTS.txt) — attribution and modification
  notice included there.

## Credits

- [Kyutai Labs](https://github.com/kyutai-labs/pocket-tts) — Pocket TTS model and original Python implementation (MIT)
- [Verylicious/pocket-tts-ungated](https://huggingface.co/Verylicious/pocket-tts-ungated) — Ungated weights and tokenizer (CC-BY-4.0)
