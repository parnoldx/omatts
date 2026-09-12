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

The installer asks whether to also install the German pack (~120 MB); say
yes, or set it up front / add it later:

```bash
curl -fsSL .../install.sh | OMATTS_PACKS=de sh                       # fresh install incl. German
curl -fsSL .../install.sh | OMATTS_PACKS=de OMATTS_PACKS_ONLY=1 sh   # add German later
omatts -v de/klaus Hallo Welt
```

## Usage

```bash
omatts Hello world                        # speaks out loud
omatts -v dhh Hello world                 # pick a voice from voices/
omatts -v de Hallo Welt                   # a language tag = that pack's default voice (finn)
omatts -v ~/clips/me.mp3 Hello world      # ...or clone from any audio file
omatts -o out.wav Hello world             # write a WAV instead of playing
omatts -o - Hello world | ffplay -nodisp -autoexit -
echo "Task finished" | omatts             # read from stdin
omatts "Hello [[pause 1]] world"          # 1s of silence via [[pause N]] tag
omatts "Softly. [[volume 2]] Louder."     # inline volume, reset with [[volume]]
omatts serve --port 8080                  # OpenAI-compatible HTTP server
```

The first call warms up a background daemon that keeps the model loaded;
later calls start in milliseconds. It exits after 5 idle minutes.

In-text pauses: `[[pause N]]` inserts N seconds of silence (default `0.5` with
no number, clamped to 0–10); works on stdin, `-o`, and the HTTP API.

Symbol rules are per language pack and versioned in the repo:
`models/normalize.txt` (English) and `models-de/normalize.txt` (German) —
they speak `$100` out as "100 dollars", `50%` as "50 percent", `&` as "and",
plus `€`, `£`, `°C`, `°F`, digit `+`/`-`/`*` → plus/minus/times,
`/` → "divided by" between digits or "slash" elsewhere, `@` → "at";
German says "Dollar/Euro/Prozent/Grad Celsius".

Format: one rule per line, `regex<TAB>replacement` (std::regex syntax,
rules run in order), `#` comments; broken lines are skipped with a warning.
**No `normalize.txt` in a pack = no symbol expansion for that pack** — the
feature is opt-in per language, never built in. Rules reload when a voice
tag like `de/juergen` switches the pack. Example:

    ([0-9]) ?%	$1 Prozent
    € ?([0-9][0-9,]*(?:\.[0-9]+)?)	$1 Euro

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
| `POST /v1/audio/speech` | OpenAI-compatible TTS — a drop-in base-URL swap. `model` accepted but ignored; `response_format` is `wav` (default), `pcm`, `mp3` or `opus` (mp3/opus require ffmpeg); `speed` honored (0.5–4.0, resampled) |
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
| `-v`, `--voice <name\|file>` | `alba` / `$OMATTS_VOICE` | Voice name or audio file; a bare language tag (`-v de`) uses that pack's default voice; `tag/name` like `de/juergen` selects the language pack |
| `-o`, `--output <file>` | play | Write WAV, MP3 or OPUS (those two need ffmpeg); `-` streams WAV to stdout, `-.mp3`/`-.opus` that format |
| `-q`, `--quiet` | — | No status output |
| `--speed <f>` | `1.0` | Speaking speed 0.5–4.0, pitch unchanged |
| `--volume <f>` | `1.0` | Output gain 0.1–2 |
| `--precision <int8\|fp32>` | `int8` | The flow model always uses fp32 when available |
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
make deploy  # exports models, builds, packs dist/, publishes the release
```

`make deploy` runs `export_onnx.py`, which exports, quantizes, and validates all
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
