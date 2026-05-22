# voice-aider

> A voice input interface for [aider](https://aider.chat), a terminal-based AI coding agent.
> Hold SPACE, speak a command or coding prompt, release. Responses render in a live Streamlit dashboard.

---

## Demo setup time

**~2 minutes** from a fresh clone, dominated by `pip install` resolving wheels.
**~10 seconds** on a warm machine (just two `python` commands in two terminals).

A full setup → first voice command sequence is in §4 below.

---

## 1. What it does

Two processes, one file between them.

| Process | What it owns | How to run |
|---|---|---|
| **Voice pipeline** | mic, push-to-talk, VAD, Groq Whisper, LangGraph state machine, the aider subprocess | `python -m pipeline.main` |
| **Streamlit dashboard** | live transcript, confidence, intent badge, aider's reply, rolling history | `streamlit run ui/app.py` |

The pipeline writes `tmp/state.json` atomically after every utterance; the dashboard polls it once per second. Two completely independent processes, one tiny JSON file between them.

A walkthrough of the architecture, the LangGraph node diagram, and every key design decision — including the bugs I hit and how I fixed them — lives in **[REPORT.md](./REPORT.md)**.

---

## 2. One-time setup

### Clone + dependencies

```bash
git clone <repo-url> voice-aider
cd voice-aider
python -m venv .venv

# activate the venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

pip install -r requirements.txt

# Optional dev extras (tests, lint, type-check)
pip install pytest pytest-cov ruff mypy
```

> **Don't run `pip install -e .`** — on Windows paths containing spaces (e.g. `D:\data science\…`), setuptools' editable install creates a `.pth` file that shadows the local `graph` package as an empty namespace, causing `ImportError: cannot import name 'build_graph' from 'graph' (unknown location)`. The project doesn't need an editable install — `conftest.py` puts the project root on `sys.path` for tests, and the runtime commands resolve imports from the current directory.

### Configure secrets

```bash
cp .env.example .env
# then edit .env and paste your Groq key:
#   GROQ_API_KEY=gsk_...
```

Get a free Groq key at <https://console.groq.com/keys>. The free tier handles the entire demo — Whisper STT and Llama-3.1-8b classification are both included.

### Create the IPC bridge folder

```bash
mkdir -p tmp && touch tmp/.gitkeep      # macOS / Linux
mkdir tmp && type nul > tmp\.gitkeep    # Windows
```

### (Linux only) audio prereq

```bash
sudo apt-get install -y portaudio19-dev
```

macOS and Windows ship the audio backend in the PyPI wheels — nothing extra needed.

---

## 3. Verifying the install (optional but useful)

```bash
pytest          # 49 tests, all should pass in <1s
```

If `pytest` is green, your project structure, imports, settings loader, and graph wiring are all sane. Doesn't touch the mic or Groq.

To also verify the API connection before a live demo:

```bash
python -c "from groq_clients.llm_client import classify; print(classify('refactor this to use async'))"
```

Expected: `ClassificationResult(intent='prompt', action=None)`.

---

## 4. Running the demo

Open **two terminals** in the project root, both with the venv activated.

### Terminal 1 — voice pipeline

```bash
python -m pipeline.main
```

Wait for the banner to settle on:

```
Pre-warming aider subprocess (this takes ~3-5s)…
Aider ready.
MicCapture started: 16000 Hz, 480-sample frames
Waiting for PTT (space key)…
```

Aider is pre-warmed at startup so the first voice command doesn't pay the 3-5 second cold-start tax.

### Terminal 2 — UI

```bash
streamlit run ui/app.py
```

A browser opens at `http://localhost:8501`. Sidebar has the voice command cheat-sheet; main area shows mode, status, latest turn, aider's reply, and rolling history.

### Speak

The PTT key is **SPACEBAR**, configured in `.env`. Hold it down, speak, release. Within 1–2 seconds the dashboard reflects the turn.

Suggested demo sequence (covers all three branches of the LangGraph conditional edge):

| Say | What you should see |
|---|---|
| "Write a function that reverses a string" | Transcript appears, intent=`prompt → aider`, aider's reply renders in the agent-output panel, history gets a new entry |
| "Clear" | Fuzzy match → `/clear` sent to aider, confirmation in the cmd-result row |
| "Exit" | Pipeline detects the exit intent, both processes shut down cleanly |

---

## 5. Voice command reference

Anything not in this list is sent to aider as a free-form coding prompt.

| Say… | What happens |
|---|---|
| **"clear"**, "clear screen" | `/clear` to aider (resets the chat history) |
| **"undo"**, "undo that", "revert", "go back" | `/undo` to aider (reverts the last edit) |
| **"save"**, "commit" | `/commit` to aider |
| **"help"** | `/help` to aider |
| **"exit"**, "quit", "stop session" | Shuts down the whole session — mic, aider, pipeline |
| **"history"**, "what did I say" | UI-only — highlights the history panel |
| **"scroll up"** / **"scroll down"** | UI-only — page navigation |
| **anything else** | Sent to aider as a prompt |

Examples of free-form prompts that work well in a demo: *"Write a Python function that returns the Fibonacci sequence up to n"*, *"Refactor this file to use async/await"*, *"Explain what the aider_node does"*, *"Add a docstring to every function in utils/"*.

---

## 6. Configuration knobs

Edit `.env` to tune behaviour without touching code. Everything is loaded by `python-dotenv` through `pydantic-settings`; there are no raw `os.getenv` calls anywhere in the project.

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Free key from console.groq.com |
| `GROQ_STT_MODEL` | `whisper-large-v3` | swap for `whisper-large-v3-turbo` if you want lower latency |
| `GROQ_LLM_MODEL` | `llama-3.1-8b-instant` | used by `classify_node` |
| `CONFIDENCE_THRESHOLD` | `0.75` | below this, `stt_node` retries (capped by `MAX_STT_RETRIES`) |
| `MAX_STT_RETRIES` | `2` | retry budget per utterance |
| `VAD_AGGRESSIVENESS` | `2` | `0` (loose) → `3` (strict). Only matters in hands-free mode. |
| `SILENCE_TIMEOUT_MS` | `800` | end-of-utterance silence window for VAD |
| `PUSH_TO_TALK_KEY` | `space` | `space` / `ctrl` / `alt` |
| `HANDS_FREE_MODE` | `false` | `true` → pure VAD, no key needed |
| `AIDER_MODEL` | `groq/llama-3.1-8b-instant` | any model aider supports |
| `AIDER_EXTRA_ARGS` | `--no-auto-commit` | merged into the aider command line |

The aider subprocess is started with `--no-pretty --no-stream --yes-always --map-tokens 0 --no-show-model-warnings --no-check-update`. These flags keep the output parseable (no `prompt_toolkit` TTY assumptions), make replies atomic instead of token-streamed (matches our quiet-window detection), auto-accept add-to-chat prompts, and disable the repo-map so we stay well under Groq's free-tier 6k TPM limit. If you upgrade to a paid Groq tier, set `AIDER_EXTRA_ARGS=--map-tokens 1024 --no-auto-commit` for richer codebase awareness.

---

## 7. Project layout

```
voice-aider/
├── pipeline/                # voice process: mic + VAD + main loop
│   ├── main.py              # entry: `python -m pipeline.main`
│   ├── capture.py           # sounddevice mic input, pynput push-to-talk
│   └── vad.py               # WebRTC VAD silence detection (hands-free mode)
├── graph/                   # LangGraph state machine
│   ├── voice_graph.py       # StateGraph wiring
│   ├── state.py             # VoiceState TypedDict
│   ├── edges.py             # conditional routing functions
│   └── nodes/
│       ├── stt_node.py            # Groq Whisper → transcript + confidence
│       ├── confidence_node.py     # retry gate (skips identical retries)
│       ├── classify_node.py       # fuzzy allowlist → Groq Llama fallback
│       ├── execute_cmd.py         # cmd → aider slash command or UI signal
│       ├── aider_node.py          # persistent aider subprocess + auto-restart
│       └── write_state_node.py    # state → tmp/state.json
├── groq_clients/
│   ├── whisper_client.py    # confidence proxy from segment avg_logprob
│   └── llm_client.py        # JSON-mode classification call
├── ui/                      # Streamlit dashboard process
│   ├── app.py               # entry: `streamlit run ui/app.py`
│   ├── state_reader.py
│   └── components/          # status_bar, transcript_panel, aider_output, history_panel
├── config/
│   ├── settings.py          # pydantic-settings, single load_dotenv() call
│   ├── commands.py          # strict CMD allowlist + fuzzy threshold
│   └── prompts.py           # classify_node system prompt
├── utils/
│   ├── audio_utils.py       # int16 PCM ↔ WAV bytes
│   ├── logger.py            # module-scoped logging
│   └── state_bridge.py      # atomic JSON writes for IPC
├── tests/                   # 49 tests, all pass in <1s
├── tmp/                     # IPC bridge (runtime; .gitignored except .gitkeep)
├── conftest.py              # root pytest conftest — sys.path bootstrap
├── .env.example
├── requirements.txt
├── pyproject.toml           # pytest + ruff + mypy config (NOT an installable package)
├── README.md                # this file
└── REPORT.md                # design narrative + honest limitations
```

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: cannot import name X from 'graph' (unknown location)` | Most likely you ran `pip install -e .` and got a broken editable install. Run `pip uninstall -y voice-aider`, delete `venv\Lib\site-packages\__editable__*.pth`, clear `__pycache__`, restart. |
| `aider executable not found` | `pip install aider-chat` (already in requirements.txt — happens if you skipped `pip install -r`). |
| Dashboard stays on "waiting for pipeline…" | The pipeline process isn't running. Start Terminal 1. |
| Whisper returns empty transcripts | Mic muted, or `VAD_AGGRESSIVENESS=3` cutting off soft speech. Try `1` or `0`. |
| `GROQ_API_KEY missing` | `.env` not in the project root (must sit next to `requirements.txt`, not inside `pipeline/` or `ui/`). |
| Rate-limit errors mid-demo | You're on Groq free tier (6k TPM). `--map-tokens 0` already mitigates this; if it still fires, wait 60s. |
| Windows: `prompt_toolkit` "No Windows console" warning in aider's reply | Cosmetic; aider's response follows the warning correctly. `--no-pretty` flag suppresses most of it. |

---

## License

MIT.