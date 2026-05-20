# voice-aider

> Voice input interface for [aider](https://aider.chat), a terminal-based AI coding agent.
> Speak your prompts and commands; aider's responses render in a live Streamlit dashboard.

---

## Demo setup time: **~2 minutes** (after `.env` is filled in)

The bulk of that is `pip install` resolving wheels. A second-time demo on the same machine takes **~10 seconds** (just two `python` commands in two terminals).

---

## What it does

A two-process system:

| Process | What it owns | How to run |
|---|---|---|
| **Voice pipeline** | mic, VAD, Groq Whisper, LangGraph state machine, the aider subprocess | `python -m pipeline.main` |
| **Streamlit dashboard** | live transcript, confidence, intent badge, aider's response, rolling history | `streamlit run ui/app.py` |

The two processes communicate through a single file: `tmp/state.json`, written atomically by the pipeline and polled once per second by the UI.

Architecture, folder layout, and the LangGraph node diagram are in **REPORT.md**.

---

## One-time setup

### 1. Clone and install

```bash
git clone <repo-url> voice-aider
cd voice-aider
python -m venv .venv && source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# then edit .env and paste your Groq key:
#   GROQ_API_KEY=gsk_...
```

Get a free Groq key at <https://console.groq.com/keys>. The free tier is plenty for a live demo — Whisper STT and Llama-3.1-8b classification are both included.

### 3. Create the IPC bridge folder

```bash
mkdir -p tmp && touch tmp/.gitkeep
```

### 4. (Linux only) Audio + input permissions

```bash
sudo apt-get install -y portaudio19-dev   # for sounddevice on Linux
# pynput needs no special permissions (unlike the `keyboard` package)
```

macOS / Windows: nothing extra. PyPI wheels handle it.

---

## Running the demo

You need **two terminals** open in the project root, both with the virtualenv activated.

**Terminal 1 — voice pipeline:**

```bash
python -m pipeline.main
```

You should see:

```
==========================================================
voice-aider pipeline starting
  STT model     : whisper-large-v3
  LLM model     : llama-3.1-8b-instant
  Aider model   : groq/llama-3.1-8b-instant
  Mode          : PTT (space)
  Confidence    : 0.75 (≤2 retries)
==========================================================
MicCapture started: 16000 Hz, 480-sample frames
Waiting for PTT (space key)…
```

**Terminal 2 — UI:**

```bash
streamlit run ui/app.py
```

Streamlit opens <http://localhost:8501> in your browser. The dashboard shows mode, status, the latest transcript, aider's reply, and a rolling history.

---

## How to use it

The default mode is **push-to-talk** (PTT) on the SPACEBAR.

1. **Hold SPACE.** Speak your prompt or command.
2. **Release SPACE.** The pipeline transcribes, classifies, and either runs the command or sends the prompt to aider.
3. **Watch the dashboard.** Transcript, confidence, and aider's response appear within ~1–2 seconds.

To switch to fully hands-free mode (WebRTC VAD silence-detection, no key needed): set `HANDS_FREE_MODE=true` in `.env` and restart the pipeline.

---

## Voice command reference

Anything not in this list is treated as a prompt for aider.

| Say… | What happens |
|---|---|
| **"clear"**, "clear screen" | Clears aider's chat history (`/clear`) |
| **"undo"**, "undo that", "revert" | Reverts aider's last edit (`/undo`) |
| **"save"**, "commit" | Commits current changes (`/commit`) |
| **"exit"**, "quit", "stop session" | Shuts down aider (`/exit`) |
| **"help"** | Shows aider's help (`/help`) |
| **"history"**, "what did I say" | Highlights the history panel in the UI |
| **"scroll up"** / **"scroll down"** | UI-only navigation |
| **anything else** | Sent to aider as a prompt |

**Examples of prompts:**

- "Write a Python function that returns the Fibonacci sequence up to n."
- "Refactor this file to use async/await."
- "Add a docstring to every function in `utils/`."
- "Explain what the `aider_node` does."

---

## Configuration knobs

Edit `.env` to tune behaviour without touching code:

| Variable | Default | Notes |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.75` | Below this, `stt_node` retries |
| `MAX_STT_RETRIES` | `2` | Retry budget per utterance |
| `VAD_AGGRESSIVENESS` | `2` | `0` (loose) → `3` (strict) |
| `SILENCE_TIMEOUT_MS` | `800` | End-of-utterance silence window for VAD |
| `PUSH_TO_TALK_KEY` | `space` | `space` / `ctrl` / `alt` |
| `HANDS_FREE_MODE` | `false` | `true` → pure VAD, no key needed |
| `AIDER_MODEL` | `groq/llama-3.1-8b-instant` | Any model aider supports |

---

## Project layout

```
voice-aider/
├── pipeline/                # voice process: mic + VAD + main loop
│   ├── main.py              # entry: `python -m pipeline.main`
│   ├── capture.py           # mic input, push-to-talk (sounddevice + pynput)
│   └── vad.py               # WebRTC VAD silence detection (hands-free mode)
├── graph/                   # LangGraph state machine
│   ├── voice_graph.py       # StateGraph wiring
│   ├── state.py             # VoiceState TypedDict
│   ├── edges.py             # conditional routing functions
│   └── nodes/
│       ├── stt_node.py            # Groq Whisper → transcript + confidence
│       ├── confidence_node.py     # retry gate
│       ├── classify_node.py       # fuzzy allowlist → Groq Llama fallback
│       ├── execute_cmd.py         # slash commands → aider stdin
│       ├── aider_node.py          # persistent aider subprocess + I/O
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
├── tmp/
│   └── state.json           # IPC bridge (runtime; .gitignored except .gitkeep)
├── tests/
├── .env.example
├── requirements.txt
├── README.md
└── REPORT.md                # design narrative
```

---

## Troubleshooting

**"aider executable not found"** — Aider isn't installed. `pip install aider-chat` (already in requirements.txt; happens if you skipped `pip install -r`).

**The UI shows "waiting for pipeline…" forever** — The pipeline process isn't running. Start Terminal 1.

**PTT doesn't trigger on Linux without sudo** — You're probably using the `keyboard` package by accident. This project uses `pynput`, which doesn't need root.

**Whisper returns empty transcripts** — Your mic might be muted, or `VAD_AGGRESSIVENESS` is set too high. Try `1` or `0` in `.env`.

**"GROQ_API_KEY missing"** — `.env` isn't being picked up. Confirm it's in the project root (same folder as `requirements.txt`), not in `pipeline/` or `ui/`.

---

## License

MIT.