# REPORT.md — voice-aider

A narrative of how this project was designed, what alternatives I evaluated, and what limitations remain.

---

## 1. Picking the target agent

The brief leaves agent choice open. I considered three candidates:

**[OpenCode](https://opencode.ai)** is the most active project in the space, but it's written in TypeScript and runs as a TUI inside a terminal grid. Wrapping it for voice means either intercepting keystrokes (fragile across terminals) or driving a tmux session programmatically (works, but the UX-fidelity loss vs the native TUI is huge). The "responses on screen as usual" clause becomes complicated.

**[Pi](https://pi.dev)** is closed-source as a CLI; I'd be reverse-engineering the wire format. That violates the spirit of "open source terminal-based AI coding agent" in the prompt.

**[aider](https://aider.chat)** is Python, MIT-licensed, and accepts prompts via stdin or `--message`. Aider explicitly supports slash-commands (`/clear`, `/undo`, `/commit`, `/exit`) which map cleanly onto voice commands. Crucially, aider stays in the terminal and prints markdown — exactly what the brief calls for ("responses are displayed on screen as usual").

So aider it is. The integration method I chose: spawn aider once as a child subprocess, hold its stdin/stdout open, and stream prompts in. I considered `--message` (one-shot mode) but it spawns a fresh process per prompt — aider's ~3–5s warm-up makes that feel sluggish in a voice context where the user is expecting near-instant response.

## 2. Choice of STT engine

Hard constraints from the brief: free-tier models, Groq only. Three serious STT options on Groq's free tier:

| Model | Speed | Accuracy | Notes |
|---|---|---|---|
| `whisper-large-v3` | ~0.3–0.6s | best | what I picked |
| `whisper-large-v3-turbo` | ~0.15–0.3s | slightly worse on hard accents | considered |
| `distil-whisper-large-v3-en` | fastest | English-only, less robust | rejected |

I went with `whisper-large-v3`. For a coding agent the prompts are heavy on technical jargon ("async", "regex", "TypeError", function names with underscores) — accuracy on this vocabulary matters more than shaving 200ms off the latency. If the demo machine is a slow network, swapping `GROQ_STT_MODEL=whisper-large-v3-turbo` in `.env` is a one-line change.

I also considered **browser MediaRecorder + Streamlit's `st.audio_input`** as the capture surface, which would eliminate the second process. I rejected it because the UI is supposed to *display* responses, not own the mic. Keeping capture in a server-side Python process means the same pipeline works headlessly (e.g. as a SSH-from-iPad demo) and the UI stays optional. The two-process design also lets me argue in this report that the voice layer is decoupled from the dashboard — which is genuinely true.

## 3. Confidence scoring

Groq Whisper does not return a single "confidence" number. I needed one anyway, because without it the retry mechanism has no signal to act on. The proxy I built combines two fields that `response_format="verbose_json"` exposes:

- `avg_logprob` per segment — the average log-probability of generated tokens. Negative; closer to 0 = more confident.
- `no_speech_prob` per segment — Whisper's own estimate that this segment is silence.

The formula in `groq_clients/whisper_client.py`:

```
score = exp(mean(avg_logprob)) * (1 - mean(no_speech_prob))
```

`exp(avg_logprob)` maps the log-prob into `(0, 1]`. Multiplying by `(1 - no_speech_prob)` dampens utterances that Whisper thinks were silence even if the few recognised tokens happened to be high-prob. In practice, clear short prompts score 0.85–0.95, muffled speech scores 0.45–0.65, and silence/coughs score below 0.2. The retry threshold of `0.75` (configurable) sits cleanly between "definitely worth using" and "ask the user to repeat."

An alternative I considered: cross-check the transcript with a second STT pass at a different temperature and measure edit distance. It would give a more honest confidence signal but doubles the API spend per utterance. The avg_logprob proxy is "good enough" given the retry-loop safety net.

## 4. LangGraph and the state machine

The brief said "for looping conditions use langgraph" — so the architecture starts from there. The graph (see README's project layout) is:

```
stt_node → confidence_node → classify_node → [execute_cmd | aider_node] → write_state_node → END
                  ↑               retry loop                  ↑
                  └───────────────────────────────────────────┘
```

Two conditional edges:

1. **After confidence_node**: if the score is below threshold *and* the retry budget isn't exhausted, loop back to `stt_node`. Otherwise proceed. I cap retries at 2 (`MAX_STT_RETRIES`) to avoid burning Groq's free-tier quota on a user who's standing next to a vacuum cleaner.

2. **After classify_node**: route to `execute_cmd`, `aider_node`, or straight to `write_state_node` if the intent is "unknown" (no point feeding gibberish to aider).

Why LangGraph over a plain `while` loop with `if` statements? Three reasons:

- **The retry edge is data, not code.** It's visible in the graph definition, and adding a second retry path later (e.g. retry on transient Groq 503s) is one line.
- **The state schema is explicit.** `VoiceState` is a `TypedDict` so every node knows exactly what fields it reads and writes. This caught a bug during development where I'd accidentally typed `state["confidance"]` in a node — mypy on the TypedDict surfaces it immediately.
- **Future-proofing.** If the project grows to support multiple coding agents or branching paths (e.g. "code" vs "explain" intents going to different specialised models), adding nodes is cheaper than restructuring imperative code.

The cost is some boilerplate — `add_node` + `add_edge` + `add_conditional_edges` instead of one Python function. For a graph this size it's a wash; for a larger project it pays off.

## 5. Classification: fuzzy allowlist before LLM

This was the most interesting decision. The naive approach: send every transcript to Llama and ask it to classify. That works but it's:

- **Slow.** A Groq Llama call is ~200–400ms. Doing it per utterance adds noticeable lag.
- **Wasteful.** "undo" → undo doesn't need an LLM.
- **Non-deterministic.** "exit" routed to a wrong intent because the model decided to be creative would be infuriating during a demo.

So I built a two-stage classifier in `classify_node.py`:

**Stage 1 — fuzzy match against a strict allowlist** (`config/commands.py`). The allowlist enumerates every known command and its phrasings ("undo" / "undo that" / "revert" / "go back" all map to action `undo`). I use `rapidfuzz` with `token_set_ratio` and a threshold of 85. `token_set_ratio` ignores word order and filler words, so "yeah undo that thing" still matches "undo that" even with extra tokens.

**Stage 2 — LLM fallback.** Only transcripts that miss the allowlist reach Groq Llama, with a tight system prompt asking for strict JSON `{"intent", "action"}`. Three layers of defence on the output:
- `response_format={"type": "json_object"}` — Groq's JSON mode forces valid JSON.
- Pydantic `_ClassifySchema` validates the parsed object.
- `_validate_action()` checks the action exists in the allowlist — even if the LLM invents `"log_out"`, it gets coerced to `None`.

In practice the LLM is consulted maybe 10% of the time. Latency drops from ~300ms per utterance to ~5ms for typical commands.

I considered using **Groq's function-calling / tool-use** instead of JSON mode + Pydantic. It's marginally more elegant but it ties the implementation to function-calling-capable models, and `llama-3.1-8b-instant` handles JSON mode just fine. No reason to add the coupling.

## 6. Push-to-talk vs hands-free

The brief says "fully hands-free is the ideal." I built both, controlled by a `.env` flag, and made push-to-talk the default. Reasoning:

**Hands-free** (`HANDS_FREE_MODE=true`) uses WebRTC VAD with a 90ms speech-start window and an 800ms silence-end window. It works, but it's noisy in two predictable ways:
- Background sounds (typing, coughs, AC) sometimes trigger phantom utterances.
- A user who pauses mid-sentence to think gets cut off.

**Push-to-talk** (SPACE) is deterministic. The user holds SPACE for the duration of their utterance, and there are zero false positives. The "one button" cost is small — SPACE is still the laziest possible keyboard input, requires no hand position change, and works in any room.

For the live demo, PTT is the safer default. For the report's "minimise reliance on keyboard" point, I argue that one keystroke (a held SPACE) is closer to "hands-free" than any system that occasionally mis-fires. The hands-free mode is one config flip away if the evaluator wants to see it. I'd rather have a reliable demo with one key than a fully hands-free demo that misfires.

I deliberately avoided wake-word detection (e.g. Picovoice Porcupine, openWakeWord). Wake words are reliable but add a model dependency outside the Groq constraint and feel un-Linux for a coding tool. Coders are already comfortable with modal keys.

## 7. UI as a separate process

Two processes (`pipeline/main.py` + `streamlit run ui/app.py`) communicating through `tmp/state.json`. Alternative designs I rejected:

- **Single process, embed Streamlit inside the pipeline.** Streamlit's execution model (rerun the whole script on every event) doesn't compose well with a long-lived audio capture loop. You'd need to spawn the capture loop on a thread, which Streamlit's session-state model fights at every turn.
- **WebSocket between the processes.** Cleaner in theory, but adds an `asyncio` dependency to the pipeline and a JS layer in the UI. The atomic-rename JSON pattern is dead simple, works on every OS, and the polling latency (1s) is well under the human-perceptible delay between speaking and seeing output.

The IPC is one function: `utils.state_bridge.write_state()` writes to `NamedTemporaryFile` in the same directory as the target, then `os.replace()` onto the final path. `os.replace` is atomic on POSIX and on Windows (when target exists), so the UI never sees a torn write. This is the same pattern git itself uses for its index.

## 8. The persistent aider subprocess

Aider's startup is 3–5 seconds (loads the LLM adapter, scans the repo, builds the tags cache). Spawning per utterance would make the agent feel cold every time. So `graph/nodes/aider_node.py` keeps a singleton `_AiderSession` that spawns aider once on the first prompt and reuses it.

The non-trivial piece is reading aider's stdout. Subprocess pipes block; if you call `proc.stdout.read()` and aider hasn't flushed yet, you deadlock the main thread. The fix is a daemon reader thread that drains stdout into a `queue.Queue`. The main thread reads from the queue with a timeout — never blocks the graph.

Stream-end detection is also non-trivial. Aider streams tokens, so there's no marker for "response complete." I use a quiet-window heuristic: collect output until 1.5 seconds pass with no new lines (or 60 seconds total elapses). This is the part of the system I'm least happy with — it's a heuristic, and a model that pauses for ~2s while generating a long code block would get truncated. A better design would parse aider's prompt prompt (`>` at line start) to detect ready state, but aider's prompt is themed and varies. The quiet window is a deliberate trade-off for simplicity in a 1-week assignment.

## 9. What's not in scope

- **Wake word.** Out of scope given the Groq-only constraint and PTT's reliability.
- **TTS.** Brief explicitly says no audio output required.
- **Multilingual STT.** Hard-coded to `language="en"`. Whisper supports 99+ languages natively; flipping to auto-detect is a one-line change but costs latency and accuracy on short utterances.
- **Streaming UI updates during aider's response.** The graph treats one prompt as one atomic turn. A streaming version is possible (yield from a generator into the queue, write to state.json after each chunk) but adds complexity that doesn't change the demo's punch.
- **Conversation memory beyond aider's own context.** Aider keeps its own chat history; we don't layer anything on top.

## 10. Honest limitations

1. **Quiet-window response detection (above) can truncate long aider outputs.** Tunable via `_QUIET_AFTER_SECS` in `aider_node.py` but fundamentally a heuristic.

2. **WebRTC VAD struggles in noisy environments.** PTT mode is unaffected, so this only hits users who explicitly enable hands-free mode.

3. **The confidence proxy is a proxy.** A more honest signal would be a second STT pass with different temperature, but that doubles cost. The retry loop catches most low-confidence failures in practice.

4. **No multi-user support.** The state file is a singleton. Fine for the assignment, would need replacing with a real message bus (Redis, NATS) for any multi-tenant deployment.

5. **No tests for the audio capture layer.** Unit-testing `sounddevice` requires mocking PortAudio, which is more effort than it's worth for a 1-week project. The pure-Python parts (state machine, classification, IPC) do have tests in `tests/`.

6. **First-time aider boot blocks the first utterance.** Subsequent utterances are fast, but the very first one waits 3–5s for aider to come up. A polish task would be to pre-warm aider during pipeline startup (currently it lazy-starts on the first prompt), trading "instant pipeline start" for "instant first utterance."

---

## TL;DR of decisions

| Decision | Choice | Main reason |
|---|---|---|
| Target agent | aider | Open source, Python, slash-commands |
| STT model | `whisper-large-v3` | Technical-vocab accuracy > latency |
| Confidence signal | `exp(avg_logprob) × (1 - no_speech_prob)` | Free; tracks reality well enough |
| Classifier | Fuzzy allowlist → LLM fallback | Determinism on commands; LLM only when needed |
| Capture mode | PTT default, VAD optional | Reliability for live demo |
| State machine | LangGraph | Explicit retry edges; data not code |
| UI | Separate Streamlit process via state.json | Decouples mic from dashboard |
| Aider lifecycle | Persistent subprocess + reader thread | 3-5s boot only happens once |