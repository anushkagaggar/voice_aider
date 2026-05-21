"""
Voice capture + VAD + main loop.

This package is intentionally a namespace shell — sub-modules are imported
directly (e.g. `from pipeline.capture import MicCapture`) rather than
re-exported here, so that importing `pipeline.vad` alone doesn't pull in
`sounddevice`/PortAudio (which `pipeline.capture` requires).
"""