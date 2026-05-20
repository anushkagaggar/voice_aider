"""Groq API clients — STT (Whisper) and intent classification (Llama)."""
from groq_clients.llm_client import ClassificationResult, classify
from groq_clients.whisper_client import TranscriptionResult, transcribe

__all__ = [
    "transcribe",
    "TranscriptionResult",
    "classify",
    "ClassificationResult",
]