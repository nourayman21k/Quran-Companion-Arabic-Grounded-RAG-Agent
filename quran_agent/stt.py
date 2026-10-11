"""Speech-to-text with Whisper large-v3 hosted on Groq (same GROQ_API_KEY as the LLM)."""
from groq import Groq

from . import config

LANGUAGES = {"Auto-detect": None, "Arabic": "ar", "English": "en"}   # ISO-639-1 codes for Whisper


class Transcriber:
    def __init__(self, model: str = config.WHISPER_MODEL):
        self.client = Groq(max_retries=config.LLM_MAX_RETRIES)   # reads GROQ_API_KEY
        self.model = model

    def transcribe(self, audio_bytes: bytes, language: str | None = None, filename: str = "audio.wav") -> str:
        """Transcribe wav/mp3/flac/ogg/m4a/webm bytes (max 25 MB). `language` = 'ar' / 'en' / None (auto-detect)."""
        kwargs = {"language": language} if language else {}
        result = self.client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=self.model,
            response_format="json",
            temperature=0.0,
            **kwargs,
        )
        return result.text.strip()
