"""Optional server-side speech-to-text, backing the frontend mic button.

The mic button's default engine is the browser's own Web Speech API (free, no
backend). This module is the *fallback*: when a browser lacks Web Speech (or the
operator opts in), the frontend records audio and POSTs it to ``/api/transcribe``
(see ``nanny/server.py``), which calls Google Cloud Speech-to-Text here.

Gated behind ``NANNY_STT_ENABLED`` — off (the default, and every local/test/
sandbox run) nothing here touches Google Cloud and the endpoint reports the
feature disabled. On, ``transcribe`` calls the Speech-to-Text API with the
deployment's service-account credentials (ADC), which is why it can't run
without GCP credentials. ``google-cloud-speech`` is only imported inside
``transcribe`` (the ``speech`` extra), so this module imports fine without it.
"""

from __future__ import annotations

import os


def stt_enabled() -> bool:
    """The single flag the transcribe endpoints check before touching GCP."""
    return os.environ.get("NANNY_STT_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def transcribe(audio: bytes, mime: str = "audio/webm") -> str:
    """Transcribes recorded audio to text; returns the top result, or "".

    Expects the browser's ``MediaRecorder`` output (WebM/Opus); Speech-to-Text
    reads the sample rate from the container header, so none is passed.
    """
    from google.cloud import speech

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    response = client.recognize(
        config=config, audio=speech.RecognitionAudio(content=audio)
    )
    for result in response.results:
        if result.alternatives:
            return result.alternatives[0].transcript.strip()
    return ""
