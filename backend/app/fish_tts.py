"""
Fish Audio TTS integration — premium-quality Arabic + voice cloning.

API docs: https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech
- POST https://api.fish.audio/v1/tts
- Auth: Bearer <FISH_AUDIO_API_KEY>
- Header `model: s2-pro` (best quality) or `s1`
- Built-in voice: pass `reference_id` (model UUID from fish.audio gallery)
- Zero-shot cloning: pass `references=[{audio: bytes, text: transcript}]` — REQUIRES msgpack
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
import msgpack


FISH_API_URL = "https://api.fish.audio/v1/tts"


class FishAudioError(Exception):
    """Fish Audio API error with HTTP status + message."""

    def __init__(self, status: int, message: str):
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.message = message


def _get_key() -> str:
    key = os.environ.get("FISH_AUDIO_API_KEY", "").strip()
    if not key:
        raise FishAudioError(500, "FISH_AUDIO_API_KEY not configured on backend.")
    return key


def synthesize_fish(
    text: str,
    *,
    reference_id: Optional[str] = None,
    reference_audio: Optional[bytes] = None,
    reference_transcript: Optional[str] = None,
    model: str = "s2-pro",
    temperature: float = 0.7,
    top_p: float = 0.7,
    speed: float = 1.0,
    volume: float = 0.0,
    chunk_length: int = 300,
    sample_rate: int = 44100,
    mp3_bitrate: int = 128,
    latency: str = "normal",
    timeout: float = 120.0,
) -> bytes:
    """
    Generate MP3 bytes from text using Fish Audio.

    Modes:
      1. Built-in voice: pass `reference_id` (model UUID).
      2. Zero-shot cloning: pass `reference_audio` (raw bytes) + `reference_transcript`.
      3. Default voice: pass neither (uses Fish Audio default voice).
    """
    if not text.strip():
        raise FishAudioError(400, "Text is empty.")

    payload: dict = {
        "text": text,
        "format": "mp3",
        "mp3_bitrate": mp3_bitrate,
        "sample_rate": sample_rate,
        "latency": latency,
        "temperature": temperature,
        "top_p": top_p,
        "chunk_length": chunk_length,
        "normalize": True,
        "prosody": {
            "speed": speed,
            "volume": volume,
            "normalize_loudness": True,
        },
    }

    use_msgpack = False

    if reference_id:
        payload["reference_id"] = reference_id

    if reference_audio:
        if not reference_transcript:
            raise FishAudioError(
                400,
                "reference_transcript is required when using reference_audio (zero-shot cloning).",
            )
        payload["references"] = [
            {"audio": reference_audio, "text": reference_transcript}
        ]
        use_msgpack = True  # msgpack is required for inline binary audio

    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "model": model,
    }

    if use_msgpack:
        body = msgpack.packb(payload, use_bin_type=True)
        headers["Content-Type"] = "application/msgpack"
    else:
        import json

        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(FISH_API_URL, content=body, headers=headers)

    if resp.status_code != 200:
        try:
            data = resp.json()
            msg = data.get("message") or data.get("detail") or resp.text
        except Exception:
            msg = resp.text or f"HTTP {resp.status_code}"
        raise FishAudioError(resp.status_code, str(msg))

    return resp.content


# --------------------------------------------------------------------------- #
# Game character voice presets — Fish Audio model IDs
# --------------------------------------------------------------------------- #
# Map app voice names -> {model_id, model, params}
# The user provided their preferred sample voice; Zelda character mapping uses
# the same model with prosody variations until per-character voice IDs are added.

# User-discovered voice from fish.audio (Arabic, deep storyteller tone)
USER_PREFERRED_AR_VOICE = "1f9388401ff04902894a7cbefdc0cc2d"

FISH_VOICE_PRESETS: dict[str, dict] = {
    # Default / direct
    "fish_arabic_default": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 1.0,
        "temperature": 0.7,
    },
    # Zelda heroes
    "fish_zelda_link": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 1.05,
        "temperature": 0.75,
    },
    "fish_zelda_zelda": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 0.95,
        "temperature": 0.7,
    },
    "fish_zelda_ganon": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 0.85,
        "temperature": 0.65,
    },
    "fish_zelda_rauru": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 0.9,
        "temperature": 0.7,
    },
    "fish_zelda_sheik": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 1.0,
        "temperature": 0.8,
    },
    # Generic game archetypes
    "fish_game_hero": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 1.05,
        "temperature": 0.75,
    },
    "fish_game_villain": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 0.85,
        "temperature": 0.65,
    },
    "fish_game_narrator": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 0.95,
        "temperature": 0.7,
    },
    "fish_game_wise": {
        "reference_id": USER_PREFERRED_AR_VOICE,
        "model": "s2-pro",
        "speed": 0.9,
        "temperature": 0.7,
    },
}
