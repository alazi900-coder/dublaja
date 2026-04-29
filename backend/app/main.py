"""
Dublaja — Free voice dubbing & TTS backend.

Features:
- /api/tts        : text -> MP3 (gTTS, supports Arabic + many languages)
- /api/translate  : translate text between languages (deep-translator / Google)
- /api/dub        : uploaded audio -> transcribe -> translate -> TTS in target language
- /api/effect     : uploaded audio -> apply character voice effect
- /api/languages  : supported language codes for each engine
- /api/voices     : supported TTS "voice" presets
"""

from __future__ import annotations

import io
import os
import tempfile
import uuid
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from gtts import gTTS
from gtts.lang import tts_langs
from deep_translator import GoogleTranslator
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import speedup


app = FastAPI(title="Dublaja API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class TTSRequest(BaseModel):
    text: str
    lang: str = "ar"
    slow: bool = False
    tld: str = "com"  # accent: com, co.uk, com.au, ca, co.in, ie, co.za


class TranslateRequest(BaseModel):
    text: str
    source: str = "auto"
    target: str = "ar"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Map UI voice presets -> (gTTS lang, tld-for-accent, slow flag, post-effect)
# gTTS itself doesn't give character voices; we approximate via tld (accent)
# and optional post-processing pitch/speed.
VOICE_PRESETS: dict[str, dict] = {
    "arabic_default":    {"lang": "ar", "tld": "com",     "slow": False, "effect": None},
    "arabic_slow":       {"lang": "ar", "tld": "com",     "slow": True,  "effect": None},
    "english_us":        {"lang": "en", "tld": "com",     "slow": False, "effect": None},
    "english_uk":        {"lang": "en", "tld": "co.uk",   "slow": False, "effect": None},
    "english_australia": {"lang": "en", "tld": "com.au",  "slow": False, "effect": None},
    "english_india":     {"lang": "en", "tld": "co.in",   "slow": False, "effect": None},
    "french":            {"lang": "fr", "tld": "fr",      "slow": False, "effect": None},
    "spanish":           {"lang": "es", "tld": "es",      "slow": False, "effect": None},
    "german":            {"lang": "de", "tld": "de",      "slow": False, "effect": None},
    "turkish":           {"lang": "tr", "tld": "com.tr",  "slow": False, "effect": None},
    "japanese":          {"lang": "ja", "tld": "co.jp",   "slow": False, "effect": None},
    # Character voices = language + post effect
    "character_robot":   {"lang": "ar", "tld": "com",     "slow": False, "effect": "robot"},
    "character_monster": {"lang": "ar", "tld": "com",     "slow": False, "effect": "monster"},
    "character_child":   {"lang": "ar", "tld": "com",     "slow": False, "effect": "child"},
    "character_giant":   {"lang": "ar", "tld": "com",     "slow": False, "effect": "giant"},
    "character_echo":    {"lang": "ar", "tld": "com",     "slow": False, "effect": "echo"},
}


EFFECTS: list[str] = ["robot", "monster", "child", "giant", "echo", "chipmunk", "demon"]


def apply_effect(audio: AudioSegment, effect: str) -> AudioSegment:
    """Apply a character-voice effect to an AudioSegment."""
    if effect == "robot":
        # Lower pitch + add slight echo
        shifted = _pitch_shift(audio, -3)
        return shifted.overlay(shifted - 8, position=80)
    if effect == "monster" or effect == "demon":
        # Very low, slow
        shifted = _pitch_shift(audio, -7)
        return shifted
    if effect == "child" or effect == "chipmunk":
        # Higher pitch
        return _pitch_shift(audio, +6)
    if effect == "giant":
        # Very low pitch
        return _pitch_shift(audio, -10)
    if effect == "echo":
        out = audio
        for delay_ms, atten in [(180, 6), (360, 12), (540, 18)]:
            out = out.overlay(audio - atten, position=delay_ms)
        return out
    return audio


def _pitch_shift(audio: AudioSegment, semitones: int) -> AudioSegment:
    """Shift pitch by `semitones` without changing length (approx via frame rate + resample)."""
    new_rate = int(audio.frame_rate * (2.0 ** (semitones / 12.0)))
    shifted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
    return shifted.set_frame_rate(audio.frame_rate)


def synthesize(text: str, lang: str, tld: str = "com", slow: bool = False,
               effect: str | None = None) -> bytes:
    """Run gTTS and return MP3 bytes (optionally with effect applied)."""
    if not text.strip():
        raise HTTPException(400, "Text is empty")

    tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)

    if not effect:
        return buf.read()

    audio = AudioSegment.from_file(buf, format="mp3")
    audio = apply_effect(audio, effect)
    out = io.BytesIO()
    audio.export(out, format="mp3")
    out.seek(0)
    return out.read()


def transcribe(file_bytes: bytes, filename: str, language: str | None) -> str:
    """Transcribe audio using Google's free Speech Recognition (no API key)."""
    # Convert to WAV via pydub+ffmpeg because SpeechRecognition wants PCM WAV
    ext = os.path.splitext(filename)[1].lower().lstrip(".") or "mp3"
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=ext)
    except Exception:
        # Fallback: let ffmpeg auto-detect
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))

    # Resample to 16kHz mono for STT
    audio = audio.set_frame_rate(16000).set_channels(1)

    tmp_path = os.path.join(tempfile.gettempdir(), f"dublaja_{uuid.uuid4().hex}.wav")
    audio.export(tmp_path, format="wav")

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        lang_tag = _to_stt_locale(language) if language else None
        if lang_tag:
            text = recognizer.recognize_google(audio_data, language=lang_tag)
        else:
            text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        raise HTTPException(422, "لم نتمكن من فهم الصوت في الملف. جرّب ملفاً أوضح.")
    except sr.RequestError as e:
        raise HTTPException(
            502,
            f"خدمة التعرّف على الصوت غير متاحة حالياً: {e}. "
            "حاول مرة أخرى بعد قليل.",
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _to_stt_locale(lang: str) -> str:
    """Map 2-letter lang codes to Google STT locale tags."""
    mapping = {
        "ar": "ar-SA",
        "en": "en-US",
        "fr": "fr-FR",
        "es": "es-ES",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-PT",
        "ru": "ru-RU",
        "tr": "tr-TR",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
        "hi": "hi-IN",
        "ur": "ur-PK",
    }
    return mapping.get(lang, lang)


def translate_text(text: str, source: str, target: str) -> str:
    if not text.strip():
        return text
    try:
        return GoogleTranslator(source=source or "auto", target=target).translate(text)
    except Exception as e:
        raise HTTPException(502, f"فشل الترجمة: {e}")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
def root():
    return {"name": "Dublaja API", "status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/languages")
def languages():
    """Return supported gTTS languages + suggested source STT languages."""
    return JSONResponse({
        "tts": tts_langs(),  # {"ar": "Arabic", "en": "English", ...}
        "stt": {
            "ar": "العربية",
            "en": "English",
            "fr": "Français",
            "es": "Español",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "tr": "Türkçe",
            "ja": "日本語",
            "ko": "한국어",
            "zh": "中文",
            "hi": "हिन्दी",
            "ur": "اردو",
        },
    })


@app.get("/api/voices")
def voices():
    return {
        "voices": [
            {"id": k, "lang": v["lang"], "effect": v.get("effect")}
            for k, v in VOICE_PRESETS.items()
        ],
        "effects": EFFECTS,
    }


@app.post("/api/tts")
def api_tts(req: TTSRequest):
    data = synthesize(req.text, req.lang, req.tld, req.slow, None)
    return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg")


@app.post("/api/tts/voice")
def api_tts_voice(text: str = Form(...), voice: str = Form("arabic_default")):
    preset = VOICE_PRESETS.get(voice)
    if not preset:
        raise HTTPException(400, f"صوت غير معروف: {voice}")
    data = synthesize(
        text,
        lang=preset["lang"],
        tld=preset["tld"],
        slow=preset["slow"],
        effect=preset["effect"],
    )
    return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg")


@app.post("/api/translate")
def api_translate(req: TranslateRequest):
    return {"translation": translate_text(req.text, req.source, req.target)}


@app.post("/api/dub")
async def api_dub(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("ar"),
    voice: str | None = Form(None),
):
    """
    Upload audio -> transcribe -> translate -> synthesize in target language.
    Returns JSON with transcript, translation, and a URL-safe MP3 (base64) + direct endpoint.
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "ملف فارغ")

    # STT
    stt_lang = None if source_lang in (None, "", "auto") else source_lang
    transcript = transcribe(content, file.filename or "audio.mp3", stt_lang)

    # Translate
    if target_lang and target_lang != source_lang:
        translated = translate_text(
            transcript,
            source=source_lang if source_lang and source_lang != "auto" else "auto",
            target=target_lang,
        )
    else:
        translated = transcript

    # TTS
    if voice and voice in VOICE_PRESETS:
        p = VOICE_PRESETS[voice]
        mp3 = synthesize(translated, p["lang"], p["tld"], p["slow"], p["effect"])
    else:
        mp3 = synthesize(translated, lang=target_lang)

    import base64
    return {
        "transcript": transcript,
        "translation": translated,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "audio_base64": base64.b64encode(mp3).decode("ascii"),
        "mime": "audio/mpeg",
    }


@app.post("/api/effect")
async def api_effect(
    file: UploadFile = File(...),
    effect: str = Form("robot"),
):
    """Apply a character-voice effect to an uploaded audio file."""
    if effect not in EFFECTS:
        raise HTTPException(400, f"تأثير غير معروف: {effect}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "ملف فارغ")

    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".") or "mp3"
    try:
        audio = AudioSegment.from_file(io.BytesIO(content), format=ext)
    except Exception:
        audio = AudioSegment.from_file(io.BytesIO(content))

    out_audio = apply_effect(audio, effect)
    out = io.BytesIO()
    out_audio.export(out, format="mp3")
    out.seek(0)
    return StreamingResponse(out, media_type="audio/mpeg")
