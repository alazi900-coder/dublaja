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

# Make sure ffmpeg + ffprobe are on PATH (works even when system packages are not available).
try:
    import static_ffmpeg  # type: ignore
    static_ffmpeg.add_paths(weak=True)  # downloads/caches the binaries on first call
except Exception as _e:  # pragma: no cover
    print(f"[warn] static_ffmpeg unavailable: {_e}")

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
# and optional post-processing pitch/speed/echo/distortion.
VOICE_PRESETS: dict[str, dict] = {
    # --- Base languages & accents ---
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

    # --- Game character voices (Arabic base) ---
    # Heroes / warriors
    "game_hero":         {"lang": "ar", "tld": "com",     "slow": False, "effect": "hero"},
    "game_warrior":      {"lang": "ar", "tld": "com",     "slow": False, "effect": "warrior"},
    "game_knight":       {"lang": "ar", "tld": "com",     "slow": False, "effect": "knight"},
    "game_samurai":      {"lang": "ar", "tld": "com",     "slow": False, "effect": "samurai"},
    # Villains / dark
    "game_villain":      {"lang": "ar", "tld": "com",     "slow": False, "effect": "villain"},
    "game_demon":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "demon"},
    "game_dark_lord":    {"lang": "ar", "tld": "com",     "slow": True,  "effect": "dark_lord"},
    # Creatures
    "game_dragon":       {"lang": "ar", "tld": "com",     "slow": True,  "effect": "dragon"},
    "game_orc":          {"lang": "ar", "tld": "com",     "slow": False, "effect": "orc"},
    "game_goblin":       {"lang": "ar", "tld": "com",     "slow": False, "effect": "goblin"},
    "game_troll":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "troll"},
    "game_monster":      {"lang": "ar", "tld": "com",     "slow": False, "effect": "monster"},
    "game_giant":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "giant"},
    "game_zombie":       {"lang": "ar", "tld": "com",     "slow": True,  "effect": "zombie"},
    # Magic
    "game_wizard":       {"lang": "ar", "tld": "com",     "slow": False, "effect": "wizard"},
    "game_witch":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "witch"},
    "game_fairy":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "fairy"},
    "game_ghost":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "ghost"},
    # NPC / roles
    "game_announcer":    {"lang": "ar", "tld": "com",     "slow": False, "effect": "announcer"},
    "game_narrator":     {"lang": "ar", "tld": "com",     "slow": False, "effect": "narrator"},
    "game_old_wise":     {"lang": "ar", "tld": "com",     "slow": True,  "effect": "old_wise"},
    "game_child":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "child"},
    # Sci-fi / tech
    "game_robot":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "robot"},
    "game_alien":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "alien"},
    "game_ai":           {"lang": "ar", "tld": "com",     "slow": False, "effect": "ai"},
    "game_radio":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "radio"},
    # Elemental
    "game_fire_spirit":  {"lang": "ar", "tld": "com",     "slow": False, "effect": "fire"},
    "game_ice_spirit":   {"lang": "ar", "tld": "com",     "slow": True,  "effect": "ice"},
    "game_underwater":   {"lang": "ar", "tld": "com",     "slow": False, "effect": "underwater"},

    # --- Zelda series voice pack (Arabic base, tuned per character) ---
    # Heroes & royalty
    "zelda_link":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_link"},
    "zelda_zelda":       {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_zelda"},
    "zelda_king":        {"lang": "ar", "tld": "com",     "slow": True,  "effect": "z_king"},
    "zelda_impa":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_impa"},
    "zelda_rauru":       {"lang": "ar", "tld": "com",     "slow": True,  "effect": "z_rauru"},
    "zelda_sheik":       {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_sheik"},
    # Villains
    "zelda_ganon":       {"lang": "ar", "tld": "com",     "slow": True,  "effect": "z_ganon"},
    "zelda_demise":      {"lang": "ar", "tld": "com",     "slow": True,  "effect": "z_demise"},
    "zelda_yiga":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_yiga"},
    "zelda_skull_kid":   {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_skull_kid"},
    # Races / tribes
    "zelda_goron":       {"lang": "ar", "tld": "com",     "slow": True,  "effect": "z_goron"},
    "zelda_zora":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_zora"},
    "zelda_rito":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_rito"},
    "zelda_gerudo":      {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_gerudo"},
    "zelda_korok":       {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_korok"},
    "zelda_deku":        {"lang": "ar", "tld": "com",     "slow": True,  "effect": "z_deku"},
    # Spirits / guides
    "zelda_fi":          {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_fi"},
    "zelda_midna":       {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_midna"},
    "zelda_navi":        {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_navi"},
    "zelda_guardian":    {"lang": "ar", "tld": "com",     "slow": False, "effect": "z_guardian"},
    "zelda_red_lions":   {"lang": "ar", "tld": "com",     "slow": True,  "effect": "z_red_lions"},
}


EFFECTS: list[str] = [
    # Legacy / simple effects
    "robot", "monster", "child", "giant", "echo", "chipmunk", "demon",
    # Game character effects
    "hero", "warrior", "knight", "samurai",
    "villain", "dark_lord",
    "dragon", "orc", "goblin", "troll", "zombie",
    "wizard", "witch", "fairy", "ghost",
    "announcer", "narrator", "old_wise",
    "alien", "ai", "radio",
    "fire", "ice", "underwater",
    # Zelda pack
    "z_link", "z_zelda", "z_king", "z_impa", "z_rauru", "z_sheik",
    "z_ganon", "z_demise", "z_yiga", "z_skull_kid",
    "z_goron", "z_zora", "z_rito", "z_gerudo", "z_korok", "z_deku",
    "z_fi", "z_midna", "z_navi", "z_guardian", "z_red_lions",
]


def _add_echo(audio: AudioSegment, delays_atten: list[tuple[int, int]]) -> AudioSegment:
    out = audio
    for delay_ms, atten in delays_atten:
        out = out.overlay(audio - atten, position=delay_ms)
    return out


def _speedup(audio: AudioSegment, factor: float) -> AudioSegment:
    """Change duration by `factor` while keeping pitch (approx)."""
    if factor == 1.0:
        return audio
    new_rate = int(audio.frame_rate * factor)
    faster = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
    return faster.set_frame_rate(audio.frame_rate)


def _lowpass(audio: AudioSegment, cutoff: int) -> AudioSegment:
    try:
        return audio.low_pass_filter(cutoff)
    except Exception:
        return audio


def _highpass(audio: AudioSegment, cutoff: int) -> AudioSegment:
    try:
        return audio.high_pass_filter(cutoff)
    except Exception:
        return audio


def apply_effect(audio: AudioSegment, effect: str) -> AudioSegment:
    """Apply a character-voice effect to an AudioSegment."""
    e = effect.lower()

    # ---------- Basic / legacy ----------
    if e == "robot":
        shifted = _pitch_shift(audio, -3)
        metallic = _add_echo(shifted, [(60, 4), (120, 10)])
        return _highpass(metallic, 300)
    if e == "monster":
        shifted = _pitch_shift(audio, -7)
        return _add_echo(shifted, [(140, 6)])
    if e == "child" or e == "chipmunk":
        return _pitch_shift(audio, +6)
    if e == "giant":
        shifted = _pitch_shift(audio, -10)
        return _add_echo(shifted, [(180, 6), (320, 10)])
    if e == "echo":
        return _add_echo(audio, [(180, 6), (360, 12), (540, 18)])
    if e == "demon":
        shifted = _pitch_shift(audio, -9)
        return _add_echo(shifted, [(90, 4), (220, 10), (400, 18)])

    # ---------- Heroes / warriors ----------
    if e == "hero":
        # Slight low pitch + presence + subtle echo
        shifted = _pitch_shift(audio, -2)
        presence = shifted + 2  # +2 dB
        return _add_echo(presence, [(120, 16)])
    if e == "warrior":
        shifted = _pitch_shift(audio, -3)
        return _add_echo(shifted + 1, [(100, 10), (200, 16)])
    if e == "knight":
        # Metallic, regal — HP filter + echo
        shifted = _pitch_shift(audio, -2)
        return _highpass(_add_echo(shifted, [(150, 12)]), 200)
    if e == "samurai":
        shifted = _pitch_shift(audio, -2)
        return _add_echo(shifted, [(200, 14), (420, 20)])

    # ---------- Villains / dark ----------
    if e == "villain":
        shifted = _pitch_shift(audio, -5)
        return _add_echo(shifted, [(80, 6), (160, 12), (280, 18)])
    if e == "dark_lord":
        shifted = _pitch_shift(audio, -8)
        layered = _add_echo(shifted, [(120, 4), (260, 10), (420, 16)])
        return _lowpass(layered, 2800)

    # ---------- Creatures ----------
    if e == "dragon":
        shifted = _pitch_shift(audio, -9)
        growly = _add_echo(shifted, [(60, 4), (140, 8), (300, 14)])
        return _lowpass(growly, 2500)
    if e == "orc":
        shifted = _pitch_shift(audio, -6)
        return _add_echo(shifted - 1, [(70, 6), (180, 12)])
    if e == "goblin":
        shifted = _pitch_shift(audio, +4)
        return _add_echo(shifted, [(50, 5), (100, 10)])
    if e == "troll":
        shifted = _pitch_shift(audio, -8)
        return _add_echo(shifted, [(180, 8)])
    if e == "zombie":
        shifted = _pitch_shift(audio, -4)
        muffled = _lowpass(shifted, 2000)
        return _add_echo(muffled, [(220, 10), (440, 18)])

    # ---------- Magic ----------
    if e == "wizard":
        shifted = _pitch_shift(audio, -1)
        return _add_echo(shifted, [(180, 6), (380, 12), (560, 20)])
    if e == "witch":
        shifted = _pitch_shift(audio, +3)
        cackle = _add_echo(shifted, [(120, 8), (260, 14)])
        return _highpass(cackle, 400)
    if e == "fairy":
        shifted = _pitch_shift(audio, +5)
        sparkly = _add_echo(shifted, [(90, 6), (180, 12)])
        return _highpass(sparkly, 500)
    if e == "ghost":
        shifted = _pitch_shift(audio, -2)
        whispered = _add_echo(shifted - 2, [(140, 4), (300, 10), (520, 16), (800, 22)])
        return _highpass(whispered, 250)

    # ---------- NPC / roles ----------
    if e == "announcer":
        loud = audio + 4
        return _add_echo(loud, [(160, 18)])
    if e == "narrator":
        shifted = _pitch_shift(audio, -1)
        return shifted + 1
    if e == "old_wise":
        shifted = _pitch_shift(audio, -2)
        return _lowpass(shifted, 3200)

    # ---------- Sci-fi / tech ----------
    if e == "alien":
        shifted = _pitch_shift(audio, +3)
        warbled = _add_echo(shifted, [(40, 4), (90, 8), (160, 12)])
        return _highpass(warbled, 300)
    if e == "ai":
        shifted = _pitch_shift(audio, -1)
        cold = _add_echo(shifted, [(60, 6), (140, 14)])
        return _highpass(cold, 250)
    if e == "radio":
        # Walkie-talkie: bandpass + crackle via highpass+lowpass cascade
        hp = _highpass(audio, 400)
        lp = _lowpass(hp, 3000)
        return lp + 2

    # ---------- Elemental ----------
    if e == "fire":
        shifted = _pitch_shift(audio, -3)
        return _add_echo(shifted, [(50, 5), (110, 10), (200, 16)])
    if e == "ice":
        shifted = _pitch_shift(audio, +2)
        cold = _add_echo(shifted, [(180, 10), (360, 18), (560, 26)])
        return _highpass(cold, 350)
    if e == "underwater":
        muffled = _lowpass(audio, 1200)
        return _add_echo(muffled, [(120, 10), (240, 16)])

    # ---------- Zelda character pack ----------
    # Link — young hero, slight deeper, a bit of warmth
    if e == "z_link":
        shifted = _pitch_shift(audio, -1)
        return _add_echo(shifted + 1, [(140, 20)])
    # Princess Zelda — neutral, noble, light echo
    if e == "z_zelda":
        return _add_echo(audio, [(160, 18), (320, 26)])
    # King Rhoam / King of Hyrule — deep, authoritative, hall echo
    if e == "z_king":
        shifted = _pitch_shift(audio, -5)
        return _add_echo(shifted + 1, [(140, 10), (320, 16), (540, 22)])
    # Impa — strong, firm, older
    if e == "z_impa":
        shifted = _pitch_shift(audio, -1)
        return _add_echo(shifted, [(110, 16)])
    # Rauru — old wise sage
    if e == "z_rauru":
        shifted = _pitch_shift(audio, -4)
        low = _lowpass(shifted, 3000)
        return _add_echo(low, [(200, 14), (420, 22)])
    # Sheik — mysterious, airy, with reverb + highpass
    if e == "z_sheik":
        hp = _highpass(audio, 350)
        return _add_echo(hp, [(140, 12), (300, 20), (500, 28)])

    # Ganondorf — very deep, menacing, heavy reverb, lowpass
    if e == "z_ganon":
        shifted = _pitch_shift(audio, -9)
        dark = _add_echo(shifted, [(100, 4), (240, 10), (420, 16), (640, 22)])
        return _lowpass(dark, 2600)
    # Demise — ancient demon king, deeper than Ganon
    if e == "z_demise":
        shifted = _pitch_shift(audio, -10)
        dark = _add_echo(shifted, [(120, 4), (260, 10), (460, 16), (700, 24)])
        return _lowpass(dark, 2400)
    # Yiga Clan — sneaky, quick, dramatic
    if e == "z_yiga":
        shifted = _pitch_shift(audio, -2)
        return _add_echo(shifted + 1, [(70, 6), (160, 12)])
    # Skull Kid — creepy childlike echo
    if e == "z_skull_kid":
        shifted = _pitch_shift(audio, +4)
        eerie = _add_echo(shifted, [(160, 10), (340, 18), (560, 26)])
        return _highpass(eerie, 300)

    # Goron — deep, rocky, slow rumble
    if e == "z_goron":
        shifted = _pitch_shift(audio, -7)
        return _add_echo(shifted + 1, [(80, 5), (180, 10)])
    # Zora — water-tinged smooth voice
    if e == "z_zora":
        watery = _lowpass(audio, 2200)
        return _add_echo(watery, [(140, 10), (300, 18)])
    # Rito — bird people, higher-pitched, airy
    if e == "z_rito":
        shifted = _pitch_shift(audio, +3)
        return _add_echo(shifted, [(80, 8), (180, 14)])
    # Gerudo — strong, desert-edge female
    if e == "z_gerudo":
        shifted = _pitch_shift(audio, -1)
        return _add_echo(shifted + 1, [(120, 14)])
    # Korok — tiny playful forest spirits
    if e == "z_korok":
        shifted = _pitch_shift(audio, +6)
        playful = _add_echo(shifted, [(60, 8), (140, 14)])
        return _highpass(playful, 350)
    # Great Deku Tree — ancient, massive, very deep
    if e == "z_deku":
        shifted = _pitch_shift(audio, -9)
        huge = _add_echo(shifted, [(160, 4), (340, 10), (580, 16)])
        return _lowpass(huge, 2400)

    # Fi (sword spirit) — synthetic feminine, crystal-clear
    if e == "z_fi":
        shifted = _pitch_shift(audio, +1)
        crystal = _add_echo(shifted, [(60, 10), (140, 18)])
        return _highpass(crystal, 400)
    # Midna — playful, mischievous
    if e == "z_midna":
        shifted = _pitch_shift(audio, +2)
        return _add_echo(shifted, [(90, 10), (200, 16)])
    # Navi — tiny fairy, sparkly high + echo
    if e == "z_navi":
        shifted = _pitch_shift(audio, +7)
        sparkly = _add_echo(shifted, [(70, 10), (160, 18)])
        return _highpass(sparkly, 500)
    # Guardian / Sheikah tech — ancient robotic
    if e == "z_guardian":
        shifted = _pitch_shift(audio, -2)
        hp = _highpass(shifted, 500)
        lp = _lowpass(hp, 3200)  # bandpass for radio-like feel
        return _add_echo(lp, [(60, 6), (140, 12)])
    # King of Red Lions (Wind Waker) — wise old on the sea
    if e == "z_red_lions":
        shifted = _pitch_shift(audio, -5)
        sea = _add_echo(shifted, [(140, 10), (300, 18), (520, 24)])
        return _lowpass(sea, 2600)

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
