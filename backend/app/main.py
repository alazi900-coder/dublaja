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

import asyncio

import edge_tts
from gtts import gTTS
from gtts.lang import tts_langs
from deep_translator import GoogleTranslator
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import speedup

from app.fish_tts import (
    FISH_VOICE_PRESETS,
    FishAudioError,
    synthesize_fish,
)


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

# --------------------------------------------------------------------------- #
# Voice presets — Microsoft Edge Neural TTS (free, no API key)
# --------------------------------------------------------------------------- #
# Each preset picks a real neural voice + optional prosody (rate/pitch/volume)
# + optional LIGHT post-effect for flavor. Different characters use DIFFERENT
# voices (male/female, young/old, dialects) instead of heavy echo/pitch
# stacks on a single voice.

# Shortcut voices we reuse a lot
_AR_FEM_NOBLE   = "ar-SA-ZariyahNeural"     # clear, noble female
_AR_FEM_NOURA   = "ar-KW-NouraNeural"       # softer female
_AR_FEM_AMANY   = "ar-SY-AmanyNeural"       # strong female
_AR_FEM_LAYLA   = "ar-LB-LaylaNeural"       # higher/airy female
_AR_FEM_FATIMA  = "ar-AE-FatimaNeural"      # confident female
_AR_FEM_SALMA   = "ar-EG-SalmaNeural"       # playful female
_AR_MALE_HAMED  = "ar-SA-HamedNeural"       # calm male
_AR_MALE_BASSEL = "ar-IQ-BasselNeural"      # deep male
_AR_MALE_ALI    = "ar-BH-AliNeural"         # authoritative male
_AR_MALE_OMAR   = "ar-LY-OmarNeural"        # mysterious male
_AR_MALE_SHAKIR = "ar-EG-ShakirNeural"      # energetic male
_AR_MALE_LAITH  = "ar-SY-LaithNeural"       # older male
_AR_MALE_HAMDAN = "ar-AE-HamdanNeural"      # crisp male
_AR_MALE_RAMI   = "ar-LB-RamiNeural"        # smooth male

VOICE_PRESETS: dict[str, dict] = {
    # --- Base languages & accents ---
    "arabic_default":    {"edge": _AR_FEM_NOBLE,          "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "arabic_slow":       {"edge": _AR_FEM_NOBLE,          "rate": "-20%", "pitch": "+0Hz",  "effect": None},
    "arabic_male":       {"edge": _AR_MALE_HAMED,         "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "arabic_egypt_f":    {"edge": _AR_FEM_SALMA,          "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "arabic_egypt_m":    {"edge": _AR_MALE_SHAKIR,        "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "english_us":        {"edge": "en-US-AriaNeural",     "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "english_us_male":   {"edge": "en-US-GuyNeural",      "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "english_uk":        {"edge": "en-GB-SoniaNeural",    "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "english_uk_male":   {"edge": "en-GB-RyanNeural",     "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "english_australia": {"edge": "en-AU-NatashaNeural",  "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "english_india":     {"edge": "en-IN-NeerjaNeural",   "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "french":            {"edge": "fr-FR-DeniseNeural",   "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "spanish":           {"edge": "es-ES-ElviraNeural",   "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "german":            {"edge": "de-DE-KatjaNeural",    "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "turkish":           {"edge": "tr-TR-EmelNeural",     "rate": "+0%",  "pitch": "+0Hz",  "effect": None},
    "japanese":          {"edge": "ja-JP-NanamiNeural",   "rate": "+0%",  "pitch": "+0Hz",  "effect": None},

    # --- Game character archetypes (natural voices, minimal processing) ---
    "game_hero":         {"edge": _AR_MALE_HAMDAN,        "rate": "+0%",  "pitch": "-2Hz",  "effect": None},
    "game_warrior":      {"edge": _AR_MALE_BASSEL,        "rate": "-5%",  "pitch": "-3Hz",  "effect": None},
    "game_knight":       {"edge": _AR_MALE_ALI,           "rate": "-5%",  "pitch": "+0Hz",  "effect": None},
    "game_samurai":      {"edge": _AR_MALE_LAITH,         "rate": "-5%",  "pitch": "-2Hz",  "effect": None},
    "game_villain":      {"edge": _AR_MALE_BASSEL,        "rate": "-10%", "pitch": "-4Hz",  "effect": "v_light"},
    "game_demon":        {"edge": _AR_MALE_BASSEL,        "rate": "-15%", "pitch": "-6Hz",  "effect": "demon_light"},
    "game_dark_lord":    {"edge": _AR_MALE_BASSEL,        "rate": "-20%", "pitch": "-5Hz",  "effect": "darklord_light"},
    "game_dragon":       {"edge": _AR_MALE_BASSEL,        "rate": "-20%", "pitch": "-6Hz",  "effect": "dragon_light"},
    "game_orc":          {"edge": _AR_MALE_BASSEL,        "rate": "-10%", "pitch": "-5Hz",  "effect": None},
    "game_goblin":       {"edge": _AR_MALE_SHAKIR,        "rate": "+15%", "pitch": "+5Hz",  "effect": None},
    "game_troll":        {"edge": _AR_MALE_BASSEL,        "rate": "-15%", "pitch": "-5Hz",  "effect": None},
    "game_monster":      {"edge": _AR_MALE_BASSEL,        "rate": "-15%", "pitch": "-6Hz",  "effect": "monster_light"},
    "game_giant":        {"edge": _AR_MALE_BASSEL,        "rate": "-20%", "pitch": "-7Hz",  "effect": None},
    "game_zombie":       {"edge": _AR_MALE_LAITH,         "rate": "-25%", "pitch": "-4Hz",  "effect": "zombie_light"},
    "game_wizard":       {"edge": _AR_MALE_ALI,           "rate": "-10%", "pitch": "+0Hz",  "effect": "wizard_light"},
    "game_witch":        {"edge": _AR_FEM_AMANY,          "rate": "+5%",  "pitch": "+3Hz",  "effect": "witch_light"},
    "game_fairy":        {"edge": _AR_FEM_LAYLA,          "rate": "+10%", "pitch": "+6Hz",  "effect": "fairy_light"},
    "game_ghost":        {"edge": _AR_FEM_LAYLA,          "rate": "-10%", "pitch": "+2Hz",  "effect": "ghost_light"},
    "game_announcer":    {"edge": _AR_MALE_HAMDAN,        "rate": "+0%",  "pitch": "+0Hz",  "effect": "announcer_light"},
    "game_narrator":     {"edge": _AR_MALE_OMAR,          "rate": "-5%",  "pitch": "-1Hz",  "effect": None},
    "game_old_wise":     {"edge": _AR_MALE_LAITH,         "rate": "-15%", "pitch": "-3Hz",  "effect": None},
    "game_child":        {"edge": _AR_FEM_LAYLA,          "rate": "+10%", "pitch": "+5Hz",  "effect": None},
    "game_robot":        {"edge": _AR_MALE_HAMED,         "rate": "+0%",  "pitch": "+0Hz",  "effect": "robot_light"},
    "game_alien":        {"edge": _AR_FEM_AMANY,          "rate": "+5%",  "pitch": "+4Hz",  "effect": "alien_light"},
    "game_ai":           {"edge": _AR_FEM_NOURA,          "rate": "+0%",  "pitch": "+0Hz",  "effect": "ai_light"},
    "game_radio":        {"edge": _AR_MALE_HAMED,         "rate": "+0%",  "pitch": "+0Hz",  "effect": "radio_light"},
    "game_fire_spirit":  {"edge": _AR_MALE_BASSEL,        "rate": "-5%",  "pitch": "-3Hz",  "effect": "fire_light"},
    "game_ice_spirit":   {"edge": _AR_FEM_LAYLA,          "rate": "-10%", "pitch": "+2Hz",  "effect": "ice_light"},
    "game_underwater":   {"edge": _AR_MALE_RAMI,          "rate": "+0%",  "pitch": "+0Hz",  "effect": "underwater_light"},

    # --- Zelda character pack ---
    # Heroes & royalty
    "zelda_link":        {"edge": _AR_MALE_HAMDAN,        "rate": "+0%",  "pitch": "-1Hz",  "effect": None},   # young hero
    "zelda_zelda":       {"edge": _AR_FEM_NOBLE,          "rate": "-5%",  "pitch": "+0Hz",  "effect": None},   # noble princess
    "zelda_king":        {"edge": _AR_MALE_ALI,           "rate": "-10%", "pitch": "-3Hz",  "effect": None},   # authoritative king
    "zelda_impa":        {"edge": _AR_FEM_AMANY,          "rate": "-5%",  "pitch": "-2Hz",  "effect": None},   # strong elder female
    "zelda_rauru":       {"edge": _AR_MALE_LAITH,         "rate": "-15%", "pitch": "-2Hz",  "effect": None},   # wise old sage
    "zelda_sheik":       {"edge": _AR_MALE_OMAR,          "rate": "+0%",  "pitch": "+0Hz",  "effect": "sheik_light"},  # mysterious
    # Villains
    "zelda_ganon":       {"edge": _AR_MALE_BASSEL,        "rate": "-15%", "pitch": "-6Hz",  "effect": "ganon_light"},   # menacing deep
    "zelda_demise":      {"edge": _AR_MALE_BASSEL,        "rate": "-20%", "pitch": "-7Hz",  "effect": "demise_light"},  # ancient evil
    "zelda_yiga":        {"edge": _AR_MALE_SHAKIR,        "rate": "+5%",  "pitch": "+0Hz",  "effect": None},            # sneaky
    "zelda_skull_kid":   {"edge": _AR_FEM_SALMA,          "rate": "+10%", "pitch": "+5Hz",  "effect": "skullkid_light"},
    # Races / tribes
    "zelda_goron":       {"edge": _AR_MALE_BASSEL,        "rate": "-15%", "pitch": "-5Hz",  "effect": None},   # rocky deep
    "zelda_zora":        {"edge": _AR_MALE_RAMI,          "rate": "+0%",  "pitch": "+0Hz",  "effect": "zora_light"},    # smooth, watery
    "zelda_rito":        {"edge": _AR_FEM_FATIMA,         "rate": "+5%",  "pitch": "+3Hz",  "effect": None},   # bird-like
    "zelda_gerudo":      {"edge": _AR_FEM_AMANY,          "rate": "+0%",  "pitch": "-1Hz",  "effect": None},   # desert-strong
    "zelda_korok":       {"edge": _AR_FEM_LAYLA,          "rate": "+15%", "pitch": "+7Hz",  "effect": None},   # tiny playful
    "zelda_deku":        {"edge": _AR_MALE_BASSEL,        "rate": "-25%", "pitch": "-5Hz",  "effect": "deku_light"},    # ancient huge
    # Spirits / guides
    "zelda_fi":          {"edge": _AR_FEM_NOURA,          "rate": "-5%",  "pitch": "+2Hz",  "effect": "fi_light"},      # synthetic clear
    "zelda_midna":       {"edge": _AR_FEM_SALMA,          "rate": "+5%",  "pitch": "+2Hz",  "effect": None},   # playful
    "zelda_navi":        {"edge": _AR_FEM_LAYLA,          "rate": "+20%", "pitch": "+8Hz",  "effect": None},   # tiny fairy
    "zelda_guardian":    {"edge": _AR_MALE_HAMED,         "rate": "+0%",  "pitch": "+0Hz",  "effect": "guardian_light"},# Sheikah tech
    "zelda_red_lions":   {"edge": _AR_MALE_LAITH,         "rate": "-10%", "pitch": "-2Hz",  "effect": None},   # wise old sea
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

    # ---------- *_light variants: very subtle colouring on top of a
    # real neural voice (tuned to NOT sound robotic/slow/echoy) ----------
    if e == "v_light":                       # villain tint
        return _add_echo(audio + 1, [(90, 28)])
    if e == "demon_light":                   # subtle demon rumble
        return _add_echo(audio, [(80, 22)])
    if e == "darklord_light":
        return _add_echo(audio, [(140, 26)])
    if e == "dragon_light":                  # slight lowpass, no echo
        return _lowpass(audio, 3200)
    if e == "monster_light":
        return _add_echo(audio, [(70, 26)])
    if e == "zombie_light":                  # muffled
        return _lowpass(audio, 2800)
    if e == "wizard_light":                  # tiny hall
        return _add_echo(audio, [(160, 30)])
    if e == "witch_light":
        return _add_echo(audio, [(120, 30)])
    if e == "fairy_light":                   # sparkly top-end only
        return _highpass(audio, 300)
    if e == "ghost_light":                   # airy tail
        return _add_echo(audio, [(180, 28)])
    if e == "announcer_light":               # loud with small room
        return _add_echo(audio + 2, [(140, 28)])
    if e == "robot_light":                   # slight metallic band
        return _highpass(_lowpass(audio, 3400), 220)
    if e == "alien_light":
        return _add_echo(audio, [(60, 26)])
    if e == "ai_light":                      # crisp
        return _highpass(audio, 200)
    if e == "radio_light":                   # walkie-talkie band
        hp = _highpass(audio, 380)
        return _lowpass(hp, 3200)
    if e == "fire_light":
        return _add_echo(audio, [(90, 28)])
    if e == "ice_light":
        return _add_echo(audio, [(200, 30)])
    if e == "underwater_light":              # muffled
        return _lowpass(audio, 1400)

    # Zelda-specific light colouring
    if e == "sheik_light":                   # mysterious
        return _add_echo(_highpass(audio, 280), [(140, 30)])
    if e == "ganon_light":                   # menacing
        return _add_echo(_lowpass(audio, 3000), [(120, 26)])
    if e == "demise_light":
        return _add_echo(_lowpass(audio, 2800), [(180, 26)])
    if e == "skullkid_light":
        return _add_echo(audio, [(120, 30)])
    if e == "zora_light":
        return _lowpass(audio, 2500)
    if e == "deku_light":
        return _add_echo(_lowpass(audio, 2800), [(180, 28)])
    if e == "fi_light":
        return _highpass(audio, 300)
    if e == "guardian_light":                # ancient tech
        hp = _highpass(audio, 420)
        lp = _lowpass(hp, 3400)
        return _add_echo(lp, [(70, 28)])

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


# --------------------------------------------------------------------------- #
# TTS engines
# --------------------------------------------------------------------------- #

# Map ISO language codes -> default Edge Neural voice (used when the caller
# passes a bare lang without a preset)
_DEFAULT_EDGE_VOICE = {
    "ar": "ar-SA-ZariyahNeural",
    "en": "en-US-AriaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "tr": "tr-TR-EmelNeural",
    "ja": "ja-JP-NanamiNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "hi": "hi-IN-SwaraNeural",
    "ur": "ur-PK-UzmaNeural",
}


async def _edge_bytes(text: str, voice: str, rate: str = "+0%",
                      pitch: str = "+0Hz", volume: str = "+0%") -> bytes:
    """Run Edge TTS and return MP3 bytes."""
    communicate = edge_tts.Communicate(
        text=text, voice=voice, rate=rate, pitch=pitch, volume=volume
    )
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def synthesize_edge(text: str, voice: str, rate: str = "+0%",
                    pitch: str = "+0Hz", effect: str | None = None) -> bytes:
    """Neural TTS via Microsoft Edge + optional light post-effect."""
    if not text.strip():
        raise HTTPException(400, "Text is empty")

    try:
        mp3 = asyncio.run(_edge_bytes(text, voice, rate=rate, pitch=pitch))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(502, f"Edge TTS failed: {exc}") from exc

    if not effect:
        return mp3

    try:
        audio = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
        audio = apply_effect(audio, effect)
        out = io.BytesIO()
        audio.export(out, format="mp3")
        return out.getvalue()
    except Exception:
        # If ffmpeg isn't ready yet, just return the clean neural audio.
        return mp3


def synthesize(text: str, lang: str, tld: str = "com", slow: bool = False,
               effect: str | None = None) -> bytes:
    """Backward-compatible synthesize: prefer Edge Neural, fall back to gTTS."""
    if not text.strip():
        raise HTTPException(400, "Text is empty")

    voice = _DEFAULT_EDGE_VOICE.get(lang.split("-")[0])
    if voice:
        rate = "-20%" if slow else "+0%"
        try:
            return synthesize_edge(text, voice=voice, rate=rate, effect=effect)
        except HTTPException:
            pass  # fall through to gTTS

    # gTTS fallback
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
            {
                "id": k,
                "edge": v.get("edge"),
                "rate": v.get("rate"),
                "pitch": v.get("pitch"),
                "effect": v.get("effect"),
            }
            for k, v in VOICE_PRESETS.items()
        ],
        "fish_voices": [
            {
                "id": k,
                "reference_id": v.get("reference_id"),
                "model": v.get("model", "s2-pro"),
                "speed": v.get("speed", 1.0),
            }
            for k, v in FISH_VOICE_PRESETS.items()
        ],
        "fish_enabled": bool(os.environ.get("FISH_AUDIO_API_KEY", "").strip()),
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
    data = synthesize_edge(
        text,
        voice=preset["edge"],
        rate=preset.get("rate", "+0%"),
        pitch=preset.get("pitch", "+0Hz"),
        effect=preset.get("effect"),
    )
    return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg")


@app.post("/api/tts/fish")
def api_tts_fish(
    text: str = Form(...),
    voice: str | None = Form(None),
    reference_id: str | None = Form(None),
    model: str = Form("s2-pro"),
    speed: float = Form(1.0),
    temperature: float = Form(0.7),
    top_p: float = Form(0.7),
):
    """
    Fish Audio TTS — premium quality + voice cloning via preset.

    - `voice`: preset key from /api/voices fish_voices (e.g. fish_zelda_link)
    - `reference_id`: Fish Audio model UUID (overrides preset's reference_id)
    """
    preset_speed = speed
    preset_temp = temperature
    preset_top_p = top_p
    preset_model = model
    ref_id = reference_id

    if voice and voice in FISH_VOICE_PRESETS:
        p = FISH_VOICE_PRESETS[voice]
        ref_id = ref_id or p.get("reference_id")
        preset_model = p.get("model", model)
        preset_speed = p.get("speed", speed)
        preset_temp = p.get("temperature", temperature)
        preset_top_p = p.get("top_p", top_p)

    try:
        mp3 = synthesize_fish(
            text,
            reference_id=ref_id,
            model=preset_model,
            speed=preset_speed,
            temperature=preset_temp,
            top_p=preset_top_p,
        )
    except FishAudioError as e:
        raise HTTPException(e.status, e.message)

    return StreamingResponse(io.BytesIO(mp3), media_type="audio/mpeg")


@app.post("/api/tts/fish/clone")
async def api_tts_fish_clone(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_transcript: str = Form(...),
    model: str = Form("s2-pro"),
    speed: float = Form(1.0),
    temperature: float = Form(0.7),
    top_p: float = Form(0.7),
):
    """
    Fish Audio zero-shot voice cloning.

    Upload a 6-30s reference audio + its transcript -> synthesize new text in
    the same voice (premium quality).
    """
    audio_bytes = await reference_audio.read()
    if not audio_bytes:
        raise HTTPException(400, "ملف العينة فارغ")
    if len(audio_bytes) > 20 * 1024 * 1024:  # 20MB cap
        raise HTTPException(400, "ملف العينة كبير جداً (>20 ميغا).")

    try:
        mp3 = synthesize_fish(
            text,
            reference_audio=audio_bytes,
            reference_transcript=reference_transcript,
            model=model,
            speed=speed,
            temperature=temperature,
            top_p=top_p,
        )
    except FishAudioError as e:
        raise HTTPException(e.status, e.message)

    return StreamingResponse(io.BytesIO(mp3), media_type="audio/mpeg")


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
        mp3 = synthesize_edge(
            translated,
            voice=p["edge"],
            rate=p.get("rate", "+0%"),
            pitch=p.get("pitch", "+0Hz"),
            effect=p.get("effect"),
        )
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
