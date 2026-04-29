# دبلجة — Dublaja

تطبيق ويب **مجاني بالكامل** لدبلجة أصوات الألعاب والنصوص.

A free web app for game voice dubbing, text-to-speech, translation, and character voice effects — built entirely with free/open-source tools.

## ✨ Features

1. **نص إلى صوت (Text → Speech)** — تحويل أي نص إلى صوت بلغات متعددة ولهجات مختلفة.
2. **دبلجة ملف صوتي (Audio Dubbing)** — رفع ملف صوتي → تحويله إلى نص → ترجمته → إعادة نطقه بلغة جديدة.
3. **تأثيرات الشخصيات (Character Effects)** — تطبيق تأثيرات صوتية: روبوت، وحش، شيطان، طفل، سنجاب، عملاق، صدى.

## 🛠 Stack

- **Backend:** FastAPI + [gTTS](https://pypi.org/project/gTTS/) + [deep-translator](https://pypi.org/project/deep-translator/) + [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) + [pydub](https://pypi.org/project/pydub/)
- **Frontend:** Static HTML + Tailwind (via CDN) + vanilla JS with Arabic RTL UI.
- **Free APIs only:** No paid API keys required. Uses Google's public/unofficial free endpoints.

## 🚀 Run locally

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8001
```

### Frontend

Just open `frontend/index.html` in a browser, or serve statically:

```bash
cd frontend
python -m http.server 5173
```

Then edit `frontend/config.js` and set `window.__API_URL__` to your backend URL if different from `http://localhost:8001`.

## 🌐 API Endpoints

| Method | Path               | Purpose                                                    |
|--------|--------------------|------------------------------------------------------------|
| GET    | `/healthz`         | Health check                                               |
| GET    | `/api/languages`   | Supported TTS & STT languages                              |
| GET    | `/api/voices`      | Voice presets & effect list                                |
| POST   | `/api/tts`         | `{text, lang, slow, tld}` → MP3                            |
| POST   | `/api/tts/voice`   | `text`, `voice` preset → MP3                               |
| POST   | `/api/translate`   | `{text, source, target}` → translated text                 |
| POST   | `/api/dub`         | upload file + `source_lang`, `target_lang`, `voice` → JSON |
| POST   | `/api/effect`      | upload file + `effect` → MP3                               |

## 📝 License

MIT.
