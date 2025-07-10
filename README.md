# Voice-Agent

## 🎙️ Voice-In → Gemini → Voice-Out App

This is a simple Gradio app that:
1️⃣ Takes your voice as input.  
2️⃣ Uses Whisper to transcribe it.  
3️⃣ Uses **Google Gemini** to generate a reply.  
4️⃣ Uses Coqui TTS to convert the reply back to voice.

---

## 🏃 How to Run on Replit (5 mins)

✅ 1. Fork this repo to your Replit account.  
✅ 2. Add your **Google Gemini API Key** in `app.py` → replace `YOUR_GOOGLE_GEMINI_API_KEY`.  
✅ 3. Click **Run** — Replit will install `requirements.txt` automatically.  
✅ 4. Open the web preview.  
✅ 5. Click **Record**, speak something, and hear the reply!

---

## ⚙️ Requirements

- Python 3.9+
- Google Gemini API Key (create one at [Google AI Studio](https://aistudio.google.com/))

---

## 🔗 Credits

- Whisper (by OpenAI)
- Gradio
- Coqui TTS
- Google Gemini (API)
