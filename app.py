import gradio as gr
import whisper
from TTS.api import TTS
import requests

# ---- CONFIG ----
GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + GEMINI_API_KEY

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# ---- FUNCTION ----

def voice_to_voice(audio):
    # 1. Transcribe with Whisper
    result = whisper_model.transcribe(audio)
    text_input = result["text"]
    
    # 2. Generate LLM reply from Gemini
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": text_input
                    }
                ]
            }
        ]
    }

    response = requests.post(GEMINI_URL, json=payload)
    gemini_response = response.json()
    try:
        reply_text = gemini_response['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        reply_text = f"Error: {e} \nRaw response: {gemini_response}"

    # 3. Convert reply to speech
    tts.tts_to_file(text=reply_text, file_path="output.wav")

    return reply_text, "output.wav"

# ---- GRADIO UI ----

interface = gr.Interface(
    fn=voice_to_voice,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=[gr.Textbox(label="Transcription + Gemini Reply"), gr.Audio(label="AI Voice Reply")]
)

interface.launch()
