import gradio as gr
import openai
import whisper
from TTS.api import TTS

# ---- CONFIG ----
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
openai.api_key = OPENAI_API_KEY

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# ---- FUNCTION ----

def voice_to_voice(audio):
    # 1. Transcribe with Whisper
    result = whisper_model.transcribe(audio)
    text_input = result["text"]
    
    # 2. Generate LLM reply
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": text_input}
        ]
    )
    reply_text = response.choices[0].message.content

    # 3. Convert reply to speech
    tts.tts_to_file(text=reply_text, file_path="output.wav")

    return reply_text, "output.wav"

# ---- GRADIO UI ----

interface = gr.Interface(
    fn=voice_to_voice,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=[gr.Textbox(label="Transcription + Reply"), gr.Audio(label="AI Voice Reply")]
)

interface.launch()
