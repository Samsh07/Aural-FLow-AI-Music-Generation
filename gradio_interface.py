import gradio as gr
from audiocraft.models import MusicGen
from scipy.io.wavfile import write
from pydub import AudioSegment
import numpy as np
import torch
import os
import sys
import uuid

# --- START of RVC Integration ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.rvc_utils import load_rvc_model
# --- END of RVC Integration ---

# --- START of Singing Voice Synthesis Integration ---
from TTS.api import TTS
# --- END of Singing Voice Synthesis Integration ---

# -------------------------------
# 1. Lazy-loading models
# -------------------------------
music_model = None
singing_tts_model = None
rvc_model_config = None

def load_models_if_needed():
    global music_model, singing_tts_model, rvc_model_config

    print("Checking if models need to be loaded...")

    if singing_tts_model is None:
        print("Loading Singing Voice Synthesis model (coqui-tts)...")
        try:
            singing_tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            print("Singing TTS model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load Singing TTS model: {e}")
            singing_tts_model = None

    if music_model is None:
        print("Loading MusicGen model for background music...")
        music_model = MusicGen.get_pretrained("models/musicgen-small")
        music_model.set_generation_params(duration=20)
        print("MusicGen model loaded.")

    if rvc_model_config is None:
        print("Loading RVC model configuration...")
        model_path = "models/rvc/f0D_SingerPreTrain.pth"
        weights, config = load_rvc_model(model_path)
        if weights and config:
            rvc_model_config = {'weights': weights, 'config': config}
            print("RVC model configuration loaded successfully.")
        else:
            print("❌ Failed to load RVC model.")

os.makedirs("outputs", exist_ok=True)

# -------------------------------
# 2. Singing + RVC Conversion Function
# -------------------------------
def generate_and_convert_vocals(lyrics: str) -> str:
    load_models_if_needed()

    if singing_tts_model is None:
        print("Singing TTS model not loaded. Cannot generate vocals.")
        return None

    print("Generating pure singing vocals using a reference melody...")
    temp_vocal_path = f"outputs/temp_singing_vocals_{uuid.uuid4().hex}.wav"
    reference_melody_path = "my_melody.wav"  # your reference melody file

    try:
        singing_tts_model.tts_to_file(
            text=lyrics,
            speaker_wav=reference_melody_path,
            language="en",
            file_path=temp_vocal_path
        )
        print(f"Singing vocals generated at {temp_vocal_path}")
    except Exception as e:
        print(f"❌ Error generating singing vocals: {e}")
        print(f"Please make sure '{reference_melody_path}' exists in your project root.")
        return None

    print("Skipping full RVC conversion (demo mode). Using generated singing vocals directly.")
    return temp_vocal_path

# -------------------------------
# 3. Main function to generate music + vocals
# -------------------------------
def generate_music_and_vocals(music_prompt: str, lyrics: str, duration: int) -> str:
    load_models_if_needed()

    # ---- Step 1: Generate background music ----
    print(f"Generating background music ({duration} seconds)...")
    music_model.set_generation_params(duration=duration)
    music_wav = music_model.generate([music_prompt], progress=True)[0]

    if isinstance(music_wav, torch.Tensor):
        music_wav = music_wav.detach().cpu().numpy()
    if len(music_wav.shape) > 1:
        music_wav = music_wav[0]

    music_np = (music_wav * 32767).astype(np.int16)
    music_filename = f"outputs/generated_music_{uuid.uuid4().hex}.wav"
    write(music_filename, music_model.sample_rate, music_np)

    # ---- Step 2: Generate and Convert Vocals ----
    vocals_filename = generate_and_convert_vocals(lyrics)
    if vocals_filename is None:
        print("Vocal generation failed. Returning music only.")
        return music_filename

    # ---- Step 3: Combine music + singing vocals ----
    print("Mixing music and singing vocals...")
    from_wav_music = AudioSegment.from_wav(music_filename)
    from_wav_vocals = AudioSegment.from_wav(vocals_filename)
    
    from_wav_vocals = from_wav_vocals - 8  # Reduce vocal volume
    mix = from_wav_music.overlay(from_wav_vocals)

    final_file = f"outputs/final_mix_{uuid.uuid4().hex}.wav"
    mix.export(final_file, format="wav")
    print(f"Final mix saved to {final_file}")

    return final_file

# -------------------------------
# 4. Gradio Interface
# -------------------------------
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎶 AI Music Generation Platform (coqui-tts + RVC)")
        gr.Markdown("Generate background music + **singing vocals** using AI models. Choose your desired song duration below 👇")

        with gr.Row():
            music_input = gr.Textbox(label="🎧 Background Music Description", placeholder="e.g., lo-fi hip hop, chill, relaxed beat")
            lyrics_input = gr.Textbox(label="📝 Lyrics / Vocals", placeholder="e.g., [Verse] City lights are blurring now...")

        duration_input = gr.Slider(10, 120, value=60, step=10, label="⏱️ Song Duration (seconds)")

        output_audio = gr.Audio(label="🎵 Generated Music with Singing Vocals", type="filepath")
        generate_btn = gr.Button("🚀 Generate Music + Vocals")

        generate_btn.click(
            fn=generate_music_and_vocals,
            inputs=[music_input, lyrics_input, duration_input],
            outputs=output_audio,
        )

    return demo

# -------------------------------
# 5. Launch the App
# -------------------------------
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
