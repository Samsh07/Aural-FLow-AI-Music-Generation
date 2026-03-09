# AI Music Generation Platform

An advanced music generation platform that combines AI models to create original music with singing vocals. This project integrates multiple state-of-the-art AI models to generate high-quality background music and synthesize singing voices with custom vocal transformations.

## Features

- **Music Generation**: Generate background music from text prompts using MusicGen
- **Singing Voice Synthesis**: Create singing vocals from lyrics using Coqui TTS
- **Voice Conversion**: Transform generated vocals using RVC (Real-time Voice Conversion)
- **Web Interface**: Interactive Gradio interface for easy music generation
- **REST API**: FastAPI backend for programmatic access
- **CPU-Optimized**: Runs efficiently on CPU without GPU requirements

## Project Structure

```
.
├── app.py                     # Main Gradio web interface entry point
├── server.py                  # FastAPI REST API server
├── req.txt                    # Python dependencies
├── models/
│   ├── musicgen-small/        # MusicGen model files
│   └── rvc/                   # RVC voice conversion model
├── src/
│   ├── rvc_utils.py           # Voice conversion utilities
│   └── interfaces/
│       └── gradio_interface.py # Gradio UI implementation
└── outputs/                   # Generated music files (created at runtime)
```

## Requirements

- Python 3.8+
- PyTorch (CPU version included)
- 4GB+ RAM (8GB+ recommended)
- 5GB+ free disk space for models

## Installation

### 1. Clone/Setup the Project

```bash
cd c:\msirvc
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r req.txt
```

> **Note**: The first run will download the MusicGen and TTS models (~2-3 GB). This may take several minutes depending on your internet speed.

## Usage

### Web Interface (Recommended)

Run the Gradio interface for an interactive experience:

```bash
python app.py
```

Then open your browser to `http://127.0.0.1:7860`

**Steps**:
1. Enter a music description/prompt (e.g., "upbeat electronic dance music")
2. Enter lyrics for the singing vocals
3. Adjust duration if needed
4. Click "Generate"
5. Download the generated audio file

### REST API

Start the FastAPI server:

```bash
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

API Documentation available at `http://127.0.0.1:8000/docs`

**Example Request**:
```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d {
    "music_prompt": "upbeat electronic dance music",
    "lyrics": "Dancing in the moonlight",
    "duration": 10
  }
```

## Model Information

### MusicGen
- **Source**: Facebook Research AudioCraft
- **Model**: musicgen-small
- **Purpose**: Generates background music from text descriptions
- **Output**: WAV format audio

### Singing Voice Synthesis (TTS)
- **Source**: Coqui TTS
- **Model**: xtts_v2 (Multilingual)
- **Purpose**: Synthesizes singing vocals from text lyrics
- **Supports**: Multiple languages

### RVC (Real-time Voice Conversion)
- **Model**: f0D_SingerPreTrain.pth
- **Purpose**: Transforms vocal characteristics without changing content
- **Use Case**: Apply custom voice styles to generated vocals

## Configuration

### Model Parameters

Edit `src/interfaces/gradio_interface.py` to adjust:
- **Duration**: Change `music_model.set_generation_params(duration=20)`
- **Temperature**: Audio generation variance
- **Top-k/Top-p**: Sampling parameters for generation quality

### Server Settings

In `app.py`:
- Change `server_port=7860` to use a different port
- Set `share=True` for public URL sharing
- Modify `server_name` for network access

## Troubleshooting

### Models Not Loading
```bash
# Clear model cache and reinstall
rm -rf ~/.cache/audiocraft
pip install --upgrade audiocraft
```

### CUDA/GPU Issues
The app forces CPU-only mode for compatibility. GPU support can be enabled by modifying `app.py`.

### Out of Memory
- Reduce music duration
- Close other applications
- Use CPU instead of GPU for slower but more compatible operation

### Port Already in Use
```bash
# Change port in app.py (around line 39)
# server_port=7860 -> server_port=7861
```

### Missing Dependencies
```bash
pip install -r req.txt --upgrade
```

## Performance Tips

- **First Run**: Model loading takes 1-2 minutes
- **Generation Time**: ~20-30 seconds per song on CPU
- **VRAM**: Currently running on CPU to ensure compatibility
- **Batch Processing**: Generate multiple songs sequentially

## Contributing

Improvements welcome! Consider:
- GPU optimization
- Additional voice models
- Music style controls
- Batch generation API

## License

This project uses open-source models and libraries. Please refer to individual model licenses:
- AudioCraft: MIT License
- Coqui TTS: GPL/Custom License
- RVC: Original model license

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review model documentation links
3. Check Python and dependency versions match requirements

## References

- [AudioCraft - Meta AI](https://github.com/facebookresearch/audiocraft)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [RVC Voice Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
