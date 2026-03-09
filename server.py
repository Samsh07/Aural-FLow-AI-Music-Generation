import os
import sys
import uuid
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import shutil

# --- Import your existing logic ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'interfaces'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from gradio_interface import load_models_if_needed, generate_music_and_vocals
from rvc_utils import load_rvc_model

# --- App Config ---
app = FastAPI(
    title="AI Music Generation API",
    description="An API to generate music with singing vocals using AI models.",
    version="1.0.0"
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals ---
models_loaded = False
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Serve generated files
app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")

# --- Request/Response Models ---
class GenerationRequest(BaseModel):
    music_prompt: str
    lyrics: str
    duration: Optional[int] = 10  # default duration in seconds

class StatusResponse(BaseModel):
    status: str
    message: str
    file_url: Optional[str] = None

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    global models_loaded
    print("🚀 Loading AI models...")
    try:
        load_models_if_needed()
        models_loaded = True
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        models_loaded = False

# --- Routes ---
@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the AI Music Generation API. See /docs for usage."}

@app.get("/status", response_model=StatusResponse, tags=["General"])
async def get_status():
    if models_loaded:
        return StatusResponse(status="ready", message="Models are loaded and ready.")
    else:
        return StatusResponse(status="loading", message="Models are still loading.")

@app.post("/generate", tags=["Generation"])
async def trigger_generation(request: GenerationRequest):
    """
    Generate AI music with vocals.
    Request JSON:
    {
        "music_prompt": "lofi chill beat",
        "lyrics": "dreaming under neon lights",
        "duration": 15
    }
    """
    if not models_loaded:
        return JSONResponse(content={"status": "error", "message": "Server is not ready. Models are still loading."})

    try:
        # --- Generate music ---
        print(f"🎵 Generating with prompt='{request.music_prompt}', duration={request.duration}s")
        generated_path = generate_music_and_vocals(request.music_prompt, request.lyrics, request.duration)
        print("Generated file path:", generated_path)

        if not generated_path or not os.path.exists(generated_path):
            return JSONResponse(content={"status": "error", "message": "File not found after generation."})

        # --- Move generated file to outputs folder ---
        file_name = os.path.basename(generated_path)
        final_path = os.path.join(output_dir, file_name)
        if os.path.abspath(generated_path) != os.path.abspath(final_path):
            shutil.move(generated_path, final_path)
            print(f"Moved generated file to outputs folder: {final_path}")

        # --- Build file URL ---
        server_ip = "10.154.240.59"  # ⚠️ Change to your machine's IPv4 if different
        port = 8000
        file_url = f"http://{server_ip}:{port}/outputs/{file_name}"
        print("✅ File URL to send to Flutter:", file_url)

        return JSONResponse(content={
            "status": "success",
            "message": "Music generated successfully!",
            "file_name": file_name,
            "file_url": file_url
        })

    except Exception as e:
        print("❌ Error during generation:", e)
        return JSONResponse(content={"status": "error", "message": str(e)})

# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
