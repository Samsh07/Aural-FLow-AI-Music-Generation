import os
import sys
import torch

torch.cuda.is_available = lambda: False

original_torch_load = torch.load
def force_cpu_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return original_torch_load(*args, **kwargs)

torch.load = force_cpu_load

from src.interfaces.gradio_interface import create_gradio_interface

def main():
    """Main entry point for the AI Music Generation Platform"""
    print("🎵 AI Music Generation Platform Starting...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Because of our patch, this will now always be False
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected. Music generation will be slow on CPU.")
    
    # Add src directory to path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Create and launch the interface
        print("\n🚀 Creating Gradio interface...")
        interface = create_gradio_interface()
        
        print("\n🌐 Starting web server...")
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,  # Set to True for public access
            debug=True,
            show_error=True,
            prevent_thread_lock=True
        )
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check if port 7860 is available")
        print("3. Verify your internet connection for model downloads")
        print("4. For GPU support, ensure CUDA is properly installed")
        return

if __name__ == "__main__":
    main()