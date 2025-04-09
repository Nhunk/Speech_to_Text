import whisper
import torch

class WhisperProcessor:
    def __init__(self, model_size="base"):
        """
        Initialize the Whisper model for Vietnamese speech recognition.
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
        """
        # Check if CUDA is available and set the device accordingly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the Whisper model
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print(f"Loaded Whisper model: {model_size}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def transcribe(self, audio_file):
        """
        Transcribe Vietnamese speech to text.
        Args:
            audio_file: Path to the audio file
        Returns:
            Transcribed text
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_file,
                language="vi",
                task="transcribe",
                fp16=False if self.device == "cpu" else True
            )
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            raise