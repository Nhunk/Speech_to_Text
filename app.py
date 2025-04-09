from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import io
from scipy.io.wavfile import write as write_wav
import os
import tempfile
from audio_processor import WhisperProcessor
import logging
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create static directory if it doesn't exist
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vietnamese-speech-to-text'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Whisper processor
whisper_processor = WhisperProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('audio_data')
def handle_audio_data(data):
    try:
        # Decode the base64 audio data
        logger.info("Received audio data")
        audio_data = data['audio']
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_filename = temp_wav.name
        
        try:
            # Handle different formats of base64 data
            if ',' in audio_data:
                # Extract the actual base64 content after the comma
                audio_bytes = base64.b64decode(audio_data.split(',')[1])
            else:
                audio_bytes = base64.b64decode(audio_data)
            
            # Write the binary data directly to the file
            with open(temp_filename, 'wb') as f:
                f.write(audio_bytes)
            
            logger.info(f"Audio saved to temporary file: {temp_filename}")
            
            # Process with Whisper
            logger.info("Starting transcription with Whisper")
            transcription = whisper_processor.transcribe(temp_filename)
            logger.info(f"Transcription result: {transcription}")
            emit('transcription', {'text': transcription})
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            emit('error', {'message': f'Error processing audio: {str(e)}'})
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.info("Temporary file removed")
    except Exception as e:
        logger.error(f"Unexpected error in handle_audio_data: {e}")
        emit('error', {'message': f'Server error: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True)