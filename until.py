import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCC features from an audio file."""
    try:
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs = np.transpose(mfccs) # Transpose to have (time_step, features)
        return mfccs
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

def normalize_text(text):
    """Basic text normalization: lowercase and remove punctuation."""
    text = text.lower()
    # add more rules if needed
    return text
