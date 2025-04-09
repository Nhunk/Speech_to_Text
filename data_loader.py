import os
import numpy as np
import tensorflow as tf
from utils import extract_mfcc, normalize_text

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, max_length = 100):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')] # Assumption: .wav files
        self.max_length = max_length
        self.char_to_num = {}
        self.num_to_char = {}
        self.build_vocabulary()
        self.indexes = np.arange(len(self.audio_files))

    def build_vocabulary(self):
        """Builds a vocabulary from the transcriptions."""
        all_chars = set()
        for audio_file in self.audio_files:
            text_file = audio_file.replace('.wav', '.txt') # assumption : the transcription is in a txt file
            if os.path.exists(os.path.join(self.data_dir, text_file)):
                with open(os.path.join(self.data_dir, text_file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    normalized_text = normalize_text(text)
                    all_chars.update(list(normalized_text))
        all_chars = sorted(list(all_chars))
        all_chars = [''] + all_chars # add blank
        self.char_to_num = {char: i for i, char in enumerate(all_chars)}
        self.num_to_char = {i: char for i, char in enumerate(all_chars)}

    def __len__(self):
        return int(np.ceil(len(self.audio_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_audio_files = [self.audio_files[i] for i in batch_indexes]
        batch_mfccs = []
        batch_texts = []
        for audio_file in batch_audio_files:
            audio_path = os.path.join(self.data_dir, audio_file)
            mfccs = extract_mfcc(audio_path)
            if mfccs is None:
                continue # Skip if error
            text_file = audio_file.replace('.wav', '.txt') # assumption : the transcription is in a txt file
            text_path = os.path.join(self.data_dir, text_file)
            if not os.path.exists(text_path):
                continue
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
                normalized_text = normalize_text(text)
            # Pad or truncate
            mfccs = mfccs[:self.max_length, :]
            mfccs = np.pad(mfccs, [(0, self.max_length - mfccs.shape[0]), (0,0)])

            batch_mfccs.append(mfccs)
            encoded_text = [self.char_to_num[c] for c in normalized_text]
            batch_texts.append(encoded_text)

        batch_mfccs = np.array(batch_mfccs)

        # Padding label
        padded_batch_texts = tf.keras.preprocessing.sequence.pad_sequences(
            batch_texts, padding="post"
        )
        return batch_mfccs, np.array(padded_batch_texts)

