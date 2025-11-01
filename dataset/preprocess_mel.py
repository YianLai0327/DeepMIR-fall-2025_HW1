import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
import os
import json
import numpy as np
from tqdm import tqdm

def preprocess_and_save_mel(json_list, output_json, sr=16000, n_mels=256, hop_length=256):
    """
    Preprocess audio files to mel-spectrograms and save to a JSON file.
    Args:
        json_list (str): Path to the input JSON file containing audio paths and labels.
        output_json (str): Path to the output JSON file to save mel-spectrograms.
        sr (int): Sample rate for audio files.
        n_mels (int): Number of mel bands.
        hop_length (int): Hop length for mel-spectrogram.
    """