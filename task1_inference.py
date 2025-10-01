
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from dataset.count_score_by_me import count_score
import torch
import torchaudio
from dataloader import class_mapping
import argparse
from torchaudio.transforms import MelSpectrogram, Resample
import joblib

def extract_rich_features(mel_spectrogram):
    """Extract rich statistical features from a mel-spectrogram."""
    mel = mel_spectrogram.numpy()
    
    features = []
    
    features.append(np.mean(mel, axis=2).flatten()) 
    features.append(np.std(mel, axis=2).flatten()) 
    features.append(np.max(mel, axis=2).flatten())  
    features.append(np.min(mel, axis=2).flatten()) 
    
    delta = np.diff(mel, axis=2)
    features.append(np.mean(delta, axis=2).flatten())
    features.append(np.std(delta, axis=2).flatten())
    
    return np.concatenate(features)

idx_to_class = {v: k for k, v in class_mapping.items()}

# scaler = StandardScaler()
# pca = PCA(n_components=0.95, random_state=42)

print("\nExtracting validation features...")
Arg = argparse.ArgumentParser()
Arg.add_argument('--audio_dir', type=str, default='./dataset/artist20/test', help='path to test directory')
Arg.add_argument('--model_path', type=str, default='./models/task1_model/1759330048.pkl', help='path to the trained model')
Arg.add_argument('--output_json', type=str, default='./dataset/test_predictions.json', help='path to save the output JSON file')

args = Arg.parse_args()

test_dir = args.audio_dir
test_paths = {}


for root, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith('.mp3'):
            file_path = os.path.join(root, file)
            file_number = file.split('.')[0]
            test_paths[file_number] = file_path

# read the audio path and extract features
test_features = []
for name in tqdm(test_paths, desc="Extracting features"):
    # print(f"Processing {test_paths[name]}...")
    waveform, sr = torchaudio.load(test_paths[name], normalize=True)
    if sr != 16000:
        resampler = Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    
    # Get Mel-spectrogram
    mel_spectrogram = MelSpectrogram(sample_rate=sr, n_mels=256, hop_length=256)(waveform)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    features = extract_rich_features(log_mel_spectrogram)
    test_features.append(features)

test_features = np.array(test_features)
# load the trained SVM model
model_path = args.model_path
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)
scaler = model['scaler']
pca = model['pca']
svm_classifier = model['svm_classifier']

test_features = np.array(test_features)
test_features = scaler.transform(test_features)
test_features = pca.transform(test_features)

print(f"Test features shape: {test_features.shape}")

print("\nMaking predictions...")
val_predictions = svm_classifier.predict(test_features)
val_probabilities = svm_classifier.predict_proba(test_features)

print("=" * 50)

# store results in JSON format
output_json_path = args.output_json
results = {}

# pick top three predictions of each sample
for i, name in enumerate(test_paths):
    print(f"Predicting {name}...")
    top3_indices = np.argsort(val_probabilities[i])[-3:][::-1]
    top3_classes = [idx_to_class[idx] for idx in top3_indices]
    results[name] = top3_classes

results = dict(sorted(results.items(), key=lambda x: x[0]))

output_file = args.output_json
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nPredictions saved to {output_file}")