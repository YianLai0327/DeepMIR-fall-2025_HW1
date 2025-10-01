from dataloader import AudioDataset
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import json
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from dataset.count_score_by_me import count_score
import torch
import torchaudio
from task1_train import extract_rich_features
from dataloader import class_mapping
import argparse
from torchaudio.transforms import MelSpectrogram, Resample

idx_to_class = {v: k for k, v in class_mapping.items()}

scaler = StandardScaler()
pca = PCA(n_components=0.95, random_state=42)

print("\nExtracting validation features...")
Arg = argparse.ArgumentParser()
Arg.add_argument('--inference_dir', type=str, default='./dataset/artist20/test', help='path to test directory')
Arg.add_argument('--model_path', type=str, default='./models/task1_model/1759323935.npz', help='path to the trained model')
Arg.add_argument('--output_json', type=str, default='./dataset/test_predictions.json', help='path to save the output JSON file')

args = Arg.parse_args()

test_dir = args.inference_dir
test_paths = {}

for root, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith('.mp3'):
            file_path = os.path.join(root, file)
            file_number = file.split('.')[0]
            test_paths[file_number] = file_path

# read the audio path and extract features
for name in test_paths:
    print(f"Processing {test_paths[name]}...")
    wav, sr = torchaudio.load(test_paths[name], normalize=True)
    if sr != 16000:
        resampler = Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    
    # Get Mel-spectrogram
    mel_spectrogram = MelSpectrogram(sample_rate=sr, n_mels=256, hop_length=256)(waveform)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    features = extract_rich_features(log_mel_spectrogram)

features = np.array(features)
features = scaler.transform(features)
features = pca.transform(features)

print(f"Validation features shape: {features.shape}")

# load the trained SVM model
model_path = args.model_path
svm_classifier = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_classifier.load(model_path)

print("\nMaking predictions...")
val_predictions = svm_classifier.predict(features)
val_probabilities = svm_classifier.predict_proba(features)

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

output_file = './dataset/val_predictions.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
