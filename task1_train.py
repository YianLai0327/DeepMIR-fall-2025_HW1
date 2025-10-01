from dataloader import AudioDataset
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import json
import os
from tqdm import tqdm

# build idx_to_class mapping
from dataloader import class_mapping
idx_to_class = {v: k for k, v in class_mapping.items()}

# train_dataloader = create_dataloader(
#     json_list='./dataset/artist20/train.json', 
#     batch_size=16, 
#     mode='train'
# )
# val_dataloader = create_dataloader(
#     json_list='./dataset/artist20/val.json', 
#     batch_size=16, 
#     mode='val'
# )

# # ============ training ============
# train_features = []
# train_labels = []
# for mel_spectrogram, lengths, label in train_dataloader:
#     pooled = np.mean(mel_spectrogram.numpy(), axis=2)
#     train_features.append(pooled)
#     train_labels.append(label.numpy())

# train_features = np.vstack(train_features)
# train_labels = np.hstack(train_labels)

print("Loading training data...")
train_dataset = AudioDataset(
    json_list='./dataset/artist20/train.json',
    sr=16000,
    n_mels=128,
    hop_length=512,
    is_onehot=False,
    mode='train'
)

print("Loading validation data...")
val_dataset = AudioDataset(
    json_list='./dataset/artist20/val.json',
    sr=16000,
    n_mels=128,
    hop_length=512,
    is_onehot=False,
    mode='val'
)
def extract_rich_features(mel_spectrogram):
    """提取更豐富的特徵"""
    mel = mel_spectrogram.numpy()
    
    features = []
    
    # 1. 統計特徵 (針對頻率軸)
    features.append(np.mean(mel, axis=2).flatten())    # 平均
    features.append(np.std(mel, axis=2).flatten())     # 標準差
    features.append(np.max(mel, axis=2).flatten())     # 最大值
    features.append(np.min(mel, axis=2).flatten())     # 最小值
    
    # 2. 時間動態特徵
    delta = np.diff(mel, axis=2)
    features.append(np.mean(delta, axis=2).flatten())
    features.append(np.std(delta, axis=2).flatten())
    
    return np.concatenate(features)

print("Extracting training features...")
train_features = []
train_labels = []

for i in tqdm(range(len(train_dataset)), desc="Processing training data"):
    mel_spectrogram, label = train_dataset[i]
    # 平均池化：(1, n_mels, time) -> (n_mels,)
    features = extract_rich_features(mel_spectrogram)
    train_features.append(features)
    train_labels.append(label)

train_features = np.array(train_features)
train_labels = np.array(train_labels)

print(f"Training features shape: {train_features.shape}")
print(f"Training labels shape: {train_labels.shape}")

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

svm_classifier = SVC(
    kernel='rbf', 
    probability=True,
    C=10,
    gamma='scale',
    verbose=True
)
svm_classifier.fit(train_features, train_labels)

# ============ validation ============
# val_features = []
# val_labels = []

# # read val file paths to construct JSON later
# with open('./dataset/artist20/val.json', 'r') as f:
#     val_file_paths = json.load(f)

# for mel_spectrogram, lengths, label in val_dataloader:
#     pooled = np.mean(mel_spectrogram.numpy(), axis=2)
#     val_features.append(pooled)
#     val_labels.append(label.numpy())

# val_features = np.vstack(val_features)
# val_labels = np.hstack(val_labels)
# val_features = scaler.transform(val_features)

print("\nExtracting validation features...")
val_features = []
val_labels = []

with open('./dataset/artist20/val.json', 'r') as f:
    val_json_paths = json.load(f)

for i in tqdm(range(len(val_dataset)), desc="Processing validation data"):
    mel_spectrogram, label = val_dataset[i]
    features = extract_rich_features(mel_spectrogram)
    val_features.append(features)
    val_labels.append(label)

val_features = np.array(val_features)
val_labels = np.array(val_labels)
val_features = scaler.transform(val_features)

print(f"Validation features shape: {val_features.shape}")

print("\nMaking predictions...")
val_predictions = svm_classifier.predict(val_features)
val_probabilities = svm_classifier.predict_proba(val_features)

print("\n" + "=" * 50)
print("Classification Report:")
print(classification_report(val_labels, val_predictions, 
                          target_names=list(class_mapping.keys())))
print("=" * 50)

# store results in JSON format
results = {}

for i, file_path in enumerate(val_json_paths):
    filename = os.path.basename(file_path)
    file_number = filename.split('/')[-1].split('.')[0]
    
    top3_idx = np.argsort(val_probabilities[i])[-3:][::-1]
    top3_classes = [idx_to_class[idx] for idx in top3_idx]
    
    results[file_number] = top3_classes

# 按編號排序
results = dict(sorted(results.items(), key=lambda x: x[0]))

output_file = './dataset/predictions.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✅ Predictions saved to {output_file}")
print(f"Total samples processed: {len(results)}")

print("\nShowing top 10 results:")
print(json.dumps(dict(list(results.items())[:10]), indent=4))

print("\n" + "=" * 80)
print("Detailed Top-3 Predictions for First 10 Validation Samples:")
print(f"{'ID':<6} {'True Label':<25} {'Top-3 Predictions (with probabilities)'}")
print("-" * 80)

for i, file_path in enumerate(val_json_paths[:10]):
    filename = os.path.basename(file_path)
    file_number = filename.split('/')[-1].split('.')[0]
    true_label = idx_to_class[val_labels[i]]
    
    top3_idx = np.argsort(val_probabilities[i])[-3:][::-1]
    top3_info = ", ".join([
        f"{idx_to_class[idx]}({val_probabilities[i][idx]:.3f})" 
        for idx in top3_idx
    ])
    
    status = "✓" if true_label == idx_to_class[top3_idx[0]] else "✗"
    print(f"{status} {file_number:<6} {true_label:<25} {top3_info}")