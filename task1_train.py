from dataloader import create_dataloader
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import json
import os

# build idx_to_class mapping
from dataloader import class_mapping
idx_to_class = {v: k for k, v in class_mapping.items()}

train_dataloader = create_dataloader(
    json_list='./dataset/artist20/train.json', 
    batch_size=16, 
    mode='train'
)
val_dataloader = create_dataloader(
    json_list='./dataset/artist20/val.json', 
    batch_size=16, 
    mode='val'
)

# ============ training ============
train_features = []
train_labels = []
for mel_spectrogram, lengths, label in train_dataloader:
    pooled = np.mean(mel_spectrogram.numpy(), axis=2)
    train_features.append(pooled)
    train_labels.append(label.numpy())

train_features = np.vstack(train_features)
train_labels = np.hstack(train_labels)

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
val_features = []
val_labels = []

# read val file paths to construct JSON later
with open('./dataset/artist20/val.json', 'r') as f:
    val_file_paths = json.load(f)

for mel_spectrogram, lengths, label in val_dataloader:
    pooled = np.mean(mel_spectrogram.numpy(), axis=2)
    val_features.append(pooled)
    val_labels.append(label.numpy())

val_features = np.vstack(val_features)
val_labels = np.hstack(val_labels)
val_features = scaler.transform(val_features)

val_predictions = svm_classifier.predict(val_features)
val_probabilities = svm_classifier.predict_proba(val_features)

print("=" * 50)
print("Classification Report:")
print(classification_report(val_labels, val_predictions))
print("=" * 50)

# store results in JSON format
results = {}

for i, file_path in enumerate(val_file_paths):
    # 從檔案路徑提取編號
    # 例如: "./train/aerosmith/aerosmith_001.wav" -> "001"
    filename = os.path.basename(file_path)  # "aerosmith_001.wav"
    file_number = filename.split('/')[-1].split('.')[0]  # "001"
    
    # 獲取 Top-3 預測結果
    top3_idx = np.argsort(val_probabilities[i])[-3:][::-1]  # 機率最高的3個類別索引
    top3_classes = [idx_to_class[idx] for idx in top3_idx]
    
    results[file_number] = top3_classes

# 儲存為 JSON 文件
output_file = './dataset/predictions.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"predictions saved to {output_file}")

print("\nshowing top 10 results:")
print(json.dumps(dict(list(results.items())[:10]), indent=4))

print("\n" + "=" * 80)
print("Detailed Top-3 Predictions for First 10 Validation Samples:")
print(f"{'ID':<6} {'True Label':<25} {'Top-3 Predictions (with probabilities)'}")
print("-" * 80)

for i, file_path in enumerate(val_file_paths[:10]):
    filename = os.path.basename(file_path)
    file_number = filename.split('_')[-1].split('.')[0]
    true_label = idx_to_class[val_labels[i]]
    
    top3_idx = np.argsort(val_probabilities[i])[-3:][::-1]
    top3_info = ", ".join([
        f"{idx_to_class[idx]}({val_probabilities[i][idx]:.3f})" 
        for idx in top3_idx
    ])
    
    print(f"{file_number:<6} {true_label:<25} {top3_info}")