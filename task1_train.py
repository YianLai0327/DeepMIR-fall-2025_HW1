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
import joblib

# build idx_to_class mapping
from dataloader import class_mapping

def extract_features_with_augmentation(dataset, n_augmentations=3):
    """
    對每個樣本提取多次特徵（帶隨機增強）
    """
    features = []
    labels = []
    
    for i in tqdm(range(len(dataset)), desc="Processing with augmentation"):
        for aug_idx in range(n_augmentations):
            mel_spectrogram, label = dataset[i]
            if aug_idx != 0:
                # 時間遮蔽 (Time Masking)
                if np.random.rand() > 0.5:
                    mel = mel_spectrogram.numpy().copy()
                    t_mask_width = np.random.randint(5, 20)
                    t_start = np.random.randint(0, max(1, mel.shape[2] - t_mask_width))
                    mel[:, :, t_start:t_start+t_mask_width] = 0
                    mel_spectrogram = torch.from_numpy(mel)
                
                # 頻率遮蔽 (Frequency Masking)
                if np.random.rand() > 0.5:
                    mel = mel_spectrogram.numpy().copy()
                    f_mask_width = np.random.randint(5, 15)
                    f_start = np.random.randint(0, max(1, mel.shape[1] - f_mask_width))
                    mel[:, f_start:f_start+f_mask_width, :] = 0
                    mel_spectrogram = torch.from_numpy(mel)
                
                # 加入輕微噪音
                if np.random.rand() > 0.5:
                    noise = np.random.normal(0, 0.01, mel_spectrogram.shape)
                    mel_spectrogram = mel_spectrogram + torch.from_numpy(noise).float()
            
            rich_feature = extract_rich_features(mel_spectrogram)
            features.append(rich_feature)
            labels.append(label)
    
    return np.array(features), np.array(labels)

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

n_augmentations = 3
data_augmentation = True  # Set to True to enable data augmentation

idx_to_class = {v: k for k, v in class_mapping.items()}

print("Loading training data...")
train_dataset = AudioDataset(
    json_list='./dataset/artist20/train.json',
    sr=16000,
    n_mels=256,
    hop_length=256,
    is_onehot=False,
    mode='train'
)

print("Loading validation data...")
val_dataset = AudioDataset(
    json_list='./dataset/artist20/val.json',
    sr=16000,
    n_mels=256,
    hop_length=256,
    is_onehot=False,
    mode='val'
)

print("Extracting training features...")
train_features = []
train_labels = []

if not data_augmentation:
    for i in tqdm(range(len(train_dataset)), desc="Processing training data"):
        mel_spectrogram, label = train_dataset[i]
        features = extract_rich_features(mel_spectrogram)
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
else:
    train_features, train_labels = extract_features_with_augmentation(train_dataset, n_augmentations=n_augmentations)
print("\nFeature extraction completed.")

print(f"Training features shape: {train_features.shape}")
print(f"Training labels shape: {train_labels.shape}")

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

print("\nApplying PCA...")

pca = PCA(n_components=0.95, random_state=42)
train_features = pca.fit_transform(train_features)

svm_classifier = SVC(
    kernel='rbf', 
    probability=True,
    C=10,
    gamma='scale',
    class_weight='balanced',
    verbose=True
)
svm_classifier.fit(train_features, train_labels)

# Show training performance
train_predictions = svm_classifier.predict(train_features)
train_probabilities = svm_classifier.predict_proba(train_features)

# stores model weights
import time
model_pth = f'./models/task1_model/{int(time.time())}.pkl'
os.makedirs(os.path.dirname(model_pth), exist_ok=True)
joblib.dump({
    'svm_classifier': svm_classifier,
    'scaler': scaler,
    'pca': pca
}, model_pth)

with open('./dataset/artist20/train.json', 'r') as f:
    # only load pred of original samples, not augmented ones
    train_json_paths = json.load(f)
    
train_result = {}

for i, file_path in enumerate(train_json_paths):
    j = i * n_augmentations  # only take the original sample's prediction
    filename = os.path.basename(file_path)
    file_number = filename.split('/')[-1].split('.')[0]
    top3_idx = np.argsort(train_probabilities[j])[-3:][::-1]
    top3_classes = [idx_to_class[idx] for idx in top3_idx]
    train_result[file_number] = top3_classes

train_result = dict(sorted(train_result.items(), key=lambda x: x[0]))
output_file = './dataset/train_predictions.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(train_result, f, indent=4, ensure_ascii=False)

print(f"\n✅ Training predictions saved to {output_file}")
count_score(gt_pth="./dataset/train_gt.json", pred_pth=output_file)

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
val_features = pca.transform(val_features)

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

results = dict(sorted(results.items(), key=lambda x: x[0]))

output_file = './dataset/val_predictions.json'
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

print("=" * 80)
print("\nValidation Set Performance:")
count_score(gt_pth="./dataset/val_gt.json", pred_pth=output_file)

# ============ 新增：Confusion Matrix 分析 ============
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 60)
print("CONFUSION MATRIX ANALYSIS")
print("=" * 60)

# 計算混淆矩陣
cm = confusion_matrix(val_labels, val_predictions)

# 1. 繪製原始混淆矩陣（計數）
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(class_mapping.keys()),
            yticklabels=list(class_mapping.keys()),
            cbar_kws={'label': 'Count'})

plt.title('Confusion Matrix - Validation Set', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_output = './dataset/confusion_matrix.png'
plt.savefig(cm_output, dpi=300, bbox_inches='tight')
print(f"\n✅ Confusion matrix saved to {cm_output}")
plt.close()

# # 2. 繪製正規化的混淆矩陣（百分比）
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plt.figure(figsize=(16, 14))
# sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
#             xticklabels=list(class_mapping.keys()),
#             yticklabels=list(class_mapping.keys()),
#             cbar_kws={'label': 'Proportion'},
#             vmin=0, vmax=1)

# plt.title('Normalized Confusion Matrix - Validation Set', fontsize=16, pad=20)
# plt.ylabel('True Label', fontsize=12)
# plt.xlabel('Predicted Label', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()

# cm_norm_output = './dataset/confusion_matrix_normalized.png'
# plt.savefig(cm_norm_output, dpi=300, bbox_inches='tight')
# print(f"✅ Normalized confusion matrix saved to {cm_norm_output}")
# plt.close()

# 3. 分析最容易混淆的類別對
print("\n" + "=" * 60)
print("TOP 15 MOST CONFUSED CLASS PAIRS")
print("=" * 60)

confused_pairs = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i][j] > 0:
            confused_pairs.append((
                idx_to_class[i], 
                idx_to_class[j], 
                cm[i][j],
                cm[i][j] / cm[i].sum() * 100
            ))

confused_pairs.sort(key=lambda x: x[2], reverse=True)

print(f"{'True Label':<30} {'Predicted As':<30} {'Count':<8} {'%'}")
print("-" * 80)
for true_cls, pred_cls, count, percent in confused_pairs[:15]:
    print(f"{true_cls:<30} {pred_cls:<30} {count:<8} {percent:.1f}%")

# 4. 每個類別的準確率
print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

class_accuracy = []
for i in range(len(cm)):
    correct = cm[i][i]
    total = cm[i].sum()
    accuracy = correct / total * 100 if total > 0 else 0
    class_accuracy.append((idx_to_class[i], accuracy, correct, total))

class_accuracy.sort(key=lambda x: x[1], reverse=True)

print(f"{'Class':<30} {'Accuracy':<12} {'Correct/Total'}")
print("-" * 60)
for cls, acc, correct, total in class_accuracy:
    print(f"{cls:<30} {acc:>6.2f}%      {correct}/{total}")

print(f"\nBest performing: {class_accuracy[0][0]} ({class_accuracy[0][1]:.2f}%)")
print(f"Worst performing: {class_accuracy[-1][0]} ({class_accuracy[-1][1]:.2f}%)")

# 5. 繪製每個類別的準確率長條圖
plt.figure(figsize=(14, 8))
classes = [x[0] for x in class_accuracy]
accuracies = [x[1] for x in class_accuracy]

bars = plt.bar(range(len(classes)), accuracies, color='steelblue', alpha=0.8)

# 為低於平均的類別標紅色
mean_acc = np.mean(accuracies)
for i, (cls, acc, _, _) in enumerate(class_accuracy):
    if acc < mean_acc:
        bars[i].set_color('coral')

plt.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.1f}%')
plt.xlabel('Artist', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy - Validation Set', fontsize=14, pad=20)
plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

acc_bar_output = './dataset/per_class_accuracy.png'
plt.savefig(acc_bar_output, dpi=300, bbox_inches='tight')
print(f"\n✅ Per-class accuracy chart saved to {acc_bar_output}")
plt.close()

print("\n" + "=" * 60)
print("CONFUSION MATRIX ANALYSIS COMPLETE")
print("=" * 60)