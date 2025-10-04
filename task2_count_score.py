import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from dataloader import create_dataloader
from tqdm import tqdm, trange
import os
from task2_model import MusicCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16
n_mels = 128
hop_length = 512
num_classes = 20
chunk_duration = 30.0  # seconds



result = {} # stores top1 and top3 predictions of 

model = MusicCNN(n_classes=num_classes, n_mels=n_mels).to(device)
# Load the best model
model_pth = "./models/task2/best_cnn_model.pth"
state_dict = torch.load(model_pth, map_location=device)
model_weight = state_dict['model_state_dict']
model.load_state_dict(model_weight)
print(f"Loaded model from {model_pth}")

model.eval()
top1_correct = 0
val_total = 0
top3_correct = 0

with torch.no_grad():
    val_pbar = tqdm(val_loader, leave=True)
    for mel_spec, _, labels in val_pbar:
        mel_spec = mel_spec.to(device)
        labels = labels.to(device)

        if mel_spec.shape[1] != 1:
            mel_spec = mel_spec.transpose(0, 1)
        
        outputs = model(mel_spec)
        
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        # top1 accuracy
        top1_correct += (predicted == labels).sum().item()
        # top3 accuracy
        top3_correct += sum([1 if labels[i] in outputs[i].topk(3).indices else 0 for i in range(labels.size(0))])
    
top1_acc = 100 * top1_correct / val_total
top3_acc = 100 * top3_correct / val_total

print(f"Top1 Acc: {top1_acc:.2f}, Top3 Acc: {top3_acc:.2f}%")