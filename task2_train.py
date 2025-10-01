import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import create_dataloader
from tqdm import tqdm
from task2_model import MusicCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
n_epochs = 100
batch_size = 32
learning_rate = 0.001
weight_decay = 1e-4

# DataLoaders
train_loader = create_dataloader(
    json_list='./dataset/artist20/train.json',
    batch_size=batch_size,
    num_workers=4,
    sr=22050,
    n_mels=128,
    hop_length=512,
    is_onehot=False,
    mode='train'
)

val_loader = create_dataloader(
    json_list='./dataset/artist20/val.json',
    batch_size=batch_size,
    num_workers=4,
    sr=22050,
    n_mels=128,
    hop_length=512,
    is_onehot=False,
    mode='val'
)

# Model
model = MusicCNN(n_classes=20, n_mels=128).to(device)

# Loss
from collections import Counter

train_labels = []
for _, _, labels in train_loader:
    train_labels.extend(labels.numpy())

class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([total / class_counts[i] for i in range(20)]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# Training loop
best_val_acc = 0.0

for epoch in range(n_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for mel_spec, lengths, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Train"):
        mel_spec = mel_spec.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(mel_spec)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for mel_spec, lengths, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Val"):
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)
            
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\nEpoch {epoch+1}/{n_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, './models/best_cnn_model.pth')
        print(f"Saved best model with val acc: {val_acc:.2f}%")

print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")