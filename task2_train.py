import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import create_dataloader
from tqdm import tqdm, trange
from task2_model import MusicCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
n_epochs = 100
batch_size = 8
learning_rate = 0.001
weight_decay = 1e-4
n_mels = 128
hop_length = 512
num_classes = 20
accumulation_steps = 4  # For gradient accumulation

# DataLoaders
train_loader = create_dataloader(
    json_list='./dataset/artist20/train.json',
    batch_size=batch_size,
    num_workers=4,
    sr=16000,
    n_mels=n_mels,
    hop_length=hop_length,
    is_onehot=False,
    mode='train'
)
print(f"Number of training batches: {len(train_loader)}")

val_loader = create_dataloader(
    json_list='./dataset/artist20/val.json',
    batch_size=batch_size,
    num_workers=4,
    sr=16000,
    n_mels=n_mels,
    hop_length=hop_length,
    is_onehot=False,
    mode='val'
)
print(f"Number of validation batches: {len(val_loader)}")
print("Successfully created dataloaders.")

# Model
print("Initializing model...")
model = MusicCNN(n_classes=num_classes, n_mels=n_mels).to(device)

print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
from collections import Counter
train_labels = []
for _, _, labels in train_loader:
    train_labels.extend(labels.numpy())

class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([total / class_counts[i] for i in range(num_classes)]).to(device)

# Loss
print("Setting up loss function...")
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()

# Optimizer and Scheduler
print("Setting up optimizer and scheduler...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# Training loop
best_val_acc = 0.0

print("Starting training...")
train_pbar = trange(n_epochs, desc="Overall Training Progress", leave=True)
for epoch in train_pbar:
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    it = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Train", leave=True)
    for mel_spec, lengths, labels in train_pbar:
        mel_spec = mel_spec.to(device)
        labels = labels.to(device)
        mel_spec = mel_spec.transpose(0, 1)
        B, C, F, T = mel_spec.shape
        if B != batch_size and C != 1:
            mel_spec = mel_spec.transpose(0, 1)
        # print(mel_spec.shape, labels.shape)  # Debugging line to check shapes
        
        # Forward
        outputs = model(mel_spec)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()

        # Gradient accumulation
        if (it + 1) % accumulation_steps == 0:
            optimizer.zero_grad()
            optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        it += 1

        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Val", leave=True)
        for mel_spec, lengths, labels in val_pbar:
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