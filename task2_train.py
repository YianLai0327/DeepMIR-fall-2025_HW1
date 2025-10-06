import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import create_dataloader, MixedAudioDataset
from tqdm import tqdm, trange
import os
from task2_model import MusicCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
n_epochs = 50
batch_size = 16
learning_rate = 0.001
weight_decay = 1e-4
n_mels = 128
hop_length = 512
num_classes = 20
accumulation_steps = 4  # For gradient accumulation
chunk_duration = 30.0  # seconds
chunk_overlap = 0.5  # 50% overlap for training

# DataLoaders
train_loader = create_dataloader(
    # json_list='./dataset/train_vocal.json',
    json_list='./dataset/artist20/train.json',
    batch_size=batch_size,
    num_workers=4,
    sr=16000,
    n_mels=n_mels,
    hop_length=hop_length,
    chunk_duration=chunk_duration,
    overlap=chunk_overlap, 
    is_onehot=False,
    mode='train'
)
# train_loader = MixedAudioDataset.create_mixed_dataloader(
#     vocal_json="dataset/train_vocal.json",
#     full_json="dataset/artist20/train.json",
#     batch_size=16,
#     chunk_duration=30.0,
#     overlap=0.5,
#     vocal_ratio=0.5,
#     mode='train'
# )

val_loader = create_dataloader(
    json_list='./dataset/artist20/val.json',
    batch_size=batch_size,
    num_workers=4,
    sr=16000,
    n_mels=n_mels,
    hop_length=hop_length,
    chunk_duration=chunk_duration,
    overlap=0.0, 
    is_onehot=False,
    mode='val'
)
print(f"Number of validation batches: {len(val_loader)}")
print("Successfully created dataloaders.")

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

sample_batch = next(iter(train_loader))
print(f"Sample batch shape: {sample_batch[0].shape}")
print(f"Expected time frames: ~{int(chunk_duration * 16000 / hop_length)}")

# Model
print("Initializing model...")
model = MusicCNN(n_classes=num_classes, n_mels=n_mels).to(device)

print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# from collections import Counter
# train_labels = []
# for _, _, labels in train_loader:
#     train_labels.extend(labels.numpy())

# class_counts = Counter(train_labels)
# total = len(train_labels)
# class_weights = torch.tensor([total / class_counts[i] for i in range(num_classes)]).to(device)

# Loss
print("Setting up loss function...")
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()

# Optimizer and Scheduler
print("Setting up optimizer and scheduler...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

#  Warmup scheduler
# from torch.optim.lr_scheduler import LinearLR, SequentialLR
# warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
# main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

optimizer.zero_grad()
# Training loop
best_val_acc = 0.0

print("Starting training...")
epoch_pbar = trange(n_epochs, desc="Overall Training Progress", leave=True)
for epoch in epoch_pbar:
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Train", leave=True)
    for it, (mel_spec, _, labels) in enumerate(train_pbar):
        mel_spec = mel_spec.to(device)
        labels = labels.to(device)
        # shape is (batch, 1, n_mels, time)
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)  # (batch, n_mels, time) -> (batch, 1, n_mels, time)
        elif mel_spec.shape[1] != 1:
            mel_spec = mel_spec.transpose(0, 1)
        # print(mel_spec.shape, labels.shape)  # Debugging line to check shapes
        
        # Forward
        outputs = model(mel_spec)
        loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss for gradient accumulation
        
        # Backward
        loss.backward()

        # Gradient accumulation
        if (it + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Statistics
        train_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        train_pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'acc': f'{100 * train_correct / train_total:.2f}%'
        })
    
    if (it + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Val", leave=True)
        for mel_spec, _, labels in val_pbar:
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)

            if mel_spec.shape[1] != 1:
                mel_spec = mel_spec.transpose(0, 1)
            
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\nEpoch {epoch+1}/{n_epochs}:")
    print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Save best model
    
    os.makedirs('./models/task2', exist_ok=True)
    if val_acc > best_val_acc and val_acc > 60.0:  # Save only if val_acc > 60%
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': {
                'chunk_duration': chunk_duration,
                'n_mels': n_mels,
                'hop_length': hop_length,
            }
        }, './models/task2/best_cnn_model.pth')
        print(f"Saved best model with val acc: {val_acc:.2f}%")

    if optimizer.param_groups[0]['lr'] < 1e-6:
        print("Learning rate too small, stopping training.")
        break

print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")