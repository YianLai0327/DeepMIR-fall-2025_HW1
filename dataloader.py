import librosa as lr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchaudio_augmentations import *
import json


# mapping class names to integers
class_mapping = {
    'aerosmith': 0,
    'beatles': 1,
    'creedence_clearwater_revival': 2,
    'cure': 3,
    'dave_matthews_band': 4,
    'depeche_mode': 5,
    'fleetwood_mac': 6,
    'garth_brooks': 7,
    'green_day': 8,
    'led_zeppelin': 9,
    'madonna': 10,
    'metallica': 11,
    'prince': 12,
    'queen': 13,
    'radiohead': 14,
    'roxette': 15,
    'steely_dan': 16,
    'suzanne_vega': 17,
    'tori_amos': 18,
    'u2': 19
}

class AudioDataset(Dataset):
    """
    Extract log mel-spectrogram features from audio files and create a dataset.
    """
    def __init__(self, json_list, sr=22050, n_mels=128, hop_length=512, is_onehot=False, mode='train'):
        """
        Args:
            file_list (list): List of paths to audio files.
            sr (int): Sampling rate for loading audio.
            n_mels (int): Number of mel bands to generate.
            hop_length (int): Number of samples between successive frames.
        """
        self.json_list = json_list
        self.audios = []
        self.labels = []
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.mode = mode

        print("Loading audio files and extracting features...")
        print(f"reading from {json_list}")

        if not os.path.exists(json_list):
            raise ValueError(f"{json_list} is not a valid file path")
        
        with open(json_list, 'r') as f:
            datas = json.load(f)
        
        for audio_path in datas:
            real_audio_path = audio_path.replace('./', './dataset/artist20/')
            if not os.path.isfile(real_audio_path):
                print(f"Warning: {real_audio_path} does not exist. Skipping.")
                continue
            self.audios.append(real_audio_path)
            label = audio_path.split('/')[-3]
            print(f"Processing {real_audio_path}, label: {label}")
            if is_onehot:
                onehot = np.zeros(len(class_mapping), dtype=np.float32)
                onehot[class_mapping[label]] = 1.0
                self.labels.append(onehot)
            else:
                self.labels.append(class_mapping[label])
        
        if len(self.audios) == 0:
            raise ValueError(f"No audio files found in {json_list}")

        print(f"Found {len(self.audios)} audio files in {json_list}")

        if self.mode == 'train':
            # Apply data augmentation for training set
            transforms = [
                RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
                RandomApply([Gain()], p=0.2),
                RandomApply([HighLowPass(sample_rate=self.sr)], p=0.8),
                RandomApply([Delay(sample_rate=self.sr)], p=0.5),
                RandomApply([PitchShift(n_samples=1, sample_rate=self.sr)], p=0.4),
                RandomApply([Reverb(sample_rate=self.sr)], p=0.3),
            ]
            self.augmentation = Compose(transforms=transforms)

    def __len__(self):
        return len(self.audios)

    def collate_fn(self, batch):
        """
        Collate function to pad sequences to the same length in a batch.
        
        Args:
            batch (list): List of log mel-spectrograms.
            
        Returns:
            torch.Tensor: Padded batch of log mel-spectrograms.
            np.ndarray: Original lengths of each spectrogram in the batch.
            torch.Tensor: Labels corresponding to each spectrogram.
        """
        # Find the maximum length in the batch
        max_length = max(spectrogram.shape[1] for spectrogram, _ in batch)
        
        # Initialize a padded batch with zeros
        padded_batch = np.zeros((len(batch), batch[0][0].shape[0], max_length), dtype=np.float32)
        lengths = np.zeros(len(batch), dtype=np.int64)
        labels = []

        for i, (spectrogram, label) in enumerate(batch):
            length = spectrogram.shape[1]
            padded_batch[i, :, :length] = spectrogram
            lengths[i] = length
            labels.append(label)
        
        labels = torch.stack(labels)

        return torch.tensor(padded_batch, dtype=torch.float32), lengths, labels

    def __getitem__(self, idx):
        # Load the audio file
        audio_path = self.audios[idx]
        y, _ = lr.load(audio_path, sr=self.sr)
        # Apply data augmentation if in training mode
        if self.mode == 'train':
            y = self.augmentation(torch.tensor(y).unsqueeze(0)).squeeze(0).numpy()
        # Compute the log mel-spectrogram
        mel_spectrogram = lr.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
        log_mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=np.max)
        label = self.labels[idx]
        
        # covert to tensor
        log_mel_spectrogram = torch.tensor(log_mel_spectrogram, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32) if isinstance(label, np.ndarray) else torch.tensor(label, dtype=torch.long)

        return log_mel_spectrogram, label

def create_dataloader(json_list, batch_size=32, shuffle=True, num_workers=4, sr=22050, n_mels=128, hop_length=512, is_onehot=False, mode='train'):
    dataset = AudioDataset(json_list=json_list, sr=sr, n_mels=n_mels, hop_length=hop_length, is_onehot=is_onehot, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers, collate_fn=dataset.collate_fn)
    return dataloader
    

if __name__ == "__main__":
    # Example usage
    train_path = "dataset/artist20/train.json"
    val_path = "dataset/artist20/val.json"
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Dataset not found. Please ensure the dataset is available at the specified paths.")
    train_loader = create_dataloader(json_list='./dataset/artist20/train.json', batch_size=16, mode='train')
    val_loader = create_dataloader(json_list='./dataset/artist20/val.json', batch_size=16, mode='val')

    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))
    print("Successfully created dataloaders.")

    for batch in train_loader:
        inputs, lengths, labels = batch
        print(f"Input shape: {inputs.shape}, Lengths: {lengths}, Labels: {labels}")
        break

    print("DataLoader test completed.")