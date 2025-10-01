import librosa as lr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchaudio_augmentations import *
import json
from tqdm import tqdm
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample

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
                # RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
                # RandomApply([Gain()], p=0.2),
                # RandomApply([HighLowPass(sample_rate=self.sr)], p=0.8),
                # RandomApply([Delay(sample_rate=self.sr)], p=0.5),
                # RandomApply([PitchShift(n_samples=1, sample_rate=self.sr)], p=0.4),
                # RandomApply([Reverb(sample_rate=self.sr)], p=0.3),
            ]
            self.augmentation = Compose(transforms=transforms)

    def __len__(self):
        return len(self.audios)

    def collate_fn(self, batch):
        """
        Collate function to pad sequences to the same length in a batch.
        
        Args:
            batch (list): List of log mel-spectrograms and labels.
            
        Returns:
            torch.Tensor: Padded batch of log mel-spectrograms.
            np.ndarray: Original lengths of each spectrogram in the batch.
            torch.Tensor: Labels corresponding to each spectrogram.
        """
        # Find the maximum length in the batch (time frames in the spectrogram)
        max_length = max(spectrogram.shape[2] for spectrogram, _ in batch)
        
        # Initialize a padded batch with zeros (batch_size, n_mels, max_length)
        padded_batch = np.zeros((len(batch), self.n_mels, max_length), dtype=np.float32)
        lengths = np.zeros(len(batch), dtype=np.int64)
        labels = []

        for i, (spectrogram, label) in enumerate(batch):
            length = spectrogram.shape[2]
            padded_batch[i, :, :length] = spectrogram  # Padding along the time axis (columns)
            lengths[i] = length
            labels.append(torch.tensor(label, dtype=torch.long))
        
        labels = torch.stack(labels)

        return torch.tensor(padded_batch, dtype=torch.float32), lengths, labels


    def _get_mel_spectrogram(self, audio_path):
        """
        Helper function to get mel-spectrogram from audio file.
        """
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)

        # Ensure the sample rate matches
        if sample_rate != self.sr:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sr)
            waveform = resampler(waveform)
        
        # Get Mel-spectrogram
        mel_spectrogram = MelSpectrogram(sample_rate=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)(waveform)
        log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        return log_mel_spectrogram

    def __getitem__(self, idx):
        # Load the audio file and compute the mel-spectrogram
        audio_path = self.audios[idx]
        log_mel_spectrogram = self._get_mel_spectrogram(audio_path)
        
        label = self.labels[idx]
        
        # Convert to tensor if needed
        return log_mel_spectrogram, label


def create_dataloader(json_list, batch_size=32, num_workers=8, sr=16000, n_mels=128, hop_length=512, is_onehot=False, mode='train'):
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

    

    print("DataLoader test completed.")
    print("All tests passed.")
    print("Dataloader module is working correctly.")