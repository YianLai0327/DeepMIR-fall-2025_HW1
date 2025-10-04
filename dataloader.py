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
from torch.utils.data import Dataset, ConcatDataset

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
    def __init__(self, json_list, sr=22050, n_mels=128, hop_length=512, chunk_duration=10.0, overlap=0.5, is_onehot=False, mode='train'):
        """
        Args:
            file_list (list): List of paths to audio files.
            sr (int): Sampling rate for loading audio.
            n_mels (int): Number of mel bands to generate.
            hop_length (int): Number of samples between successive frames.
        """
        self.json_list = json_list
        # self.audios = []
        # self.labels = []
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.mode = mode
        self.chunk_duration = chunk_duration
        self.overlap = overlap

        self.chunk_samples = int(chunk_duration * sr)
        self.hop_samples = int(self.chunk_samples * (1 - overlap))

        self.chunks = []  # List to hold (audio_path, start_sample, label) tuples

        print(f"Loading audio files from {json_list}")
        print(f"Chunk duration: {chunk_duration}s, Overlap: {overlap}")

        print("Loading audio files and extracting features...")
        print(f"reading from {json_list}")

        if not os.path.exists(json_list):
            raise ValueError(f"{json_list} is not a valid file path")
        
        with open(json_list, 'r') as f:
            datas = json.load(f)

        is_vocal = 'vocal' in json_list
        
        for audio_path in datas:
            real_audio_path = audio_path if is_vocal else audio_path.replace('./', './dataset/artist20/')

            if not os.path.isfile(real_audio_path):
                print(f"Warning: {real_audio_path} does not exist. Skipping.")
                continue

            if is_vocal:
                label = audio_path.split('/')[-2].split('-')[0]
                label = label[:-3]
                label_idx = class_mapping[label]
            else: 
                label = audio_path.split('/')[-3]
                label_idx = class_mapping[label]

            print(f"Processing {real_audio_path}, label: {label}")

            info = torchaudio.info(real_audio_path)
            audio_length = info.num_frames

            if mode == 'train':
                # For training, extract overlapping chunks
                for start in range(0, audio_length - self.chunk_samples + 1, self.hop_samples):
                    self.chunks.append((real_audio_path, start, label_idx))
                
                # Make sure to include the last chunk if it doesn't fit perfectly
                if audio_length > self.chunk_samples:
                    last_start = audio_length - self.chunk_samples
                    if last_start > 0 and (last_start not in [c[1] for c in self.chunks if c[0] == real_audio_path]):
                        self.chunks.append((real_audio_path, last_start, label_idx))
            else:
                # For validation/test, just take non-overlapping chunks
                num_chunks = max(1, audio_length // self.chunk_samples)
                for i in range(num_chunks):
                    start = i * self.chunk_samples
                    if start + self.chunk_samples <= audio_length:
                        self.chunks.append((real_audio_path, start, label_idx))
        
        # if len(self.audios) == 0:
        #     raise ValueError(f"No audio files found in {json_list}")

        print(f"Created {len(self.chunks)} chunks from {len(datas)} audio files")
        print(f"Average chunks per song: {len(self.chunks) / len(datas):.2f}")

        if self.mode == 'train':
            # Apply data augmentation for training set
            transforms = [
                RandomApply([HighLowPass(sample_rate=self.sr)], p=0.5),
                RandomApply([Gain()], p=0.3),
            ]
            self.augmentation = Compose(transforms=transforms)
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.chunks)

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
        spectrograms = []
        labels = []
        
        for spectrogram, label in batch:
            # 確保是3D: (1, n_mels, time)
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)
            spectrograms.append(spectrogram)
            labels.append(label)
        
        # Stack所有spectrograms (batch, 1, n_mels, time)
        spectrograms = torch.stack(spectrograms)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # 返回格式: (spectrograms, lengths, labels)
        # lengths全部相同，所以可以省略或返回None
        time_frames = spectrograms.shape[-1]
        lengths = np.array([time_frames] * len(batch))
        
        return spectrograms, lengths, labels

    def _get_mel_spectrogram(self, waveform):
        """Get log mel-spectrogram from waveform."""
        mel_spectrogram = MelSpectrogram(
            sample_rate=self.sr, 
            n_mels=self.n_mels, 
            hop_length=self.hop_length, 
            n_fft=2048
        )(waveform)
        log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        return log_mel_spectrogram

    def __getitem__(self, idx):
        # Load the audio file and compute the mel-spectrogram
        audio_path, start_sample, label = self.chunks[idx]
        
        # Load the audio chunk
        waveform, sample_rate = torchaudio.load(
            audio_path,
            frame_offset=start_sample,
            num_frames=self.chunk_samples,
            normalize=True
        )
        
        # Resample if needed
        if sample_rate != self.sr:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[1] != self.chunk_samples:
            if waveform.shape[1] < self.chunk_samples:
                # Pad
                padding = self.chunk_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                # Crop
                waveform = waveform[:, :self.chunk_samples]
        
        # Apply augmentation if in training mode
        if self.mode == 'train' and self.augmentation is not None:
            waveform = self.augmentation(waveform)
        
        # calculate mel spectrogram
        log_mel_spectrogram = self._get_mel_spectrogram(waveform)
        
        return log_mel_spectrogram, label


class MixedAudioDataset:
    """
    Create a mixed dataloader combining vocal-only and full mix datasets.
    """
    @staticmethod
    def create_mixed_dataloader(
        vocal_json, 
        full_json, 
        batch_size=16,
        sr=16000,
        n_mels=128,
        hop_length=512,
        chunk_duration=30.0,
        overlap=0.5,
        vocal_ratio=0.5,  # Vocal 數據的比例
        mode='train'
    ):
        """
        Args:
            vocal_json (str): Path to JSON file for vocal-only dataset.
            full_json (str): Path to JSON file for full mix dataset.
            batch_size (int): Batch size for DataLoader.
            sr (int): Sampling rate.
            n_mels (int): Number of mel bands.
            hop_length (int): Hop length for mel-spectrogram.
            chunk_duration (float): Duration of each audio chunk in seconds.
            overlap (float): Overlap ratio between chunks.
            vocal_ratio (float): Proportion of vocal data in each batch.
            mode (str): 'train' or 'val'.
        """
        
        # 創建 vocal dataset
        vocal_dataset = AudioDataset(
            json_list=vocal_json,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            chunk_duration=chunk_duration,
            overlap=overlap if mode == 'train' else 0.0,
            mode=mode
        )
        
        # 創建 full mix dataset
        full_dataset = AudioDataset(
            json_list=full_json,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            chunk_duration=chunk_duration,
            overlap=overlap if mode == 'train' else 0.0,
            mode=mode
        )
        
        print(f"\n{'='*60}")
        print(f"Mixed Dataset Statistics ({mode}):")
        print(f"  Vocal chunks: {len(vocal_dataset)}")
        print(f"  Full mix chunks: {len(full_dataset)}")
        
        # 根據 ratio 調整採樣
        if mode == 'train':
            # 方法 1: Concat 兩個 dataset（簡單）
            combined_dataset = ConcatDataset([vocal_dataset, full_dataset])
            
            # 方法 2: 使用 WeightedRandomSampler 控制比例（更精確）
            from torch.utils.data import WeightedRandomSampler
            
            # 為每個樣本分配權重
            vocal_weight = vocal_ratio
            full_weight = 1 - vocal_ratio
            
            weights = [vocal_weight] * len(vocal_dataset) + \
                     [full_weight] * len(full_dataset)
            
            # sampler = WeightedRandomSampler(
            #     weights=weights,
            #     num_samples=len(combined_dataset),
            #     replacement=True
            # )
            
            # dataloader = torch.utils.data.DataLoader(
            #     combined_dataset,
            #     batch_size=batch_size,
            #     sampler=sampler,
            #     num_workers=4,
            #     collate_fn=vocal_dataset.collate_fn,
            #     pin_memory=True
            # )
            dataloader = torch.utils.data.DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=vocal_dataset.collate_fn,
                pin_memory=True
            )
            
            print(f"  Combined: {len(combined_dataset)} chunks")
            print(f"  Vocal ratio: {vocal_ratio:.0%}")
            print(f"{'='*60}\n")
            
        else:
            # 驗證集：簡單 concat
            combined_dataset = ConcatDataset([vocal_dataset, full_dataset])
            dataloader = torch.utils.data.DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=vocal_dataset.collate_fn,
                pin_memory=True
            )
            print(f"  Combined: {len(combined_dataset)} chunks")
            print(f"{'='*60}\n")
        
        return dataloader


def create_dataloader(json_list, batch_size=32, num_workers=4, sr=16000, 
                     n_mels=128, hop_length=512, chunk_duration=10.0, 
                     overlap=0.5, is_onehot=False, mode='train'):
    dataset = AudioDataset(json_list=json_list, sr=sr, 
                           n_mels=n_mels, hop_length=hop_length,
                           chunk_duration=chunk_duration, 
                           overlap=overlap, 
                           is_onehot=is_onehot, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers, collate_fn=dataset.collate_fn, pin_memory=True)
    return dataloader
    

if __name__ == "__main__":
    # Example usage
    train_path = "dataset/train_vocal.json"
    val_path = "dataset/val_vocal.json"
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Dataset not found. Please ensure the dataset is available at the specified paths.")
    
    # for chunk_dur in [10, 15, 30]:
    #     print(f"\n{'='*50}")
    #     print(f"Testing with chunk_duration={chunk_dur}s")
    #     print(f"{'='*50}")
        
    #     train_loader = create_dataloader(
    #         json_list=train_path, 
    #         batch_size=16, 
    #         chunk_duration=chunk_dur,
    #         overlap=0.5,
    #         mode='train'
    #     )
        
    #     print("Number of training batches:", len(train_loader))
        
    #     for batch in train_loader:
    #         inputs, lengths, labels = batch
    #         print("Input shape:", inputs.shape)
    #         print("Time frames:", inputs.shape[-1])
    #         print("Estimated frames for full song (180s):", int(180 / chunk_dur) * inputs.shape[-1])
    #         break

    train_loader = MixedAudioDataset.create_mixed_dataloader(
        vocal_json="dataset/train_vocal.json",
        full_json="dataset/artist20/train.json",
        batch_size=16,
        chunk_duration=30.0,
        overlap=0.5,
        vocal_ratio=0.5,
        mode='train'
    )

    val_loader = create_dataloader(
        json_list=val_path,
        batch_size=16,
        chunk_duration=30.0,
        overlap=0.0,
        mode='val'
    )

    for batch in train_loader:
        inputs, lengths, labels = batch
        print("Train Input shape:", inputs.shape)
        print("Train Time frames:", inputs.shape[-1])
        print("Estimated frames for full song (180s):", int(180 / 30) * inputs.shape[-1])
        break