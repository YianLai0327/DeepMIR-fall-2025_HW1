import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB
import numpy as np
from collections import Counter
from task2_model import MusicCNN
from tqdm import tqdm

class SingerClassifier:
    """
    classify singer from a full song audio file using a trained CNN model.
    """
    def __init__(self, model_path, chunk_duration=15.0, stride_ratio=0.5, 
                 sr=16000, n_mels=128, hop_length=512, device='cuda'):
        """
        Args:
            model_path: path to the trained model
            chunk_duration: length of each chunk in seconds
            stride_ratio: stride ratio for overlapping chunks (0 < stride_ratio <= 1)
            sr, n_mels, hop_length: parameters for mel spectrogram
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sr)
        self.stride_samples = int(self.chunk_samples * stride_ratio)
        assert 0 < stride_ratio <= 1, "stride_ratio must be in (0, 1]"
        
        # load model
        self.model = MusicCNN(n_classes=20, n_mels=n_mels).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Mel spectrogram 轉換
        self.mel_transform = MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=2048
        )
        self.amplitude_to_db = AmplitudeToDB()
        
        # 類別名稱
        self.class_names = [
            'aerosmith', 'beatles', 'creedence_clearwater_revival', 'cure',
            'dave_matthews_band', 'depeche_mode', 'fleetwood_mac', 'garth_brooks',
            'green_day', 'led_zeppelin', 'madonna', 'metallica', 'prince',
            'queen', 'radiohead', 'roxette', 'steely_dan', 'suzanne_vega',
            'tori_amos', 'u2'
        ]
        
        print(f"Loaded model from {model_path}")
        print(f"Chunk duration: {chunk_duration}s, Stride ratio: {stride_ratio}")
        print(f"Device: {self.device}")
    
    def _extract_mel_spectrogram(self, waveform):
        """extract log mel spectrogram"""
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        return log_mel_spec
    
    def _load_audio(self, audio_path):
        """load audio file and resample to target sr"""
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        
        # Resample if needed
        if sample_rate != self.sr:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sr)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def predict_song(self, audio_path):
        """
        predict singer from a full song audio file
        
        Args:
            audio_path: path to the audio file
        
        Returns:
            predicted_class (int): idx of the predicted singer
            predicted_name (str): name of the predicted singer
            confidence (float): confidence score (0-1)
            details (dict): detailed info about chunks and predictions
        """
        waveform = self._load_audio(audio_path)
        total_samples = waveform.shape[1]
        
        # cut into chunks
        chunks = []
        positions = [] 
        
        start = 0
        while start + self.chunk_samples <= total_samples:
            chunk = waveform[:, start:start + self.chunk_samples]
            chunks.append(chunk)
            positions.append(start / total_samples)
            start += self.stride_samples

        # last chunk handle
        if start < total_samples and total_samples >= self.chunk_samples:
            chunk = waveform[:, -self.chunk_samples:]
            chunks.append(chunk)
            positions.append((total_samples - self.chunk_samples) / total_samples)
        
        # handle very short audio
        if len(chunks) == 0:
            chunk = waveform
            if chunk.shape[1] < self.chunk_samples:
                padding = self.chunk_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            chunks.append(chunk)
            positions.append(0.0)
        
        print(f"Audio length: {total_samples / self.sr:.2f}s")
        print(f"Number of chunks: {len(chunks)}")
        
        # predict each chunk
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for chunk in chunks:
                mel_spec = self._extract_mel_spectrogram(chunk)
                
                # Add batch dimension: (1, 1, n_mels, time)
                if mel_spec.dim() == 2:
                    mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
                elif mel_spec.dim() == 3:
                    mel_spec = mel_spec.unsqueeze(0)
                
                mel_spec = mel_spec.to(self.device)
                
                output = self.model(mel_spec)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                
                all_predictions.append(prediction.item())
                all_probabilities.append(probabilities.cpu().numpy()[0])
        
        # aggregate predictions
        avg_probs = np.mean(all_probabilities, axis=0)
        predicted_class = np.argmax(avg_probs)
        confidence = avg_probs[predicted_class]
        predicted_top3_class = np.argsort(avg_probs)[-3:][::-1]
        
        predicted_name = self.class_names[predicted_class]
        predicted_top3_names = [self.class_names[i] for i in predicted_top3_class]
        
        # 詳細資訊
        details = {
            'num_chunks': len(chunks),
            'chunk_predictions': [self.class_names[p] for p in all_predictions],
            'chunk_confidences': [probs[pred] for probs, pred in zip(all_probabilities, all_predictions)],
            'audio_duration': total_samples / self.sr
        }
        
        return predicted_class, predicted_name, predicted_top3_class, predicted_top3_names, confidence, details
    
    def _get_position_weights(self, positions):
        """
        get weights based on chunk positions using a Gaussian centered at 0.5
        """
        positions = np.array(positions)
        weights = np.exp(-((positions - 0.5) ** 2) / (2 * 0.2 ** 2))
        weights = weights / np.sum(weights)
        return weights
    
    def predict_batch(self, audio_paths):
        """
        batch predict multiple audio files
        
        Args:
            audio_paths: list of audio file paths
        
        Returns:
            results: list of dicts with prediction results for each file
        """
        top1_result = []
        top3_results = {}
        results = []
        total = len(audio_paths)
        top1_acc = 0.0
        top3_acc = 0.0

        for audio_path in tqdm(audio_paths, desc="Batch Predicting"):
            label = None
            is_vocal = 'vocal' in audio_path
            if "test" not in audio_path:
                label = audio_path.split('/')[-2].split('-')[0][:-3] if is_vocal else audio_path.split('/')[-3]
                # label = label[:-3]
                print(f"Ground truth: {label}")

            real_path = audio_path if is_vocal else audio_path.replace('./', './dataset/artist20/')
            # real_path = audio_path.replace('./', './dataset/artist20/')
            print(f"Processing: {real_path}")
            audio_name = real_path.split('/')[-1].split('.')[0]
            try:
                pred_class, pred_name, pred_top3_class, pred_top3_name, confidence, details = self.predict_song(real_path)
                results.append({
                    'path': audio_path,
                    'predicted_class': int(pred_class),
                    'predicted_name': pred_name,
                    'predicted_top3_class': [int(c) for c in pred_top3_class],
                    'predicted_top3_name': pred_top3_name,
                    'confidence': float(confidence),
                    'details': details
                })
                top1_result.append(pred_name)
                top3_results[audio_name] = pred_top3_name
                if label is not None:
                    top1_acc += (pred_name == label)
                    top3_acc += (label in pred_top3_name)

            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                results.append({
                    'path': audio_path,
                    'error': str(e)
                })
        
        return results, top1_result, top3_results, top1_acc / total, top3_acc / total

if __name__ == "__main__":
    classifier = SingerClassifier(
        model_path='./models/task2/full_mix_w_vocal.pth',
        chunk_duration=30.0,  # 與訓練時一致
        stride_ratio=0.5,     # 50% 重疊
        device='cuda'
    )
    
    # audio_path = "./dataset/artist20/train_val/aerosmith/Pump/01-Young_Lust.mp3"

    # print("\n" + "="*60)
    # print("Inference")
    # print("="*60)

    # pred_class, pred_name, pred_top3_class, pred_top3_name, confidence, details = classifier.predict_song(audio_path)
    
    # print(f"Predicted: {pred_name} (class {pred_class})")
    # print(f"Top-3 Predictions: {pred_top3_name} (classes {pred_top3_class})")
    # print(f"Confidence: {confidence:.2%}")
    # print(f"Chunks analyzed: {details['num_chunks']}")
    # print(f"Chunk predictions: {details['chunk_predictions'][:5]}...")  # 顯示前5個
    
    # # 批次預測
    print("\n" + "="*60)
    print("Batch prediction:")
    print("="*60)
    
    import json
    input_audios = "./dataset/artist20/val.json"

    top1_pred_json = "./dataset/task2_val_top1_predictions_full_w_vocal.json"
    top3_pred_json = "./dataset/task2_val_top3_predictions_full_w_vocal.json"

    with open(input_audios, 'r') as f:
        datas = json.load(f)
    
    results, top1_result, top3_results, top1_acc, top3_acc = classifier.predict_batch(datas)

    if top1_acc > 0 and top3_acc > 0:
        print(f"Top-1 Accuracy: {top1_acc:.2%}")
        print(f"Top-3 Accuracy: {top3_acc:.2%}")

    with open(top1_pred_json, 'w') as f:
        json.dump(top1_result, f, indent=4)

    with open(top3_pred_json, 'w') as f:
        json.dump(top3_results, f, indent=4)

    print(f"Top-1 predictions saved to {top1_pred_json}")
    print(f"Top-3 predictions saved to {top3_pred_json}")

    # calculate confusion matrix for those data with gt
    if top1_acc > 0 and top3_acc > 0:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        # gt_labels = [p.split('/')[-2].split('-')[0][:-3] for p in datas]
        gt_labels = [p.split('/')[-3] for p in datas]
        cm = confusion_matrix(gt_labels, top1_result, labels=classifier.class_names)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classifier.class_names, yticklabels=classifier.class_names, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('./task2_confusion_matrix.png')
    
    for result in results:
        if 'error' not in result:
            print(f"{result['path']}: {result['predicted_name']} ({result['confidence']:.2%})")
        else:
            print(f"{result['path']}: ERROR - {result['error']}")