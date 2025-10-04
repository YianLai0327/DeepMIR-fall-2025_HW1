import json
import os
from pathlib import Path
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import argparse


def load_audio_paths(json_path):
    """從 JSON 文件讀取音頻路徑列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('audio_paths', [])


def separate_audio(audio_path, model, output_dir, label, device='cuda'):
    """
    使用 Demucs 分離單個音頻文件
    
    Args:
        audio_path: 音頻文件路徑
        model: Demucs 模型
        output_dir: 輸出目錄
        device: 運算設備 ('cuda' 或 'cpu')
    """
    print(f"處理: {audio_path}")
    
    # 載入音頻 (16kHz, mono)
    waveform, sr = torchaudio.load(audio_path)
    
    # 確保是單聲道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Demucs 通常使用 44.1kHz，如果需要可以重採樣
    # 但也可以直接使用 16kHz，模型會自動處理
    if sr != model.samplerate:
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        waveform = resampler(waveform)
        sr = model.samplerate
    
    # 轉換為立體聲 (Demucs 期望立體聲輸入)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    
    # 移到指定設備
    waveform = waveform.to(device)
    
    # 添加 batch 維度
    waveform = waveform.unsqueeze(0)
    
    # 執行音源分離
    with torch.no_grad():
        sources = apply_model(model, waveform, device=device)
    
    # sources shape: (batch, stems, channels, time)
    sources = sources[0]  # 移除 batch 維度
    
    # 創建輸出目錄
    audio_name = Path(audio_path).stem
    # add label to path
    audio_name = f"{label}_{audio_name}"
    audio_output_dir = Path(output_dir) / audio_name
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存分離後的音源
    # Demucs 的源通常包括: drums, bass, other, vocals
    stem_names = model.sources
    
    for i, stem_name in enumerate(stem_names):
        stem_audio = sources[i]  # shape: (channels, time)
        
        # 轉回單聲道
        if stem_audio.shape[0] > 1:
            stem_audio = stem_audio.mean(dim=0, keepdim=True)
        
        # 如果原始音頻是 16kHz，重採樣回 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            stem_audio = resampler(stem_audio.cpu())
        else:
            stem_audio = stem_audio.cpu()
        
        # 保存
        output_path = audio_output_dir / f"{stem_name}.wav"
        torchaudio.save(str(output_path), stem_audio, 16000)
        print(f"  已保存: {output_path}")
    
    print(f"完成: {audio_path}\n")


def main():
    parser = argparse.ArgumentParser(description='使用 Demucs 進行音源分離')
    parser.add_argument('--json_path', type=str, default='dataset/artist20/val.json',
                        help='包含音頻路徑列表的 JSON 文件')
    parser.add_argument('--output_dir', type=str, default='dataset/val_seperation_results',
                        help='輸出目錄路徑')
    parser.add_argument('--model', type=str, default='htdemucs',
                        help='Demucs 模型名稱 (htdemucs, htdemucs_ft, mdx_extra 等)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='運算設備 (cuda 或 cpu)')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 檢查設備
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，切換到 CPU")
        device = 'cpu'
    
    # 載入 Demucs 模型
    print(f"載入 Demucs 模型: {args.model}")
    model = get_model(args.model)
    model.to(device)
    model.eval()
    print(f"模型已載入到 {device}\n")
    
    # 讀取音頻路徑列表
    audio_paths = load_audio_paths(args.json_path)
    print(f"共有 {len(audio_paths)} 個音頻文件待處理\n")
    
    # 處理每個音頻文件
    import tqdm
    pbar = tqdm.tqdm(audio_paths, desc="處理音頻文件")
    for audio_path in pbar:
        audio_path = audio_path.replace('./', './dataset/artist20/')
        label = audio_path.split('/')[-3]
        if not os.path.exists(audio_path):
            print(f"警告: 文件不存在: {audio_path}")
            continue
        
        try:
            separate_audio(audio_path, model, args.output_dir, label, device)
        except Exception as e:
            print(f"錯誤: 處理 {audio_path} 時發生錯誤: {str(e)}\n")
            continue
    
    print("全部處理完成！")


if __name__ == "__main__":
    main()