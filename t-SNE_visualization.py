"""
使用训练好的模型进行 t-SNE 可视化
不需要重新训练
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from task2_model import MusicCNN
from dataloader import create_dataloader

class FeatureExtractor:
    """
    从训练好的模型中提取特征用于 t-SNE 可视化
    """
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 载入训练好的模型
        self.model = MusicCNN(n_classes=20, n_mels=128).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 类别名称
        self.class_names = [
            'aerosmith', 'beatles', 'creedence_clearwater_revival', 'cure',
            'dave_matthews_band', 'depeche_mode', 'fleetwood_mac', 'garth_brooks',
            'green_day', 'led_zeppelin', 'madonna', 'metallica', 'prince',
            'queen', 'radiohead', 'roxette', 'steely_dan', 'suzanne_vega',
            'tori_amos', 'u2'
        ]
        
        # 注册 hook 来提取特征
        self.features = None
        self._register_hook()
    
    def _register_hook(self):
        """
        注册 hook 在最后的全连接层之前提取特征
        """
        def hook_fn(module, input, output):
            # 提取 dropout 之前的特征（256 维）
            self.features = input[0].detach()
        
        # 在 fc 层注册 hook
        self.model.fc2.register_forward_hook(hook_fn)
    
    def extract_features(self, dataloader, max_samples=None):
        """
        从 dataloader 中提取特征
        
        Args:
            dataloader: 数据加载器
            max_samples: 最多提取多少个样本（None=全部）
        
        Returns:
            features: (N, 256) 特征矩阵
            labels: (N,) 标签数组
            predictions: (N,) 预测数组
        """
        all_features = []
        all_labels = []
        all_predictions = []
        
        sample_count = 0
        
        with torch.no_grad():
            for mel_spec, _, labels in tqdm(dataloader, desc="Extracting features"):
                mel_spec = mel_spec.to(self.device)
                
                # 确保形状正确
                if mel_spec.shape[1] != 1:
                    mel_spec = mel_spec.transpose(0, 1)
                
                # Forward pass（会触发 hook）
                outputs = self.model(mel_spec)
                predictions = torch.argmax(outputs, dim=1)
                
                # 收集特征和标签
                all_features.append(self.features.cpu().numpy())
                all_labels.append(labels.numpy())
                all_predictions.append(predictions.cpu().numpy())
                
                sample_count += len(labels)
                
                if max_samples is not None and sample_count >= max_samples:
                    break
        
        # 合并所有 batch
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)
        
        if max_samples is not None:
            features = features[:max_samples]
            labels = labels[:max_samples]
            predictions = predictions[:max_samples]
        
        return features, labels, predictions
    
    def plot_tsne(self, features, labels, predictions=None, 
                  perplexity=30, save_path='tsne_visualization.png'):
        """
        使用 t-SNE 降维并可视化
        
        Args:
            features: (N, 256) 特征矩阵
            labels: (N,) 真实标签
            predictions: (N,) 预测标签（可选）
            perplexity: t-SNE 的 perplexity 参数
            save_path: 保存路径
        """
        print(f"\nRunning t-SNE (perplexity={perplexity})...")
        print(f"Features shape: {features.shape}")
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        print("t-SNE completed!")
        
        # 创建两个子图：一个按真实标签，一个按预测标签
        if predictions is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # 配色方案
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # 子图 1：真实标签
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[i]], label=class_name, 
                       alpha=0.6, s=30, edgecolors='none')
        
        ax1.set_title('t-SNE Visualization (True Labels)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(alpha=0.3)
        
        # 子图 2：预测标签（如果提供）
        if predictions is not None:
            for i, class_name in enumerate(self.class_names):
                mask = predictions == i
                ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[colors[i]], label=class_name,
                           alpha=0.6, s=30, edgecolors='none')
            
            # 标记错误预测
            errors = labels != predictions
            if errors.sum() > 0:
                ax2.scatter(features_2d[errors, 0], features_2d[errors, 1],
                           facecolors='none', edgecolors='red', s=50, 
                           linewidths=1.5, label='Misclassified')
            
            ax2.set_title('t-SNE Visualization (Predictions)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('t-SNE Dimension 1')
            ax2.set_ylabel('t-SNE Dimension 2')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_confusion_tsne(self, features, labels, predictions,
                           save_path='tsne_confusion.png'):
        """
        t-SNE 可视化，突出显示混淆的类别对
        """
        print("\nCreating confusion-based t-SNE...")
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # 绘制所有点（半透明）
        for i in range(20):
            mask = labels == i
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[colors[i]], alpha=0.3, s=20, edgecolors='none')
        
        # 高亮显示错误预测
        errors = labels != predictions
        if errors.sum() > 0:
            ax.scatter(features_2d[errors, 0], features_2d[errors, 1],
                      facecolors='none', edgecolors='red', 
                      s=100, linewidths=2, alpha=0.8,
                      label=f'Misclassified ({errors.sum()})')
        
        ax.set_title('t-SNE: Highlighting Misclassifications', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()


# ========== 使用示例 ==========
if __name__ == "__main__":
    import os
    
    # 配置
    MODEL_PATH = './models/task2/vocal_only.pth'
    OUTPUT_DIR = './tsne_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化特征提取器
    print("Loading model...")
    extractor = FeatureExtractor(
        model_path=MODEL_PATH,
        device='cuda'
    )
    
    # 创建 dataloader
    print("\nCreating dataloader...")
    
    # 可以用 vocal 或 full mix 数据
    val_loader = create_dataloader(
        json_list='./dataset/val_vocal.json',  # 或 './dataset/artist20/val.json'
        batch_size=32,
        num_workers=4,
        sr=16000,
        n_mels=128,
        hop_length=512,
        chunk_duration=30.0,
        overlap=0.0,
        mode='val'
    )
    
    # 提取特征
    print("\nExtracting features...")
    features, labels, predictions = extractor.extract_features(
        val_loader,
        max_samples=2000  # 限制样本数（t-SNE 很慢）
    )
    
    # 计算准确率
    accuracy = (labels == predictions).mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # 生成 t-SNE 可视化
    print("\nGenerating visualizations...")
    
    # 1. 基本 t-SNE（真实标签 vs 预测标签）
    extractor.plot_tsne(
        features, labels, predictions,
        perplexity=30,
        save_path=f'{OUTPUT_DIR}/tsne_basic.png'
    )
    
    # 2. 突出混淆的可视化
    extractor.plot_confusion_tsne(
        features, labels, predictions,
        save_path=f'{OUTPUT_DIR}/tsne_confusion.png'
    )
    
    # 3. 不同 perplexity 值的比较
    for perp in [10, 30, 50]:
        extractor.plot_tsne(
            features, labels, predictions,
            perplexity=perp,
            save_path=f'{OUTPUT_DIR}/tsne_perp{perp}.png'
        )
    
    print("\nDone! Check the results in:", OUTPUT_DIR)