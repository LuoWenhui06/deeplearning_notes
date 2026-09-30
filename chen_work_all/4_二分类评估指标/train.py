import os
import sys
import random
import copy
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             average_precision_score, f1_score, recall_score, 
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置与种子固定
# ==========================================
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE_ES = 20
PATIENCE_LR = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path.home() / "Desktop" / "CIFAR10_Binary_Experiment_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]
TARGET_CLASSES = {5: 1, 3: 0, 2: 0, 4: 0, 7: 0}

# ==========================================
# 2. 全局数据集类（⚠️ 必须在顶层定义，支持Windows序列化）
# ==========================================
class BinaryDataset(Dataset):
    """二分类包装数据集 - 全局类"""
    def __init__(self, subset, labels, transform):
        self.subset = subset
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        return self.transform(img), self.labels[idx]


class BalancedSubset(Dataset):
    """平衡采样子集 - 全局类"""
    def __init__(self, parent_dataset, indices):
        self.parent = parent_dataset
        self.indices = list(indices)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]


# ==========================================
# 3. 数据集构建
# ==========================================
def get_binary_datasets(data_root):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    full_train = datasets.CIFAR10(root=data_root, train=True, download=True)
    full_test = datasets.CIFAR10(root=data_root, train=False, download=True)
    
    def filter_and_map(dataset, transform):
        indices, labels = [], []
        for i, (_, label) in enumerate(dataset):
            if label in TARGET_CLASSES:
                indices.append(i)
                labels.append(TARGET_CLASSES[label])
        subset = Subset(dataset, indices)
        return BinaryDataset(subset, labels, transform)
    
    train_ds = filter_and_map(full_train, transform_train)
    test_ds = filter_and_map(full_test, transform_test)
    
    print(f"[Data] Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    pos_count = sum(1 for l in train_ds.labels if l == 1)
    print(f"[Data] Train Pos(Dog): {pos_count}, Neg: {len(train_ds)-pos_count}")
    return train_ds, test_ds


def setup_balanced(dataset, seed=42):
    """从负类中分层抽样与正类等量的样本，组成1:1平衡数据集"""
    pos_indices = [i for i, l in enumerate(dataset.labels) if l == 1]
    neg_indices = [i for i, l in enumerate(dataset.labels) if l == 0]
    rng = np.random.RandomState(seed)
    sampled_neg = rng.choice(neg_indices, size=len(pos_indices), replace=False).tolist()
    balanced_indices = pos_indices + sampled_neg
    return BalancedSubset(dataset, balanced_indices)


# ==========================================
# 4. CBAM + 轻量级CNN模型
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return attn * x

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class LightCNN_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            CBAMBlock(256)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            CBAMBlock(256)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            CBAMBlock(256)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ==========================================
# 5. 训练与评估引擎
# ==========================================
class ExperimentRunner:
    def __init__(self, name, train_dataset, test_dataset, config):
        self.name = name
        self.config = config
        self.test_dataset = test_dataset
        
        n_val = int(len(train_dataset) * 0.2)
        n_train = len(train_dataset) - n_val
        set_seed(SEED)
        self.train_subset, self.val_subset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED)
        )
        
        sampler = None
        shuffle = True
        if config.get('weighted_sampler'):
            val_labels = [train_dataset.labels[i] for i in self.train_subset.indices]
            class_counts = np.bincount(val_labels)
            weights = 1.0 / class_counts
            sample_weights = [weights[l] for l in val_labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        
        # ⚠️ num_workers=0 彻底避免Windows多进程序列化问题，CIFAR-10小数据集单进程更快
        self.train_loader = DataLoader(self.train_subset, batch_size=BATCH_SIZE, 
                                       sampler=sampler, shuffle=shuffle, num_workers=0)
        self.val_loader = DataLoader(self.val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        set_seed(SEED)
        self.model = LightCNN_CBAM().to(DEVICE)
        pos_weight = torch.tensor([config.get('pos_weight', 1.0)]).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=PATIENCE_LR
        )
        
        self.best_f1 = 0.0
        self.best_model_state = None
        self.optimal_threshold = 0.5
        self.test_results = {}
        
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_probs, all_labels = [], []
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = self.model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
        return np.array(all_probs), np.array(all_labels)
    
    def find_optimal_threshold(self, probs, labels):
        thresholds = np.arange(0.1, 0.91, 0.01)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t, best_f1
    
    def run(self):
        print(f"\n{'='*50}")
        print(f"🚀 Running Experiment: {self.name}")
        print(f"   Config: {self.config}")
        print(f"{'='*50}")
        
        no_improve = 0
        for epoch in range(MAX_EPOCHS):
            train_loss = self.train_one_epoch()
            val_probs, val_labels = self.evaluate(self.val_loader)
            opt_th, val_f1 = self.find_optimal_threshold(val_probs, val_labels)
            self.scheduler.step(val_f1)
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.optimal_threshold = opt_th
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                
            current_lr = self.optimizer.param_groups[0]['lr']
            if (epoch+1) % 5 == 0 or no_improve == 0:
                print(f"  Epoch [{epoch+1:3d}/{MAX_EPOCHS}] Loss:{train_loss:.4f} | "
                      f"Val F1:{val_f1:.4f}(th={opt_th:.2f}) | Best F1:{self.best_f1:.4f} | LR:{current_lr:.6f}")
            
            if no_improve >= PATIENCE_ES:
                print(f"  ⏹ Early Stopping at epoch {epoch+1}")
                break
        
        self.model.load_state_dict(self.best_model_state)
        test_probs, test_labels = self.evaluate(self.test_loader)
        test_preds = (test_probs >= self.optimal_threshold).astype(int)
        
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(test_labels, test_probs)
        aupr = average_precision_score(test_labels, test_probs)
        test_f1 = f1_score(test_labels, test_preds)
        test_recall = recall_score(test_labels, test_preds)
        cm = confusion_matrix(test_labels, test_preds)
        
        self.test_results = {
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
            'precision': precision, 'recall_curve': recall, 'aupr': aupr,
            'f1': test_f1, 'recall': test_recall, 'cm': cm,
            'probs': test_probs, 'labels': test_labels,
            'threshold': self.optimal_threshold
        }
        
        print(f"\n✅ [{self.name}] Test Results:")
        print(f"   AUC={roc_auc:.4f} | AUPR={aupr:.4f} | F1={test_f1:.4f} | Recall={test_recall:.4f}")
        print(f"   Optimal Threshold={self.optimal_threshold:.2f}")
        return self.test_results


# ==========================================
# 6. 可视化模块
# ==========================================
def plot_all_results(results_dict):
    plt.rcParams['font.size'] = 12
    colors = {'A_Balanced': '#2196F3', 'B_Imbalanced': '#F44336', 'C_Fixed': '#4CAF50'}
    titles = {'A_Balanced': 'A组 (1:1 平衡)', 'B_Imbalanced': 'B组 (1:4 不平衡)', 'C_Fixed': 'C组 (1:4 加权+重采样)'}
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 7))
    
    for key, res in results_dict.items():
        c = colors[key]
        t = titles[key]
        ax_roc.plot(res['fpr'], res['tpr'], color=c, lw=2, 
                    label=f"{t} (AUC={res['roc_auc']:.4f})")
        ax_pr.plot(res['recall_curve'], res['precision'], color=c, lw=2,
                   label=f"{t} (AUPR={res['aupr']:.4f})")
    
    ax_roc.plot([0,1],[0,1],'k--',lw=1)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(OUTPUT_DIR / "ROC_Curves_Comparison.png", dpi=150)
    
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves Comparison')
    ax_pr.legend(loc='lower left')
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_ylim([0, 1.05])
    fig_pr.tight_layout()
    fig_pr.savefig(OUTPUT_DIR / "PR_Curves_Comparison.png", dpi=150)
    
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, res) in zip(axes_cm, results_dict.items()):
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-Dog', 'Dog'], yticklabels=['Non-Dog', 'Dog'])
        ax.set_title(f'{titles[key]}\nF1={res["f1"]:.4f} Recall={res["recall"]:.4f}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    fig_cm.suptitle('Confusion Matrices', fontsize=14, y=1.02)
    fig_cm.tight_layout()
    fig_cm.savefig(OUTPUT_DIR / "Confusion_Matrices.png", dpi=150, bbox_inches='tight')
    
    fig_hist, axes_hist = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, res) in zip(axes_hist, results_dict.items()):
        neg_probs = res['probs'][res['labels'] == 0]
        pos_probs = res['probs'][res['labels'] == 1]
        ax.hist(neg_probs, bins=50, alpha=0.6, color='#90CAF9', label='Non-Dog', density=True)
        ax.hist(pos_probs, bins=50, alpha=0.7, color='#EF5350', label='Dog', density=True)
        ax.axvline(x=res['threshold'], color='black', linestyle='--', lw=2, label=f'Threshold={res["threshold"]:.2f}')
        ax.set_title(titles[key])
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig_hist.suptitle('Prediction Probability Distribution', fontsize=14, y=1.02)
    fig_hist.tight_layout()
    fig_hist.savefig(OUTPUT_DIR / "Probability_Distribution.png", dpi=150, bbox_inches='tight')
    
    plt.close('all')
    print(f"\n📊 All plots saved to: {OUTPUT_DIR}")


# ==========================================
# 7. 主程序入口
# ==========================================
if __name__ == "__main__":
    DATA_ROOT = r"C:\Users\16131\Desktop\陈学姐作业\2_CNN\data"
    
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Data path not found: {DATA_ROOT}")
        sys.exit(1)
    
    train_ds, test_ds = get_binary_datasets(DATA_ROOT)
    
    experiments = {
        'A_Balanced': {
            'desc': '平衡基线 (1:1)',
            'setup_fn': lambda ds: setup_balanced(ds, SEED),
            'config': {'pos_weight': 1.0, 'weighted_sampler': False}
        },
        'B_Imbalanced': {
            'desc': '不平衡无干预 (1:4)',
            'setup_fn': lambda ds: ds,
            'config': {'pos_weight': 1.0, 'weighted_sampler': False}
        },
        'C_Fixed': {
            'desc': '不平衡加权+重采样 (1:4)',
            'setup_fn': lambda ds: ds,
            'config': {'pos_weight': 4.0, 'weighted_sampler': True}
        }
    }
    
    all_results = {}
    for exp_key, exp_info in experiments.items():
        exp_train_ds = exp_info['setup_fn'](train_ds)
        runner = ExperimentRunner(exp_key, exp_train_ds, test_ds, exp_info['config'])
        results = runner.run()
        all_results[exp_key] = results
    
    print("\n" + "="*70)
    print("📋 FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Experiment':<20} {'AUC':>8} {'AUPR':>8} {'F1':>8} {'Recall':>8} {'Threshold':>10}")
    print("-"*70)
    for key, res in all_results.items():
        print(f"{key:<20} {res['roc_auc']:>8.4f} {res['aupr']:>8.4f} "
              f"{res['f1']:>8.4f} {res['recall']:>8.4f} {res['threshold']:>10.2f}")
    print("="*70)
    
    import json
    summary = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv.tolist() if isinstance(vv, np.ndarray) else vv 
                   for kk, vv in v.items() if kk not in ['fpr','tpr','precision','recall_curve','probs','labels']} 
               for k, v in all_results.items()}
    with open(OUTPUT_DIR / "results_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    plot_all_results(all_results)
    print(f"\n🎉 All experiments completed! Results saved to:\n   {OUTPUT_DIR}")