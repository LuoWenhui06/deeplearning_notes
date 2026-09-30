import os
import sys
import time
import random
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

# ======================== 全局配置 ========================
class Config:
    # 路径配置
    DATA_DIR = r"C:\Users\16131\Desktop\陈学姐作业\2_CNN\data"
    SAVE_DIR = r"C:\Users\16131\Desktop"
    
    # 数据集配置
    POS_CLASS = 5  # Dog
    NEG_CLASSES = [2, 3, 4, 7]  # Bird, Cat, Deer, Horse
    
    # 训练超参数 (加速优化版)
    BATCH_SIZE = 256
    LR = 4e-3
    WEIGHT_DECAY = 1e-4
    POS_WEIGHT = 4.0
    MAX_EPOCHS = 30
    MIN_EPOCHS = 5
    EARLY_STOP_PATIENCE = 10
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 2
    
    # 随机种子
    CV_SEEDS = [42, 43]
    SINGLE_SEEDS = list(range(10))
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_DIR, "figures"), exist_ok=True)
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False


# ======================== 数据集构建 ========================
def build_datasets(data_dir):
    """构建不平衡二分类数据集 (Dog vs Non-Dog)"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
    full_test = datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    def filter_indices(dataset, pos_class, neg_classes):
        targets = np.array(dataset.targets)
        mask = np.isin(targets, [pos_class] + neg_classes)
        indices = np.where(mask)[0]
        labels = (targets[indices] == pos_class).astype(np.float32)
        return indices, labels
    
    pool_idx, pool_labels = filter_indices(full_train, Config.POS_CLASS, Config.NEG_CLASSES)
    test_idx, test_labels = filter_indices(full_test, Config.POS_CLASS, Config.NEG_CLASSES)
    
    print(f"[DATA] Pool: {len(pool_idx)} samples (Pos: {int(pool_labels.sum())}, Neg: {len(pool_labels)-int(pool_labels.sum())})")
    print(f"[DATA] Test: {len(test_idx)} samples (Pos: {int(test_labels.sum())}, Neg: {len(test_labels)-int(test_labels.sum())})")
    
    return full_train, pool_idx, pool_labels, full_test, test_idx, test_labels, transform_train, transform_test


class BinaryCIFAR(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, labels, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[self.indices[idx]]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ======================== 模型定义 ========================
class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        x = self.head(x)
        return x.squeeze(-1)


# ======================== 工具函数 ========================
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def warmup_lr(optimizer, epoch, step, total_steps, warmup_epochs, base_lr):
    """Linear Warmup"""
    if epoch < warmup_epochs:
        progress = (epoch * total_steps + step) / (warmup_epochs * total_steps)
        lr = base_lr * progress
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr
    return None

def find_optimal_threshold(y_true, y_prob):
    """在验证集上搜索使F1最大化的阈值"""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def evaluate(model, loader, device, threshold=0.5):
    """评估模型，返回所有指标"""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([Config.POS_WEIGHT]).to(device))
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(enabled=device.type == 'cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(all_labels)
    
    auc = roc_auc_score(all_labels, all_probs)
    aupr = average_precision_score(all_labels, all_probs)
    preds = (all_probs >= threshold).astype(int)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    return {
        'loss': avg_loss, 'auc': auc, 'aupr': aupr,
        'precision': prec, 'recall': rec, 'f1': f1,
        'probs': all_probs, 'labels': all_labels
    }


# ======================== 单次训练流程 ========================
def run_single_experiment(run_id, total_runs, train_dataset, val_dataset, 
                          test_dataset, history_store):
    """执行一次完整的 训练→早停→阈值搜索→测试 流程"""
    device = Config.DEVICE
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
                              persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
                            persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
                             persistent_workers=True, prefetch_factor=2)
    
    model = StandardCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([Config.POS_WEIGHT]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=Config.LR_PATIENCE,
        factor=Config.LR_FACTOR, min_lr=Config.MIN_LR
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    best_aupr = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    total_batches = len(train_loader)
    
    epoch_history = []  # 记录当前run的epoch级历史
    
    for epoch in range(Config.MAX_EPOCHS):
        # === Training ===
        model.train()
        train_loss_sum = 0.0
        pbar = enumerate(train_loader)
        
        for step, (imgs, labels) in pbar:
            # Warmup
            warmup_lr(optimizer, epoch, step, total_batches, Config.WARMUP_EPOCHS, Config.LR)
            
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item() * imgs.size(0)
            
            # Batch级日志
            if (step + 1) % 20 == 0 or (step + 1) == total_batches:
                print(f"[Run:{run_id+1:02d}/{total_runs}] Epoch:{epoch+1:02d}/{Config.MAX_EPOCHS} | "
                      f"Batch:{step+1:03d}/{total_batches} | LR:{get_lr(optimizer):.2e} | "
                      f"AMP:ON | Batch-Loss:{loss.item():.4f}")
        
        avg_train_loss = train_loss_sum / len(train_dataset)
        
        # === Validation ===
        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        current_lr = get_lr(optimizer)
        
        # Epoch级日志
        print(f"[Run:{run_id+1:02d}/{total_runs}] Epoch:{epoch+1:02d}/{Config.MAX_EPOCHS} | "
              f"Avg-Train-Loss:{avg_train_loss:.4f} | Val-Loss:{val_metrics['loss']:.4f} | "
              f"Val-AUC:{val_metrics['auc']:.4f} | Val-AUPR:{val_metrics['aupr']:.4f} | "
              f"Val-F1@0.50:{val_metrics['f1']:.4f} | LR:{current_lr:.2e} | "
              f"Best-AUPR:{best_aupr:.4f}(Ep:{best_epoch:02d}) | "
              f"Patience:{patience_counter}/{Config.EARLY_STOP_PATIENCE}")
        
        # 记录历史
        epoch_history.append({
            'epoch': epoch + 1, 'train_loss': avg_train_loss,
            'val_loss': val_metrics['loss'], 'val_aupr': val_metrics['aupr']
        })
        
        # 模型选择 & 早停
        if val_metrics['aupr'] > best_aupr:
            best_aupr = val_metrics['aupr']
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # LR调度 (Warmup结束后才生效)
        if epoch >= Config.WARMUP_EPOCHS:
            scheduler.step(val_metrics['aupr'])
        
        # 早停判断
        if epoch + 1 >= Config.MIN_EPOCHS and patience_counter >= Config.EARLY_STOP_PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch+1}")
            break
    
    # === 加载最佳模型 & 阈值搜索 ===
    if best_state is not None:
        model.load_state_dict(best_state)
    
    val_result = evaluate(model, val_loader, device, threshold=0.5)
    opt_thresh, val_f1_opt = find_optimal_threshold(val_result['labels'], val_result['probs'])
    
    print(f"[Run:{run_id+1:02d}/{total_runs}] THRESHOLD OPT | Val-F1:{val_f1_opt:.4f} @ T={opt_thresh:.2f}")
    
    # === 测试集评估 ===
    test_result = evaluate(model, test_loader, device, threshold=opt_thresh)
    
    print(f"[Run:{run_id+1:02d}/{total_runs}] TEST RESULT | Test-Loss:{test_result['loss']:.4f} | "
          f"AUC:{test_result['auc']:.4f} | AUPR:{test_result['aupr']:.4f} | "
          f"Prec:{test_result['precision']:.4f} | Rec:{test_result['recall']:.4f} | "
          f"F1@{opt_thresh:.2f}:{test_result['f1']:.4f} | Opt-Thresh:{opt_thresh:.2f} | "
          f"Samples:{len(test_dataset)}")
    print("-" * 90)
    
    # 存储结果
    result_row = {
        'run_id': run_id + 1, 'protocol': '',  # 由调用者填充
        'best_epoch': best_epoch, 'opt_threshold': opt_thresh,
        'val_f1_opt': val_f1_opt,
        'test_loss': test_result['loss'], 'test_auc': test_result['auc'],
        'test_aupr': test_result['aupr'], 'test_precision': test_result['precision'],
        'test_recall': test_result['recall'], 'test_f1': test_result['f1']
    }
    
    history_store.extend([{**h, 'run_id': run_id + 1} for h in epoch_history])
    
    return result_row


# ======================== 主实验流程 ========================
def main():
    Config.setup()
    print(f"[INFO] Device: {Config.DEVICE}")
    print(f"[INFO] AMP: {'Enabled' if Config.DEVICE.type == 'cuda' else 'Disabled'}")
    print("=" * 90)
    
    # 构建数据集
    full_train, pool_idx, pool_labels, full_test, test_idx, test_labels, \
        transform_train, transform_test = build_datasets(Config.DATA_DIR)
    
    # 固定测试集
    test_dataset = BinaryCIFAR(full_test, test_idx, test_labels, transform_test)
    
    all_results = []
    all_histories = []
    run_counter = 0
    total_runs = len(Config.CV_SEEDS) * 5 + len(Config.SINGLE_SEEDS)  # 10 + 10 = 20
    
    # ===== Protocol 1: Repeated Stratified 5-Fold CV =====
    print("\n" + "=" * 90)
    print("PROTOCOL 1: Repeated Stratified 5-Fold CV")
    print("=" * 90)
    
    for seed in Config.CV_SEEDS: # 遍历交叉验证的随机种子列表（如 [42, 43]），实现重复分层K折交叉验证
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    # 初始化重复分层K折分割器；n_splits=5: 将数据分为5折；n_repeats=1: 每个种子仅执行1轮完整的5折划分（通过外层seed循环实现"重复"）；random_state=seed: 固定当前种子的随机性，确保该种子下的5折划分可复现
        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(pool_idx, pool_labels)): # 对池数据(pool_idx)按标签(pool_labels)进行分层划分，保证每折正负样本比例一致；enumerate 同时获取折编号(fold_idx)和对应的训练/验证集索引
            train_ds = BinaryCIFAR(full_train, pool_idx[train_idx], pool_labels[train_idx], transform_train)# 根据当前折的索引构建训练集数据集对象，应用训练集数据增强(transform_train)
            val_ds = BinaryCIFAR(full_train, pool_idx[val_idx], pool_labels[val_idx], transform_test) # 根据当前折的索引构建验证集数据集对象，应用测试集变换(transform_test，无增强)
            
            row = run_single_experiment(run_counter, total_runs, train_ds, val_ds, test_dataset, all_histories)# 执行单次完整实验流程（训练→早停→阈值搜索→测试集评估），返回指标字典
            row['protocol'] = 'CV'# 标记评估协议类型为交叉验证
            row['seed'] = seed# 记录当前使用的随机种子
            row['fold'] = fold_idx + 1 # 记录当前折号（从1开始计数，符合人类阅读习惯）
        
            all_results.append(row) # 将本次运行结果追加到总结果列表中
            run_counter += 1 # 全局运行计数器自增，用于日志打印和进度追踪


    # ===== Protocol 2: Single Stratified Random Split =====
    print("\n" + "=" * 90)
    print("PROTOCOL 2: Single Stratified Random Split")
    print("=" * 90)
    
    for seed in Config.SINGLE_SEEDS:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(sss.split(pool_idx, pool_labels))
        
        train_ds = BinaryCIFAR(full_train, pool_idx[train_idx], pool_labels[train_idx], transform_train)
        val_ds = BinaryCIFAR(full_train, pool_idx[val_idx], pool_labels[val_idx], transform_test)
        
        row = run_single_experiment(run_counter, total_runs, train_ds, val_ds, test_dataset, all_histories)
        row['protocol'] = 'Single'
        row['seed'] = seed
        row['fold'] = 0
        all_results.append(row)
        run_counter += 1
    
    # ===== 保存结果 =====
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(Config.SAVE_DIR, "experiment_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    df_history = pd.DataFrame(all_histories)
    history_path = os.path.join(Config.SAVE_DIR, "training_history.csv")
    df_history.to_csv(history_path, index=False)
    print(f"✅ History saved to: {history_path}")
    
    # ===== 统计分析 & 可视化 =====
    generate_statistics(df_results)
    generate_visualizations(df_results, df_history)
    
    print("\n🎉 All experiments completed!")


# ======================== 统计分析 ========================
def generate_statistics(df):
    print("\n" + "=" * 90)
    print("STATISTICAL SUMMARY")
    print("=" * 90)
    
    metrics = ['test_auc', 'test_aupr', 'test_f1', 'test_precision', 'test_recall', 'opt_threshold']
    
    for protocol in ['CV', 'Single']:
        subset = df[df['protocol'] == protocol]
        print(f"\n--- {protocol} (n={len(subset)}) ---")
        for m in metrics:
            mu = subset[m].mean()
            sigma = subset[m].std()
            print(f"  {m:>15s}: {mu:.4f} ± {sigma:.4f}")
    
    # 泛化偏差
    benchmark = df[metrics].mean()
    print(f"\n--- Generalization Bias (vs 20-run benchmark) ---")
    for protocol in ['CV', 'Single']:
        subset = df[df['protocol'] == protocol]
        proto_mean = subset[metrics].mean()
        bias = proto_mean - benchmark
        print(f"\n  {protocol}:")
        for m in metrics:
            print(f"    {m:>15s}: {bias[m]:+.4f}")


# ======================== 可视化 ========================
def generate_visualizations(df, df_history):
    fig_dir = os.path.join(Config.SAVE_DIR, "figures")
    metrics = ['test_auc', 'test_aupr', 'test_f1', 'test_precision', 'test_recall']
    metric_labels = ['AUC', 'AUPR', 'F1', 'Precision', 'Recall']
    
    # 10.1 箱线图 + 小提琴图
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    for ax, m, label in zip(axes, metrics, metric_labels):
        sns.violinplot(data=df, x='protocol', y=m, ax=ax, inner=None, alpha=0.3)
        sns.boxplot(data=df, x='protocol', y=m, ax=ax, width=0.3)
        sns.stripplot(data=df, x='protocol', y=m, ax=ax, color='black', size=3)
        ax.set_title(label, fontsize=14)
    plt.suptitle('Protocol Comparison: Box + Violin Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "stability_box_violin.png"), dpi=150)
    plt.close()
    print(f"✅ Saved: stability_box_violin.png")
    
    # 10.2 泛化偏差雷达图
    benchmark = df[metrics].mean()
    biases = {}
    for protocol in ['CV', 'Single']:
        proto_mean = df[df['protocol'] == protocol][metrics].mean()
        biases[protocol] = (proto_mean - benchmark).values
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#2196F3', '#FF5722']
    for (name, vals), color in zip(biases.items(), colors):
        vals_plot = vals.tolist() + [vals[0]]
        ax.plot(angles, vals_plot, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, vals_plot, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_title('Generalization Bias Radar Chart', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "bias_radar.png"), dpi=150)
    plt.close()
    print(f"✅ Saved: bias_radar.png")
    
    # 10.3 阈值抖动图
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.stripplot(data=df, x='protocol', y='opt_threshold', jitter=True, size=8, alpha=0.7, ax=ax)
    sns.boxplot(data=df, x='protocol', y='opt_threshold', width=0.3, ax=ax)
    ax.set_title('Optimal Threshold Jitter Plot', fontsize=14)
    ax.set_ylabel('Optimal Threshold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "threshold_jitter.png"), dpi=150)
    plt.close()
    print(f"✅ Saved: threshold_jitter.png")
    
    # 10.4 Bland-Altman 图
    # 需要配对数据：这里用同一run的val_f1_opt和test_f1
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_f1 = (df['val_f1_opt'] + df['test_f1']) / 2
    diff_f1 = df['val_f1_opt'] - df['test_f1']
    
    colors_map = {'CV': '#2196F3', 'Single': '#FF5722'}
    for protocol in ['CV', 'Single']:
        mask = df['protocol'] == protocol
        ax.scatter(mean_f1[mask], diff_f1[mask], label=protocol, 
                   color=colors_map[protocol], alpha=0.7, edgecolors='white')
    
    mean_diff = diff_f1.mean()
    std_diff = diff_f1.std()
    ax.axhline(y=mean_diff, color='green', linestyle='--', label=f'Mean: {mean_diff:.4f}')
    ax.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle=':', label=f'+1.96σ: {mean_diff+1.96*std_diff:.4f}')
    ax.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle=':', label=f'-1.96σ: {mean_diff-1.96*std_diff:.4f}')
    ax.set_xlabel('Mean of Val-F1 and Test-F1', fontsize=12)
    ax.set_ylabel('Val-F1 − Test-F1', fontsize=12)
    ax.set_title('Bland-Altman: Val-F1 vs Test-F1 Consistency', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "bland_altman.png"), dpi=150)
    plt.close()
    print(f"✅ Saved: bland_altman.png")
    
    # 10.5 训练曲线图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for protocol in ['CV', 'Single']:
        subset = df_history[df_history['run_id'].isin(
            df[df['protocol'] == protocol]['run_id'].unique()
        )]
        grouped = subset.groupby('epoch').agg({'val_aupr': ['mean', 'std'], 'val_loss': ['mean', 'std']})
        epochs = grouped.index
        
        # Val-AUPR
        mean_aupr = grouped[('val_aupr', 'mean')]
        std_aupr = grouped[('val_aupr', 'std')]
        ax1.plot(epochs, mean_aupr, label=protocol, linewidth=2)
        ax1.fill_between(epochs, mean_aupr - std_aupr, mean_aupr + std_aupr, alpha=0.15)
        
        # Val-Loss
        mean_loss = grouped[('val_loss', 'mean')]
        std_loss = grouped[('val_loss', 'std')]
        ax2.plot(epochs, mean_loss, label=protocol, linewidth=2)
        ax2.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.15)
    
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Val-AUPR')
    ax1.set_title('Training Curve: Val-AUPR', fontsize=14); ax1.legend()
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val-Loss')
    ax2.set_title('Training Curve: Val-Loss', fontsize=14); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"✅ Saved: training_curves.png")


if __name__ == "__main__":
    main()