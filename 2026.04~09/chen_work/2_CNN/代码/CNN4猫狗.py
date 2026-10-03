import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random
import os

# =============================
# 1. 固定随机种子
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================
# 2. 自定义数据集类
# =============================
class MyCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train=train, download=True, transform=transform)
        target_classes = [2, 3, 4, 5, 7]
        mask = np.isin(self.targets, target_classes)
        self.data = self.data[mask]
        old_labels = np.array(self.targets)[mask]
        label_map = {2: 0, 3: 1, 4: 2, 5: 3, 7: 4}
        self.targets = [label_map[x] for x in old_labels]

# =============================
# 3. ★ 自适应 CBAM 注意力机制模块
# =============================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        # ★ 核心改进：根据通道数自适应调整压缩比
        # 防止小通道(如64)被过度压缩导致信息丢失，同时减少冗余参数
        adaptive_ratio = max(4, in_planes // 16)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // adaptive_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // adaptive_ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# =============================
# 4. 优化后的 CNN 模型
# =============================
class CNN_Optimized(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super().__init__()
        c1, c2, c3 = channels
        
        # 特征提取层：第三层去掉 MaxPool，保留 8x8 高分辨率特征图
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, c1, 3, 1, 1), nn.BatchNorm2d(c1), nn.ReLU(True), CBAM(c1), nn.MaxPool2d(2),
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(c1, c2, 3, 1, 1), nn.BatchNorm2d(c2), nn.ReLU(True), CBAM(c2), nn.MaxPool2d(2),
            # Block 3: 8x8 -> 8x8 (无池化)
            nn.Conv2d(c2, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.ReLU(True), CBAM(c3),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(c3 * 8 * 8, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# =============================
# 5. 训练函数
# =============================
def train_model(model, name, train_loader, test_loader, device, epochs=150, lr=0.001, patience=30):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_pred, best_true = None, None
    best_test_acc, no_improve_count, best_epoch = 0.0, 0, 0

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += (out.argmax(1) == label).sum().item()
            total += label.size(0)

        tr_loss = loss_sum / len(train_loader)
        tr_acc = correct / total * 100

        # --- 测试阶段 ---
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                out = model(img)
                loss = criterion(out, label)
                loss_sum += loss.item()
                pred = out.argmax(1)
                correct += (pred == label).sum().item()
                total += label.size(0)
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(label.cpu().numpy())

        te_loss = loss_sum / len(test_loader)
        te_acc = correct / total * 100
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_epoch = epoch + 1
            best_pred, best_true = all_pred, all_true
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{name}] Epoch {epoch+1:3d}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Train Acc:{tr_acc:.2f}% | Test Acc:{te_acc:.2f}% | Best:{best_test_acc:.2f}%")

        if no_improve_count >= patience:
            print(f"⚠️ [{name}] 早停触发! 最佳Epoch:{best_epoch}, 最佳Test Acc:{best_test_acc:.2f}%")
            break

    history["best_pred"] = best_pred
    history["best_true"] = best_true
    return history

# =============================
# 6. 辅助函数：计算参数量
# =============================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================
# ★ 7. 主程序入口 (Windows 多进程保护)
# =============================
if __name__ == '__main__':
    set_seed(42)
    
    BATCH_SIZE = 64
    EPOCHS = 150  # ★ 增加训练轮次，确保小模型充分收敛
    CLASS_NAMES = ["bird", "cat", "deer", "dog", "horse"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 数据增强 (包含 RandomErasing)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = MyCIFAR10("./data", True, train_transform)
    test_dataset = MyCIFAR10("./data", False, test_transform)

    # Windows下自动使用单进程避免报错，Linux/Mac自动开启多进程加速
    num_workers = 0 if os.name == 'nt' else 2 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    print(f"训练集: {len(train_dataset)}张 | 测试集: {len(test_dataset)}张\n")

    # 定义两个对比模型
    configs = {
        "3x256": [256, 256, 256],
        "64-128-256": [64, 128, 256]
    }
    
    results = {}
    models_info = {}

    for name, channels in configs.items():
        print(f"\n{'='*60}")
        print(f"开始训练: {name} (自适应CBAM + 无末层池化 + RandomErasing)")
        print(f"{'='*60}")
        
        model = CNN_Optimized(channels=channels)
        params = count_parameters(model)
        models_info[name] = {"params": params, "channels": channels}
        
        print(f"模型参数量: {params:,}")
        
        hist = train_model(model, name, train_loader, test_loader, DEVICE, epochs=EPOCHS)
        results[name] = hist

    # =============================
    # 8. 结果对比与分析
    # =============================
    print("\n" + "="*70)
    print("最终测试结果对比汇总")
    print("="*70)
    
    compare_data = []
    for name, hist in results.items():
        best_idx = np.argmax(hist["test_acc"])
        best_acc = hist["test_acc"][best_idx]
        best_loss = hist["test_loss"][best_idx]
        
        report = classification_report(hist["best_true"], hist["best_pred"], target_names=CLASS_NAMES, output_dict=True)
        cat_f1 = report["cat"]["f1-score"]
        dog_f1 = report["dog"]["f1-score"]
        
        compare_data.append({
            "Model": name,
            "Params": models_info[name]["params"],
            "Best Test Acc": best_acc,
            "Best Test Loss": best_loss,
            "Cat F1": cat_f1,
            "Dog F1": dog_f1
        })

    header = f"{'Model':<12} | {'Params':<10} | {'Test Acc':<10} | {'Test Loss':<10} | {'Cat F1':<8} | {'Dog F1':<8}"
    print(header)
    print("-" * len(header))
    for row in compare_data:
        print(f"{row['Model']:<12} | {row['Params']:<10,} | {row['Best Test Acc']:<10.2f}% | {row['Best Test Loss']:<10.4f} | {row['Cat F1']:<8.4f} | {row['Dog F1']:<8.4f}")

    # =============================
    # 9. 可视化对比
    # =============================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 图1：Test Accuracy 曲线对比
    for name, hist in results.items():
        axes[0].plot(hist["test_acc"], label=name, linewidth=2)
    axes[0].set_title("Test Accuracy Comparison (Adaptive CBAM)", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 图2：混淆矩阵对比
    for i, (name, hist) in enumerate(results.items()):
        cm = confusion_matrix(hist["best_true"], hist["best_pred"])
        axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        axes[1].set_title(f"Confusion Matrix: {name}", fontsize=14)
        tick_marks = np.arange(len(CLASS_NAMES))
        axes[1].set_xticks(tick_marks)
        axes[1].set_xticklabels(CLASS_NAMES, rotation=45)
        axes[1].set_yticks(tick_marks)
        axes[1].set_yticklabels(CLASS_NAMES)
        
        thresh = cm.max() / 2.0
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                axes[1].text(c, r, format(cm[r, c], 'd'),
                         ha="center", va="center",
                         color="white" if cm[r, c] > thresh else "black")
                         
    plt.tight_layout()
    plt.savefig("model_comparison_adaptive.png", dpi=150)
    plt.show()

    print("\n✅ 对比实验完成！图表已保存为 'model_comparison_adaptive.png'")