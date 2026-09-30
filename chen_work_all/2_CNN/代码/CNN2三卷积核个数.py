import torch
import torch.nn as nn
import torch.optim as optim
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

set_seed(42)

# =============================
# 2. 参数设置
# =============================
BATCH_SIZE = 64          # ★ 增大batch size以配合BatchNorm，训练更稳定
EPOCHS = 100
LR = 0.001
WEIGHT_DECAY = 1e-4      # ★ L2正则化系数
FC_HIDDEN = 512
PATIENCE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

TARGET_CLASSES = [2, 3, 4, 5, 7]
CLASS_NAMES = ["bird", "cat", "deer", "dog", "horse"]

# =============================
# 3. 数据集处理（灰度图）
# =============================
class MyCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train=train, download=True, transform=transform)
        mask = np.isin(self.targets, TARGET_CLASSES)
        self.data = self.data[mask]
        old_labels = np.array(self.targets)[mask]
        label_map = {2: 0, 3: 1, 4: 2, 5: 3, 7: 4}
        self.targets = [label_map[x] for x in old_labels]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,), (0.2470,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,), (0.2470,))
])

train_dataset = MyCIFAR10("./data", True, train_transform)
test_dataset = MyCIFAR10("./data", False, test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {len(train_dataset)}张")
print(f"测试集: {len(test_dataset)}张")


# =============================
# 4. 通用3层CNN模型（★ 支持自定义每层通道数 + 防过拟合）
# =============================
class CNN3Layer(nn.Module):
    """
    固定3层卷积+池化结构，通过 channels 列表控制每层卷积核数量
    包含 BatchNorm + Dropout + WeightDecay 三重防过拟合机制
    """
    def __init__(self, channels=[32, 32, 32]):
        super().__init__()
        assert len(channels) == 3, "channels 必须包含3个元素"
        c1, c2, c3 = channels

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),       # ★ BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32->16

            # Conv Block 2
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),       # ★ BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16->8

            # Conv Block 3
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),       # ★ BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 8->4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),          # ★ Dropout 防过拟合
            nn.Linear(c3 * 4 * 4, FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),          # ★ 第二层Dropout（较小概率）
            nn.Linear(FC_HIDDEN, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================
# 5. 训练函数（含早停 + Weight Decay）
# =============================
def train(model, name, patience=PATIENCE):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # ★ 加入 weight_decay 实现L2正则化
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_pred, best_true = None, None

    best_test_acc = 0.0
    no_improve_count = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        tr_loss = loss_sum / len(train_loader)
        tr_acc = correct / total * 100

        # --- 测试阶段 ---
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        all_pred, all_true = [], []

        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                out = model(img)
                loss = criterion(out, label)

                loss_sum += loss.item()
                pred = torch.argmax(out, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(label.cpu().numpy())

        te_loss = loss_sum / len(test_loader)
        te_acc = correct / total * 100

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
            print(f"[{name}] Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Acc:{tr_acc:.2f}% Loss:{tr_loss:.4f} | "
                  f"Test Acc:{te_acc:.2f}% Loss:{te_loss:.4f} | "
                  f"Best:{best_test_acc:.2f}% Patience:{no_improve_count}/{patience}")

        if no_improve_count >= patience:
            print(f"⚠️ [{name}] 早停触发! 最佳Epoch:{best_epoch}, 最佳Test Acc:{best_test_acc:.2f}%")
            break

    history["best_pred"] = best_pred
    history["best_true"] = best_true
    return history


# =============================
# 6. 运行对比实验（★ 5组通道配置）
# =============================
model_configs = {
    "3×64":           [64, 64, 64],
    "3×128":          [128, 128, 128],
    "3×256":          [256, 256, 256],
    "32-64-128":      [32, 64, 128],
    "64-128-256":     [64, 128, 256],     # ★ 注: 您原文写258，此处修正为256(惯用值)
}

results = {}
for name, channels in model_configs.items():
    print(f"\n{'='*60}")
    print(f"开始训练: {name} | 通道配置: {channels}")
    print(f"{'='*60}")
    model = CNN3Layer(channels=channels)
    results[name] = train(model, name)


# =============================
# 7. 输出最终结果 & 分类报告
# =============================
print("\n" + "="*60)
print("最终测试结果汇总")
print("="*60)
for name, r in results.items():
    best_idx = np.argmax(r["test_acc"])
    print(f"{name:15s} | Best Test Acc: {r['test_acc'][best_idx]:.2f}% | "
          f"Test Loss: {r['test_loss'][best_idx]:.4f}")

best_name = max(results, key=lambda x: max(results[x]["test_acc"]))
print(f"\n★ 最佳模型: {best_name}")
print(classification_report(
    results[best_name]["best_true"],
    results[best_name]["best_pred"],
    target_names=CLASS_NAMES, digits=4
))


# =============================
# 8. 可视化
# =============================
# 8.1 Loss曲线
plt.figure(figsize=(12, 5))
for name, r in results.items():
    plt.plot(r["test_loss"], label=name)
plt.xlabel("Epoch"); plt.ylabel("Test Loss")
plt.title("Test Loss Comparison (Channel Configs)"); plt.legend(); plt.grid()
plt.savefig("loss_compare_channels.png", dpi=150); plt.show()

# 8.2 Accuracy曲线
plt.figure(figsize=(12, 5))
for name, r in results.items():
    plt.plot(r["test_acc"], label=name)
plt.xlabel("Epoch"); plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Comparison (Channel Configs)"); plt.legend(); plt.grid()
plt.savefig("accuracy_compare_channels.png", dpi=150); plt.show()

# 8.3 混淆矩阵（最佳模型）
cm = confusion_matrix(results[best_name]["best_true"], results[best_name]["best_pred"])
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
plt.yticks(tick_marks, CLASS_NAMES)
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.title(f"Confusion Matrix ({best_name})")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("confusion_matrix_channels.png", dpi=150)
plt.show()