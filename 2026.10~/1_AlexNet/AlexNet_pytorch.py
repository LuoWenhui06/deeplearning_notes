import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),                         #Conv1
            nn.ReLU(inplace=True),                                                         #ReLU1
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),                    #LRN1
            nn.MaxPool2d(kernel_size=3, stride=2),                                         #MaxPool1

            nn.Conv2d(96, 256, kernel_size=5, padding=2),                                  #Conv2
            nn.ReLU(inplace=True),                                                         #ReLU2
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),                    #LRN2
            nn.MaxPool2d(kernel_size=3, stride=2),                                         #MaxPool2

            nn.Conv2d(256, 384, kernel_size=3, padding=1),                                 #Conv3
            nn.ReLU(inplace=True),                                                         #ReLU3

            nn.Conv2d(384, 384, kernel_size=3, padding=1),                                 #Conv4
            nn.ReLU(inplace=True),                                                         #ReLU4

            nn.Conv2d(384, 256, kernel_size=3, padding=1),                                 #Conv5
            nn.ReLU(inplace=True),                                                         #ReLU5
            nn.MaxPool2d(kernel_size=3, stride=2),                                         #MaxPool3
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),                                                  #FC1
            nn.ReLU(inplace=True),                                                         #ReLU6
            nn.Dropout(p=0.5),                                                             #Dropout1

            nn.Linear(4096, 4096),                                                         #FC2
            nn.ReLU(inplace=True),                                                         #ReLU7
            nn.Dropout(p=0.5),                                                             #Dropout2
            nn.Linear(4096, num_classes),                                                  #FC3
        )

    def forward(self, x):
        x = self.features(x)                                                               #Feature extraction
        x = torch.flatten(x, 1)                                                            #Flatten
        x = self.classifier(x)                                                             #Classification
        return x

#在训练时给每张图像随机添加一个全局的颜色偏移，实现 PCA 颜色数据增强
class PCALighting:
    def __init__(self, eigenvalues, eigenvectors, alpha_std=0.1):
        self.alpha_std = alpha_std
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            from torchvision.transforms import functional as F
            img = F.to_tensor(img)
        alpha = torch.randn(3, dtype=img.dtype, device=img.device) * self.alpha_std
        noise = torch.mv(self.eigenvectors, self.eigenvalues * alpha)
        img = img + noise.view(3, 1, 1)
        return img.clamp(0.0, 1.0)    
 #一次遍历 CIFAR-10 训练集，同时计算：RGB 三通道均值（用于 Normalize）;RGB 协方差矩阵的特征值、特征向量（用于 PCA 颜色扰动）   
def compute_pca_and_mean(root):

    dataset = datasets.CIFAR10(root=root, train=True, transform=transforms.ToTensor(), download=True)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    pixels_list = []
    for images, _ in loader:
        pixels = images.permute(0, 2, 3, 1).reshape(-1, 3)
        pixels_list.append(pixels)
    pixels = torch.cat(pixels_list, dim=0).double() 
    # 均值
    mean = pixels.mean(dim=0)         
    # PCA：协方差矩阵特征分解
    centered = pixels - mean
    cov = torch.mm(centered.t(), centered) / (centered.size(0) - 1)  # (3, 3)
    eigval, eigvec = torch.linalg.eigh(cov)  
    eigval = eigval.flip(0)       
    eigvec = eigvec.flip(1)

    return mean.float(), eigval.float(), eigvec.float()

def data(root, batch_size, num_workers=0):
    mean, eigenvalues, eigenvectors = compute_pca_and_mean(root)
    print(f"CIFAR-10 train mean: {mean}")
    print(f"CIFAR-10 PCA eigenvalues: {eigenvalues}")

    tra_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        PCALighting(eigenvalues, eigenvectors, alpha_std=0.1),
        transforms.Normalize(mean=mean.tolist(), std=[1.0, 1.0, 1.0]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=[1.0, 1.0, 1.0]),
    ])

    train = datasets.CIFAR10(
        root=root, train=True, transform=tra_transform, download=True
    )
    test = datasets.CIFAR10(
        root=root, train=False, transform=test_transform, download=True
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, len(train), len(test)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="AlexNet on CIFAR-10")
    parser.add_argument("--data", type=str, default="./data", help="CIFAR-10 数据根目录")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save", type=str, default="alexnet_cifar10.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, train_size, test_size = data(
        root=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    print(f"Train samples: {train_size}, Test samples: {test_size}")

    model = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = test(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"LR {current_lr:.5f} | "
            f"TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f} | "
            f"TestLoss {test_loss:.4f} TestAcc {test_acc:.4f} | "
            f"Time {elapsed:.1f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), args.save)
            print(f"  -> Saved best model with TestAcc {test_acc:.4f} to {args.save}")

    print(f"Done. Best TestAcc: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()