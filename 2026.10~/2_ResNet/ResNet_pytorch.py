import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CIFAR10_ROOT = r"C:\Users\16131\Desktop\deeplearning_learning\2026.10~\1_AlexNet\data"

# 数据预处理：像素均值法归一化，数据增强：随机裁剪和水平翻转
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.ToTensor(),                              # PIL/ndarray -> Tensor [0,1]
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),    # 像素均值减法 + 标准化
    transforms.RandomCrop(32, padding=4),               # 先pad到40x40，再随机裁剪为 32x32
    transforms.RandomHorizontalFlip(p=0.5),             # 50%概率水平翻转
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_dataset = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=False, transform=train_transform)
test_dataset  = datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


# ResNet18
class BasicBlock(nn.Module):
    expansion = 1                                                          # block最终输出通道数是内部基准通道数的1倍(通道数不变)
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample                                        # 若不进行下采样，则恒等映射;若进行下采样，则投影残差连接

    def forward(self, x):
        identity = x                                                        # 保存输入，用于残差连接
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))      # basic block
        if self.downsample is not None:                                     # 如果下采样，则对输入进行投影,否则保持恒等映射
            identity = self.downsample(x)                                   # 对输入进行投影，使其通道数与残差块输出一致
        out += identity                                                     # 残差连接
        out = self.relu(out)                                                # ReLU激活
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 第一个卷积层，输入通道数为3，输出通道数为64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.maxpool = nn.Identity()  # 使用恒等映射代替最大池化层，即不降低分辨率
        self.in_channels = 64                                                # 初始残差块儿输入通道数
        self.layer1 = self.make_layer(64,  2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def make_layer(self, channels, blocks, stride):
        layers = []
        downsample = None                                                   # 初始化下采样模块为None
        # 判断残差块儿内部是否需要投影捷径
        if stride != 1 or self.in_channels != channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels)
            )
        layers.append(BasicBlock(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)                                            # 不是展平层，只是展平了个向量
        x = self.fc(x)
        return x

# 配置
device = torch.device("cuda")
model  = ResNet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

if __name__ == '__main__':
    epochs = 150
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(model, train_loader, device)
        test_acc  = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:3d} | Loss {running_loss/len(train_loader):.4f} "
              f"| Train Acc {train_acc:.4f} | Test Acc {test_acc:.4f}")