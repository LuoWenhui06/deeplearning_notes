import os
import ast
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd

# ==================== 1. 全局配置与路径设置 ====================
输出根目录 = os.path.join(os.path.expanduser("~"), "Desktop", "result2")
os.makedirs(输出根目录, exist_ok=True)

随机种子 = 42

def 设置全局种子(种子值):
    random.seed(种子值)
    np.random.seed(种子值)
    torch.manual_seed(种子值)
    torch.cuda.manual_seed_all(种子值)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

设置全局种子(随机种子)

批次大小 = 64
训练轮数 = 45
学习率 = 0.01
动量系数 = 0.9
权重衰减 = 1e-4
早停耐心值 = 10

设备 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[系统信息] 当前使用设备: {设备}")
print(f"[路径信息] 所有结果将保存至: {输出根目录}")


# ==================== 2. 数据集构建与预处理 ====================
def 加载并划分数据集():
    变换操作 = transforms.Compose([transforms.ToTensor()])

    原始训练集 = datasets.MNIST(root='./data', train=True, download=True, transform=变换操作)
    原始测试集 = datasets.MNIST(root='./data', train=False, download=True, transform=变换操作)

    所有训练索引 = list(range(len(原始训练集)))
    random.shuffle(所有训练索引)
    新训练索引 = 所有训练索引[:50000]
    新测试索引 = 所有训练索引[50000:]

    新训练集 = torch.utils.data.Subset(原始训练集, 新训练索引)
    新测试集 = torch.utils.data.Subset(原始训练集, 新测试索引)
    新验证集 = 原始测试集

    assert len(新训练集) == 50000
    assert len(新测试集) == 10000
    assert len(新验证集) == 10000
    print(f"[数据校验] 训练集:{len(新训练集)} | 新测试集:{len(新测试集)} | 验证集:{len(新验证集)}")

    return 新训练集, 新测试集, 新验证集


# ==================== 3. MLP模型定义 ====================
class 多层感知机模型(nn.Module):
    def __init__(self, 隐藏层结构列表, dropout比率=0.0):
        super().__init__()
        层列表 = []
        输入维度 = 28 * 28

        for 隐藏节点数 in 隐藏层结构列表:
            层列表.append(nn.Linear(输入维度, 隐藏节点数))
            层列表.append(nn.ReLU())
            if dropout比率 > 0:
                层列表.append(nn.Dropout(dropout比率))
            输入维度 = 隐藏节点数

        层列表.append(nn.Linear(输入维度, 10))
        self.网络层 = nn.Sequential(*层列表)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.网络层(x)


# ==================== 4. 核心训练引擎 ====================
def 计算加权平均损失(批次损失列表, 总样本数):
    if not 批次损失列表:
        return 0.0
    权重列表 = []
    for i in range(len(批次损失列表)):
        if i == len(批次损失列表) - 1:
            尾批样本数 = 总样本数 - (len(批次损失列表) - 1) * 批次大小
            权重 = (尾批样本数 / 总样本数) * 100
        else:
            权重 = (批次大小 / 总样本数) * 100
        权重列表.append(权重)
    return sum(l * w for l, w in zip(批次损失列表, 权重列表)) / sum(权重列表)


def 训练单个实验组(组别名, 隐藏层结构, dropout比率, 训练集, 验证集, 测试集):
    print(f"\n{'='*50}")
    print(f"[开始训练] 组别名: {组别名} | 结构: {隐藏层结构} | Dropout: {dropout比率}")

    模型 = 多层感知机模型(隐藏层结构, dropout比率).to(设备)
    损失函数 = nn.CrossEntropyLoss()
    优化器 = optim.SGD(模型.parameters(), lr=学习率, momentum=动量系数, weight_decay=权重衰减)

    训练加载器 = DataLoader(训练集, batch_size=批次大小, shuffle=True)
    验证加载器 = DataLoader(验证集, batch_size=批次大小, shuffle=False)
    测试加载器 = DataLoader(测试集, batch_size=批次大小, shuffle=False)

    训练损失历史 = []
    验证损失历史 = []
    最佳验证损失 = float('inf')
    早停计数器 = 0
    最优模型状态 = None

    for epoch in range(训练轮数):
        # 训练阶段
        模型.train()
        当前epoch批次损失 = []
        for 图像, 标签 in 训练加载器:
            图像, 标签 = 图像.to(设备), 标签.to(设备)
            输出 = 模型(图像)
            损失 = 损失函数(输出, 标签)
            优化器.zero_grad()
            损失.backward()
            优化器.step()
            当前epoch批次损失.append(损失.item())

        epoch平均损失 = 计算加权平均损失(当前epoch批次损失, len(训练集))
        训练损失历史.append(epoch平均损失)

        # 验证阶段
        模型.eval()
        验证总损失 = 0.0
        with torch.no_grad():
            for 图像, 标签 in 验证加载器:
                图像, 标签 = 图像.to(设备), 标签.to(设备)
                输出 = 模型(图像)
                验证总损失 += 损失函数(输出, 标签).item()
        验证epoch损失 = 验证总损失 / len(验证加载器)
        验证损失历史.append(验证epoch损失)

        print(f"[{组别名}] Epoch: {epoch+1}/{训练轮数} | Train Loss: {epoch平均损失:.4f} | Val Loss: {验证epoch损失:.4f}")

        # 早停检查
        if 验证epoch损失 < 最佳验证损失:
            最佳验证损失 = 验证epoch损失
            早停计数器 = 0
            最优模型状态 = {k: v.cpu().clone() for k, v in 模型.state_dict().items()}
        else:
            早停计数器 += 1
            if 早停计数器 >= 早停耐心值:
                print(f"[早停触发] {组别名} 在第 {epoch+1} 轮停止训练")
                break

    # 测试集评估
    模型.load_state_dict(最优模型状态)
    模型.to(设备)
    模型.eval()
    正确数 = 0
    总样本数 = 0
    with torch.no_grad():
        for 图像, 标签 in 测试加载器:
            图像, 标签 = 图像.to(设备), 标签.to(设备)
            输出 = 模型(图像)
            _, 预测 = torch.max(输出, 1)
            正确数 += (预测 == 标签).sum().item()
            总样本数 += 标签.size(0)
    测试准确率 = 正确数 / 总样本数
    print(f"[测试完成] {组别名} 测试准确率: {测试准确率:.4f}")

    return {
        "组别名": 组别名,
        "结构": str(隐藏层结构),
        "Dropout": dropout比率,
        "最佳验证损失": 最佳验证损失,
        "测试准确率": 测试准确率,
        "训练损失历史": 训练损失历史,
        "验证损失历史": 验证损失历史,
        "最优模型状态": 最优模型状态
    }


# ==================== 5. 可视化与错误分析 ====================
def 绘制损失曲线(实验结果列表):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight'] = 'normal'

    图编号 = 0
    for i in range(0, len(实验结果列表), 3):
        子图组 = 实验结果列表[i:i+3]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        if len(子图组) < 3:
            axes = list(axes) + [None] * (3 - len(子图组))

        for idx, (ax, 结果) in enumerate(zip(axes, 子图组)):
            if ax is None:
                continue
            颜色 = '#2ecc71' if idx == 0 else ('#f1c40f' if idx == len(子图组)-1 else '#3498db')
            ax.plot(结果["训练损失历史"], label="训练损失", color=颜色, linewidth=2, marker='o', markevery=5, markersize=4)
            ax.plot(结果["验证损失历史"], label="验证损失", linestyle="--", color=颜色, linewidth=2, marker='s', markevery=5, markersize=4)
            ax.set_title(f"{结果['组别名']} (测试准确率: {结果['测试准确率']:.4f})", fontsize=14)
            ax.set_xlabel("训练轮次", fontsize=12)
            ax.set_ylabel("损失值", fontsize=12)
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)
        plt.suptitle(f"MLP损失曲线对比 (第{图编号+1}页)", fontsize=16, y=1.02)
        plt.tight_layout()
        图表保存路径 = os.path.join(输出根目录, f"损失曲线图_第{图编号+1}页.png")
        plt.savefig(图表保存路径, dpi=150, bbox_inches='tight')
        plt.close()
        图编号 += 1


def 生成错误分析(模型, 测试集):
    加载器 = DataLoader(测试集, batch_size=1000, shuffle=False)
    模型.eval()
    错误样本列表 = []
    with torch.no_grad():
        for 图像, 标签 in 加载器:
            图像, 标签 = 图像.to(设备), 标签.to(设备)
            输出 = 模型(图像)
            概率 = torch.softmax(输出, dim=1)
            预测置信度, 预测 = torch.max(概率, 1)
            for j in range(len(标签)):
                if 预测[j] != 标签[j]:
                    错误样本列表.append((
                        图像[j].cpu(),
                        标签[j].item(),
                        预测[j].item(),
                        预测置信度[j].item()
                    ))
            if len(错误样本列表) >= 9:
                break

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    错因标注映射 = {(4,9):"形似混淆(4/9)", (9,4):"形似混淆(9/4)", (7,1):"笔画断裂(7/1)"}
    for idx, ax in enumerate(axes.flat):
        if idx >= len(错误样本列表):
            break
        图像, 真实标签, 预测标签, 置信度 = 错误样本列表[idx]
        ax.imshow(图像.squeeze(), cmap='gray')
        错因 = 错因标注映射.get((真实标签, 预测标签), f"误判为{预测标签}")
        ax.set_title(f"真:{真实标签} -> 预:{预测标签} (置信度:{置信度:.2f})\n{错因}", fontsize=11)
        ax.axis('off')
    plt.suptitle("典型错误样本分析", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(输出根目录, "错误分析图.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 6. 主执行流程 ====================
if __name__ == "__main__":
    设置全局种子(随机种子)
    训练集, 测试集, 验证集 = 加载并划分数据集()

    所有实验结果 = []

    # 直接指定四组对比实验配置
    指定实验配置 = {
        "L2_max": {"结构": [178, 178], "Dropout": 0.0},
        "D1":     {"结构": [178, 178], "Dropout": 0.2},
        "L2_plu": {"结构": [256, 256], "Dropout": 0.0},
        "D1_plu": {"结构": [256, 256], "Dropout": 0.2}
    }

    print(f"\n[实验计划] 共 {len(指定实验配置)} 组对比实验: {list(指定实验配置.keys())}")
    
    for 别名, 配置 in 指定实验配置.items():
        结果 = 训练单个实验组(
            组别名=别名,
            隐藏层结构=配置["结构"],
            dropout比率=配置["Dropout"],
            训练集=训练集,
            验证集=验证集,
            测试集=测试集
        )
        所有实验结果.append(结果)

    # 生成Excel汇总表
    汇总表 = sorted(所有实验结果, key=lambda x: x["最佳验证损失"])
    excel数据列表 = []
    for i, r in enumerate(汇总表):
        excel数据列表.append({
            "排名": i + 1,
            "组别名": r['组别名'],
            "隐藏层结构": r['结构'],
            "Dropout比率": r['Dropout'],
            "最佳验证损失": round(r['最佳验证损失'], 4),
            "测试准确率": round(r['测试准确率'], 4)
        })

    df = pd.DataFrame(excel数据列表)
    excel保存路径 = os.path.join(输出根目录, "最终结果汇总表.xlsx")
    df.to_excel(excel保存路径, index=False, sheet_name="实验结果汇总")
    print(f"\n[Excel已保存] {excel保存路径}")

    # 生成可视化图片
    绘制损失曲线(所有实验结果)

    # 错误分析（使用验证损失最优的模型）
    最优组结果 = 汇总表[0]
    最优组隐藏层结构 = ast.literal_eval(最优组结果["结构"])
    最优组dropout = 最优组结果["Dropout"]
    print(f"[错误分析] 使用最优模型: {最优组结果['组别名']} | 结构: {最优组隐藏层结构} | Dropout: {最优组dropout}")

    最优模型 = 多层感知机模型(最优组隐藏层结构, dropout比率=最优组dropout).to(设备)
    最优模型.load_state_dict(最优组结果["最优模型状态"])
    生成错误分析(最优模型, 测试集)

    print(f"\n[全部完成] 所有结果已保存至: {输出根目录}")