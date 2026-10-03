# train.py
import os
import time
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== 配置区 =====================
OUTPUT_DIR = r"C:\Users\16131\Desktop\IMDB_LSTM_Experiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 64
EMBED_DIM = 128
LR = 0.001
GRAD_CLIP = 1.0
DROPOUT = 0.5
MAX_EPOCHS = 30
PATIENCE = 5
LOG_INTERVAL = 31  # ~10% of 313 steps

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===================== 加载数据 =====================
print(">>> Loading encoded data...")
train_data = torch.load(os.path.join(OUTPUT_DIR, "train_encoded.pt"), weights_only=True)
val_data = torch.load(os.path.join(OUTPUT_DIR, "val_encoded.pt"), weights_only=True)
test_data = torch.load(os.path.join(OUTPUT_DIR, "test_encoded.pt"), weights_only=True)

with open(os.path.join(OUTPUT_DIR, "preprocess_stats.json")) as f:
    prep_stats = json.load(f)

VOCAB_SIZE = prep_stats["vocab_size"]
MAX_LEN = prep_stats["MAX_LEN"]

train_loader = DataLoader(TensorDataset(train_data["sequences"], train_data["lengths"], train_data["labels"]),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data["sequences"], val_data["lengths"], val_data["labels"]),
                        batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(test_data["sequences"], test_data["lengths"], test_data["labels"]),
                         batch_size=BATCH_SIZE, shuffle=False)

# ===================== 模型定义 =====================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, bidirectional, dropout, fc_input_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.fc = nn.Linear(fc_input_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        # pack_padded_sequence: 跳过PAD, 确保隐藏状态仅来自真实token
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        
        # 取最后一层的输出
        if self.bidirectional:
            # 拼接正向和反向的最后一层
            h_forward = hidden[-2]
            h_backward = hidden[-1]
            out = torch.cat([h_forward, h_backward], dim=1)
        else:
            out = hidden[-1]
            
        out = self.dropout(out)
        logits = self.fc(out).squeeze(1)
        return logits

MODEL_CONFIGS = {
    "A_UniLSTM_64":  {"hidden": 64,  "num_layers": 2, "bidir": False, "fc_dim": 64},
    "B_UniLSTM_128": {"hidden": 128, "num_layers": 2, "bidir": False, "fc_dim": 128},
    "C_BiLSTM_64":   {"hidden": 64,  "num_layers": 2, "bidir": True,  "fc_dim": 128},
}

# ===================== 训练与评估函数 =====================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_probs, all_labels, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for seqs, lens, labels in loader:
            seqs, lens, labels = seqs.to(DEVICE), lens.to(DEVICE), labels.to(DEVICE)
            logits = model(seqs, lens)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {"loss": avg_loss, "acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": auc, "cm": cm,
            "preds": all_preds, "probs": all_probs, "labels": all_labels}

def train_one_model(name, config):
    print(f"\n{'='*60}")
    print(f">>> Training {name}")
    print(f"{'='*60}")

    model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, config["hidden"], config["num_layers"],
                           config["bidir"], DROPOUT, config["fc_dim"]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    params = count_params(model)
    print(f"Parameters: {params:,}")

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_epoch, patience_counter = 0.0, 0, 0
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        ep_start = time.time()

        # 训练批次进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{MAX_EPOCHS}", ncols=100, leave=False)
        for batch_idx, (seqs, lens, labels) in enumerate(pbar, 1):
            seqs, lens, labels = seqs.to(DEVICE), lens.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seqs, lens)
            loss = criterion(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            bs = labels.size(0)
            epoch_loss += loss.item() * bs
            preds = (torch.sigmoid(logits) >= 0.5).long()
            epoch_correct += (preds == labels.long()).sum().item()
            epoch_total += bs

            # 更新进度条后缀信息
            pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.2f}")

            # 日志: 每31批打印到控制台
            if batch_idx % LOG_INTERVAL == 0 or batch_idx == len(train_loader):
                elapsed = time.time() - ep_start
                b_acc = (preds == labels.long()).float().mean().item() * 100
                print(f"[{name}] [Train] Epoch: [{epoch:02d}/{MAX_EPOCHS}] | Batch: [{batch_idx:03d}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | Acc: {b_acc:.2f}% | Grad_Norm: {grad_norm:.2f} | LR: {LR:.1e} | Time: {elapsed:.1f}s")

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        # Validation
        val_metrics = evaluate(model, val_loader, criterion)
        print(f"[{name}] [Eval]  Epoch: [{epoch:02d}/{MAX_EPOCHS}] | Val_Loss: {val_metrics['loss']:.4f} | "
              f"Val_Acc: {val_metrics['acc']*100:.2f}% | Val_F1: {val_metrics['f1']:.3f} | "
              f"Best_F1: {best_f1:.3f} (Ep{best_epoch:02d}) | LR: {LR:.1e}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{name}_best_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"[{name}] [Stop]  Early Stopping at Epoch {epoch}. Best Val_F1: {best_f1:.3f} (Epoch {best_epoch}). Restoring best weights.")
            break

    train_time = (time.time() - start_time) / 60.0

    # 加载最佳权重进行测试
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{name}_best_model.pth"), weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion)

    # 保存训练日志CSV
    with open(os.path.join(OUTPUT_DIR, f"{name}_training_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({k: history[k][i] for k in history})

    # 训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{name} Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc")
    axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc")
    axes[1].set_title(f"{name} Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    axes[2].plot(history["epoch"], history["val_f1"], label="Val F1", color="green")
    axes[2].set_title(f"{name} Val F1"); axes[2].legend(); axes[2].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_train_val_curves.png"), dpi=150)
    plt.close()

    return {
        "name": name, "params": params, "best_epoch": best_epoch, "train_time": train_time,
        "test_acc": test_metrics["acc"], "test_f1": test_metrics["f1"], "test_auc": test_metrics["auc"],
        "test_cm": test_metrics["cm"], "history": history, 
        "test_probs": test_metrics["probs"], "test_labels": test_metrics["labels"]
    }

# ===================== 主流程 =====================
results = {}
for name, cfg in MODEL_CONFIGS.items():
    results[name] = train_one_model(name, cfg)

# ===================== 单模型测试报告 =====================
for name, r in results.items():
    cm = r["test_cm"]
    report = f"""==================== Final Test Report ====================
Model: {name}
Test Accuracy : {r['test_acc']*100:.2f}%
Test F1-Score : {r['test_f1']:.3f}
Test AUC-ROC  : {r['test_auc']:.3f}

Confusion Matrix:
                 Predicted Neg  Predicted Pos
Actual Neg          {cm[0][0]:>6d}          {cm[0][1]:>6d}
Actual Pos          {cm[1][0]:>6d}          {cm[1][1]:>6d}
==========================================================="""
    print(report)
    with open(os.path.join(OUTPUT_DIR, f"{name}_test_report.txt"), "w") as f:
        f.write(report)

# ===================== 三模型对比总报告 =====================
names = list(MODEL_CONFIGS.keys())
ra, rb, rc = results[names[0]], results[names[1]], results[names[2]]

comparison = f"""==================== Model Comparison Report ====================

┌─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┐
│     Model       │ Test Acc │ Test F1  │ Test AUC │ Params   │ Best Ep  │ Train Time │
├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────────┤
│ A: UniLSTM-64   │ {ra['test_acc']*100:>6.2f}%  │  {ra['test_f1']:.3f}   │  {ra['test_auc']:.3f}   │ {ra['params']:>7,d}  │   {ra['best_epoch']:>2d}     │  {ra['train_time']:>5.1f} min  │
│ B: UniLSTM-128  │ {rb['test_acc']*100:>6.2f}%  │  {rb['test_f1']:.3f}   │  {rb['test_auc']:.3f}   │ {rb['params']:>7,d}  │   {rb['best_epoch']:>2d}     │  {rb['train_time']:>5.1f} min  │
│ C: BiLSTM-64    │ {rc['test_acc']*100:>6.2f}%  │  {rc['test_f1']:.3f}   │  {rc['test_auc']:.3f}   │ {rc['params']:>7,d}  │   {rc['best_epoch']:>2d}     │  {rc['train_time']:>5.1f} min  │
└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────────┘

Pairwise Comparison:
  B vs A (容量效应): Acc {(rb['test_acc']-ra['test_acc'])*100:+.2f}%, F1 {rb['test_f1']-ra['test_f1']:+.3f}
  C vs B (结构效应): Acc {(rc['test_acc']-rb['test_acc'])*100:+.2f}%, F1 {rc['test_f1']-rb['test_f1']:+.3f}
  C vs A (综合效应): Acc {(rc['test_acc']-ra['test_acc'])*100:+.2f}%, F1 {rc['test_f1']-ra['test_f1']:+.3f}

================================================================"""
print(comparison)
with open(os.path.join(OUTPUT_DIR, "comparison_report.txt"), "w") as f:
    f.write(comparison)

# ===================== 对比可视化 =====================
print(">>> Generating comparison plots...")
# Acc对比柱状图
fig, ax = plt.subplots(figsize=(8, 5))
accs = [results[n]["test_acc"]*100 for n in names]
ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("Test Accuracy (%)"); ax.set_title("Model Comparison: Test Accuracy")
ax.set_ylim(min(accs)-3, max(accs)+3)
for i, v in enumerate(accs): ax.text(i, v+0.3, f"{v:.2f}%", ha="center")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "comparison_test_acc.png"), dpi=150); plt.close()

# F1对比柱状图
fig, ax = plt.subplots(figsize=(8, 5))
f1s = [results[n]["test_f1"] for n in names]
ax.bar(names, f1s, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("Test F1-Score"); ax.set_title("Model Comparison: Test F1")
ax.set_ylim(min(f1s)-0.03, max(f1s)+0.03)
for i, v in enumerate(f1s): ax.text(i, v+0.003, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "comparison_test_f1.png"), dpi=150); plt.close()

# ROC曲线叠加 (已清理冗余代码，直接复用 evaluate 返回的 probs 和 labels)
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#4C72B0", "#DD8452", "#55A868"]
for n, c in zip(names, colors):
    r = results[n]
    fpr, tpr, _ = roc_curve(r["test_labels"], r["test_probs"])
    ax.plot(fpr, tpr, color=c, label=f"{n} (AUC={r['test_auc']:.3f})")

ax.plot([0,1],[0,1],"k--")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves Comparison"); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "comparison_roc_curves.png"), dpi=150); plt.close()

# Val Loss收敛曲线叠加
fig, ax = plt.subplots(figsize=(10, 5))
for n, c in zip(names, colors):
    h = results[n]["history"]
    ax.plot(h["epoch"], h["val_loss"], color=c, label=n)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Loss")
ax.set_title("Validation Loss Convergence"); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "comparison_loss_curves.png"), dpi=150); plt.close()

print(f"\n>>> All training complete! Results saved to: {OUTPUT_DIR}")