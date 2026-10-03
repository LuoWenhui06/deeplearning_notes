# preprocess.py
import os
import re
import json
import hashlib
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== 自定义 JSON 编码器 =====================
class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 数据类型的 JSON 编码器，防止 int64/float64 序列化报错"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ===================== 配置区 =====================
DATA_DIR = r"C:\Users\16131\Desktop\陈学姐作业\3_RNN\4_LSTM\aclImdb"
OUTPUT_DIR = r"C:\Users\16131\Desktop\IMDB_LSTM_Experiment"
SEED = 42
MAX_LEN = None      # 由Step4自动确定
MAX_VOCAB = None    # 由Step5自动确定
PAD_ID, UNK_ID = 0, 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)

# ===================== Step 1-3: 读取与清洗 =====================
def load_imdb(split):
    """读取aclImdb原始数据（含进度条）"""
    texts, labels = [], []
    for label_name, label_val in [("pos", 1), ("neg", 0)]:
        dir_path = os.path.join(DATA_DIR, split, label_name)
        file_list = os.listdir(dir_path)
        # 添加tqdm进度条
        for fname in tqdm(file_list, desc=f"Loading {split}/{label_name}", ncols=80):
            with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as f:
                text = f.read()
            # Step1: 清洗HTML标签
            text = re.sub(r"<[^>]+>", " ", text)
            # Step2: 小写化
            text = text.lower()
            # Step3: 正则分词(单词+标点独立)
            tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
            texts.append(tokens)
            labels.append(label_val)
    return texts, labels

print(">>> Loading raw data...")
train_texts, train_labels = load_imdb("train")
test_texts, test_labels = load_imdb("test")

# 分层抽样划分: 25k -> 20k train + 5k val
indices = np.arange(len(train_texts))
train_idx, val_idx = train_test_split(indices, test_size=5000, random_state=SEED, stratify=train_labels)

train_20k = [train_texts[i] for i in train_idx]
train_20k_labels = [train_labels[i] for i in train_idx]
val_5k = [train_texts[i] for i in val_idx]
val_5k_labels = [train_labels[i] for i in val_idx]

print(f">>> Split complete -> Train: {len(train_20k)}, Val: {len(val_5k)}, Test: {len(test_texts)}")

# ===================== Step 4: 长度统计 & 确定MAX_LEN =====================
train_lengths = [len(t) for t in train_20k]
stats = {
    "min": int(np.min(train_lengths)),
    "median": float(np.median(train_lengths)),
    "mean": float(np.mean(train_lengths)),
    "p90": float(np.percentile(train_lengths, 90)),
    "p95": float(np.percentile(train_lengths, 95)),
    "max": int(np.max(train_lengths))
}
print(f">>> Length Stats (20k Train): {stats}")

# 可视化: 长度分布直方图+箱线图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(train_lengths, bins=100, edgecolor="black")
axes[0].set_title("Token Length Distribution (20k Train)")
axes[0].axvline(stats["p95"], color="r", linestyle="--", label=f"p95={stats['p95']:.0f}")
axes[0].legend()
axes[1].boxplot(train_lengths, vert=True)
axes[1].set_title("Length Boxplot")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step4_length_distribution.png"), dpi=150)
plt.close()

# 覆盖率曲线 → 自动确定MAX_LEN
candidates = list(range(100, stats["max"]+1, 50))
coverages = [sum(1 for l in train_lengths if l <= c) / len(train_lengths) for c in candidates]

# 决策: 取>=95%覆盖率的最小候选值
MAX_LEN = next((c for c, cov in zip(candidates, coverages) if cov >= 0.95), candidates[-1])
print(f">>> Determined MAX_LEN = {MAX_LEN}")

plt.figure(figsize=(10, 5))
plt.plot(candidates, coverages, marker="o", markersize=3)
plt.axhline(0.95, color="r", linestyle="--", label="95% coverage")
plt.axvline(MAX_LEN, color="g", linestyle="--", label=f"MAX_LEN={MAX_LEN}")
plt.xlabel("MAX_LEN"); plt.ylabel("Coverage")
plt.title("Truncation Coverage Analysis (Head Truncation)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step4_truncation_analysis.png"), dpi=150)
plt.close()

# ===================== Step 5: 词频统计 & 确定MAX_VOCAB =====================
word_counter = Counter()
for tokens in tqdm(train_20k, desc="Counting words", ncols=80):
    word_counter.update(tokens)

# 累积覆盖率
total_tokens = sum(word_counter.values())
# 第一排序词频降序, 第二排序字典序升序
sorted_words = sorted(word_counter.items(), key=lambda x: (-x[1], x[0]))
cumsum = np.cumsum([cnt for _, cnt in sorted_words]) / total_tokens

# 找95%覆盖率对应词表大小
idx_95 = np.searchsorted(cumsum, 0.95) + 1
MAX_VOCAB = min(idx_95, len(sorted_words))
print(f">>> Determined MAX_VOCAB = {MAX_VOCAB}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumsum)+1), cumsum)
plt.axhline(0.95, color="r", linestyle="--", label="95% token coverage")
plt.axvline(MAX_VOCAB, color="g", linestyle="--", label=f"MAX_VOCAB={MAX_VOCAB}")
plt.xlabel("Vocabulary Size"); plt.ylabel("Cumulative Token Coverage")
plt.title("Word Frequency Coverage Curve")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step5_coverage_curve.png"), dpi=150)
plt.close()

# Top-50词频图
top50 = sorted_words[:50]
plt.figure(figsize=(16, 5))
plt.bar([w for w, _ in top50], [c for _, c in top50])
plt.xticks(rotation=90); plt.title("Top-50 Word Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step6_word_frequency.png"), dpi=150)
plt.close()

# ===================== Step 6: 构建词表 & 编码 =====================
vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
for word, _ in sorted_words[:MAX_VOCAB]:
    vocab[word] = len(vocab)

# config_hash
config_str = f"{MAX_LEN}_{MAX_VOCAB}_{SEED}"
config_hash = hashlib.md5(config_str.encode()).hexdigest()

# 保存词表 (使用 NumpyEncoder 防止潜在的类型问题)
with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump({"vocab": vocab, "config_hash": config_hash}, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def encode_and_pad(tokens, vocab_dict, max_len):
    """头部截断 + 尾部PAD填充"""
    ids = [vocab_dict.get(t, UNK_ID) for t in tokens]
    # 头部截断: 只保留前max_len个token
    ids = ids[:max_len]
    # 尾部填充
    actual_len = len(ids)
    ids += [PAD_ID] * (max_len - actual_len)
    return ids, actual_len

def encode_dataset(texts, labels, vocab_dict, max_len):
    encoded_seqs, lengths, lbls = [], [], []
    for tokens, label in tqdm(zip(texts, labels), total=len(texts), desc="Encoding", ncols=80):
        ids, length = encode_and_pad(tokens, vocab_dict, max_len)
        encoded_seqs.append(ids)
        lengths.append(length)
        lbls.append(label)
    return {
        "sequences": torch.tensor(encoded_seqs, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(lbls, dtype=torch.float32)
    }

print(">>> Encoding datasets...")
train_data = encode_dataset(train_20k, train_20k_labels, vocab, MAX_LEN)
val_data = encode_dataset(val_5k, val_5k_labels, vocab, MAX_LEN)
test_data = encode_dataset(test_texts, test_labels, vocab, MAX_LEN)

torch.save(train_data, os.path.join(OUTPUT_DIR, "train_encoded.pt"))
torch.save(val_data, os.path.join(OUTPUT_DIR, "val_encoded.pt"))
torch.save(test_data, os.path.join(OUTPUT_DIR, "test_encoded.pt"))

# 保存预处理统计信息 (使用 NumpyEncoder 解决 int64 报错)
stats_output = {
    **stats, 
    "MAX_LEN": MAX_LEN, 
    "MAX_VOCAB": MAX_VOCAB, 
    "vocab_size": len(vocab), 
    "config_hash": config_hash
}
with open(os.path.join(OUTPUT_DIR, "preprocess_stats.json"), "w") as f:
    json.dump(stats_output, f, indent=2, cls=NumpyEncoder)

print(f">>> Preprocessing complete! All outputs saved to: {OUTPUT_DIR}")