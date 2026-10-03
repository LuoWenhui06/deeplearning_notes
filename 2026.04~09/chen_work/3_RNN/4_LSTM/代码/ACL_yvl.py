"""
ACL IMDb 数据集预处理完整流水线（含下载进度条）
=============================================
6步流程：清洗HTML → 小写化 → 分词 → 长度分析与截断填充 → 覆盖率曲线确定词表 → 整数编码
所有可视化与结果文件保存至桌面 ACL_ycl 文件夹
"""

import os
import re
import sys
import json
import time
import urllib.request
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# ============================================================
# 全局配置
# ============================================================
# 自动检测桌面路径（兼容 Windows / macOS / Linux）
for candidate in ["~/Desktop", "~/桌面"]:
    p = Path(os.path.expanduser(candidate))
    if p.exists():
        DESKTOP = p
        break
else:
    DESKTOP = Path(os.path.expanduser("~")) / "Desktop"
    DESKTOP.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = DESKTOP / "ACL_ycl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = DESKTOP / "aclImdb"

# 可视化中文显示支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print(f"📁 输出目录: {OUTPUT_DIR}")


# ============================================================
# 第 0 步：数据加载（含自动下载 + 实时进度条）
# ============================================================
def load_imdb_data(data_dir: Path):
    """
    加载 ACL IMDb 数据集。
    如果本地不存在，自动从 Stanford 镜像下载（约 80MB），并实时显示进度。
    返回: (train_texts, train_labels, test_texts, test_labels)
    """
    if not data_dir.exists():
        print("⬇️  未找到本地数据集，正在下载 ACL IMDb Dataset (~80MB)...")
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        tar_path = DESKTOP / "aclImdb_v1.tar.gz"

        # ---------- 带进度条的下载（纯标准库实现） ----------
        def _progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded / total_size * 100)
                bar_len = 40
                filled = int(bar_len * percent / 100)
                bar = '█' * filled + '░' * (bar_len - filled)
                speed_mb = downloaded / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                sys.stdout.write(
                    f'\r   [{bar}] {percent:5.1f}% ({speed_mb:.1f}/{total_mb:.1f} MB)'
                )
            else:
                sys.stdout.write(
                    f'\r   已下载: {downloaded / 1024 / 1024:.1f} MB'
                )
            sys.stdout.flush()

        try:
            urllib.request.urlretrieve(url, str(tar_path), reporthook=_progress_hook)
            print()  # 下载完成后换行
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            print("   请手动下载: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
            print(f"   解压到: {DESKTOP}")
            raise SystemExit(1)
        # --------------------------------------------------

        print("📦 解压中...")
        with tarfile.open(str(tar_path), "r:gz") as tar:
            tar.extractall(path=str(DESKTOP))
        tar_path.unlink()
        print("✅ 数据集准备完成！\n")

    def read_split(split_dir):
        texts, labels = [], []
        for label, label_name in [(1, "pos"), (0, "neg")]:
            folder = split_dir / label_name
            for fpath in sorted(folder.glob("*.txt")):
                texts.append(fpath.read_text(encoding="utf-8"))
                labels.append(label)
        return texts, labels

    train_texts, train_labels = read_split(data_dir / "train")
    test_texts, test_labels = read_split(data_dir / "test")

    print(f"📊 数据加载完成:")
    print(f"   训练集: {len(train_texts)} 条 "
          f"(正例 {sum(train_labels)} / 负例 {len(train_labels) - sum(train_labels)})")
    print(f"   测试集: {len(test_texts)} 条 "
          f"(正例 {sum(test_labels)} / 负例 {len(test_labels) - sum(test_labels)})")
    print()
    return train_texts, train_labels, test_texts, test_labels


# ============================================================
# 第 1 步：清洗 HTML 标签
# ============================================================
def clean_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def step1_clean_html(train_texts, test_texts):
    print("🧹 [第1步] 清洗 HTML 标签...")
    sample_idx = 0
    print(f"   示例 (前80字符):")
    print(f"   清洗前: {train_texts[sample_idx][:80]}...")

    train_cleaned = [clean_html(t) for t in train_texts]
    test_cleaned = [clean_html(t) for t in test_texts]

    print(f"   清洗后: {train_cleaned[sample_idx][:80]}...")
    print(f"   ✅ 训练集 {len(train_cleaned)} 条 + 测试集 {len(test_cleaned)} 条 清洗完成\n")
    return train_cleaned, test_cleaned


# ============================================================
# 第 2 步：小写化
# ============================================================
def step2_lowercase(train_texts, test_texts):
    print("🔡 [第2步] 小写化...")
    sample_idx = 0
    print(f"   示例: '{train_texts[sample_idx][:50]}' → '{train_texts[sample_idx][:50].lower()}'")

    train_lower = [t.lower() for t in train_texts]
    test_lower = [t.lower() for t in test_texts]

    print(f"   ✅ 小写化完成\n")
    return train_lower, test_lower


# ============================================================
# 第 3 步：分词（按空格和标点）
# ============================================================
TOKEN_PATTERN = re.compile(r"""\b\w+(?:'\w+)?\b|[^\w\s]""")


def tokenize(text: str) -> list:
    return TOKEN_PATTERN.findall(text)


def step3_tokenize(train_texts, test_texts):
    print("✂️  [第3步] 分词...")
    train_tokens = [tokenize(t) for t in train_texts]
    test_tokens = [tokenize(t) for t in test_texts]

    sample_idx = 0
    print(f"   示例: '{train_texts[sample_idx][:60]}'")
    print(f"   分词: {train_tokens[sample_idx][:15]}...")
    print(f"   ✅ 训练集共 {sum(len(t) for t in train_tokens):,} 个 Token")
    print(f"   ✅ 测试集共 {sum(len(t) for t in test_tokens):,} 个 Token\n")
    return train_tokens, test_tokens


# ============================================================
# 第 4 步：序列长度统计、可视化与截断/填充
# ============================================================
def step4_length_analysis(train_tokens, output_dir: Path):
    print("📏 [第4步] 序列长度统计与可视化...")
    lengths = [len(t) for t in train_tokens]
    lengths_arr = np.array(lengths)

    stats = {
        "样本数": int(len(lengths)),
        "最小值": int(np.min(lengths_arr)),
        "中位数": float(np.median(lengths_arr)),
        "平均数": round(float(np.mean(lengths_arr)), 2),
        "标准差": round(float(np.std(lengths_arr)), 2),
        "90%分位数": int(np.percentile(lengths_arr, 90)),
        "95%分位数": int(np.percentile(lengths_arr, 95)),
        "99%分位数": int(np.percentile(lengths_arr, 99)),
        "最大值": int(np.max(lengths_arr)),
    }

    print("   长度统计:")
    for k, v in stats.items():
        print(f"     {k}: {v}")

    MAX_LEN = stats["95%分位数"]
    stats["选定的MAX_LEN"] = MAX_LEN
    coverage_at_maxlen = float(np.mean(lengths_arr <= MAX_LEN)) * 100
    stats["MAX_LEN覆盖率"] = f"{coverage_at_maxlen:.1f}%"
    print(f"\n   🎯 选定 MAX_LEN = {MAX_LEN} (覆盖 {coverage_at_maxlen:.1f}% 的训练样本)")

    # 可视化1：长度分布直方图 + 箱线图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1 = axes[0]
    ax1.hist(lengths, bins=200, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(x=MAX_LEN, color='red', linestyle='--', linewidth=2,
                label=f'MAX_LEN={MAX_LEN} (95%分位)')
    ax1.axvline(x=stats["平均数"], color='orange', linestyle='--', linewidth=1.5,
                label=f'平均数={stats["平均数"]}')
    ax1.axvline(x=stats["中位数"], color='green', linestyle='--', linewidth=1.5,
                label=f'中位数={stats["中位数"]}')
    ax1.set_xlabel('序列长度 (Token数)', fontsize=12)
    ax1.set_ylabel('样本数', fontsize=12)
    ax1.set_title('训练集序列长度分布直方图', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.boxplot(lengths, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='navy'),
                capprops=dict(color='navy'),
                flierprops=dict(marker='.', markerfacecolor='gray', markersize=2, alpha=0.3))
    ax2.set_ylabel('序列长度 (Token数)', fontsize=12)
    ax2.set_title('训练集序列长度箱线图', fontsize=14)
    ax2.set_xticklabels(['所有样本'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.annotate(f'最大值: {stats["最大值"]}', xy=(1.15, stats["最大值"]), fontsize=9, color='gray')
    ax2.annotate(f'95%分位: {MAX_LEN}', xy=(1.15, MAX_LEN), fontsize=9, color='red')
    ax2.annotate(f'中位数: {stats["中位数"]:.0f}', xy=(1.15, stats["中位数"]), fontsize=9, color='green')

    plt.tight_layout()
    fig.savefig(output_dir / "step4_length_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 长度分布图已保存: step4_length_distribution.png")

    # 可视化2：截断比例分析图
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    candidate_lens = list(range(100, min(stats["最大值"] + 1, 2000), 50))
    coverages = [float(np.mean(lengths_arr <= cl)) * 100 for cl in candidate_lens]
    ax3.plot(candidate_lens, coverages, 'b-o', markersize=3, linewidth=1.5)
    ax3.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% 覆盖率')
    ax3.axvline(x=MAX_LEN, color='red', linestyle='--', alpha=0.7)
    ax3.fill_between(candidate_lens, coverages, alpha=0.1, color='blue')
    ax3.set_xlabel('候选 MAX_LEN', fontsize=12)
    ax3.set_ylabel('未截断样本比例 (%)', fontsize=12)
    ax3.set_title('不同 MAX_LEN 下的样本覆盖率', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(output_dir / "step4_truncation_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"   📊 截断分析图已保存: step4_truncation_analysis.png")

    with open(output_dir / "step4_length_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"   📄 统计摘要已保存: step4_length_stats.json\n")
    return MAX_LEN, stats


# ============================================================
# 第 5 步：绘制累积词频覆盖率曲线，确定词表大小
# ============================================================
def step5_vocab_size(train_tokens, output_dir: Path):
    print("📈 [第5步] 累积词频覆盖率曲线...")
    all_tokens = [tok for tokens in train_tokens for tok in tokens]
    word_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    total_types = len(word_counts)

    print(f"   训练集总 Token 数: {total_tokens:,}")
    print(f"   训练集不重复词汇数: {total_types:,}")

    sorted_counts = sorted(word_counts.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts)
    coverage = cumsum / total_tokens
    vocab_sizes = np.arange(1, len(sorted_counts) + 1)

    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    threshold_info = {}
    print("\n   覆盖率阈值分析:")
    for t in thresholds:
        idx = int(np.searchsorted(cumsum, t * total_tokens))
        idx = min(idx, len(sorted_counts) - 1)
        vocab_size_needed = idx + 1
        actual_coverage = cumsum[idx] / total_tokens
        threshold_info[f"{int(t * 100)}%"] = {
            "需要词表大小": int(vocab_size_needed),
            "实际覆盖率": f"{actual_coverage * 100:.2f}%"
        }
        print(f"     {t * 100:.0f}% 覆盖率 → 词表大小: {vocab_size_needed:,} "
              f"(实际: {actual_coverage * 100:.2f}%)")

    idx_95 = int(np.searchsorted(cumsum, 0.95 * total_tokens))
    idx_95 = min(idx_95, len(sorted_counts) - 1)
    MAX_VOCAB = idx_95 + 1 + 2
    print(f"\n   🎯 选定 MAX_VOCAB = {MAX_VOCAB} (含 <PAD> 和 <UNK>)")

    # 可视化：累积覆盖率曲线（线性 + 对数双图）
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    colors = ['gray', 'orange', 'green', 'red', 'purple']

    ax1 = axes[0]
    ax1.plot(vocab_sizes, coverage * 100, 'b-', linewidth=2, label='累积覆盖率')
    ax1.fill_between(vocab_sizes, coverage * 100, alpha=0.1, color='blue')
    for t, c in zip(thresholds, colors):
        ax1.axhline(y=t * 100, color=c, linestyle=':', alpha=0.6, label=f'{t * 100:.0f}%')
    ax1.plot(MAX_VOCAB - 2, 0.95 * 100, 'ro', markersize=10, zorder=5)
    ax1.annotate(f'MAX_VOCAB≈{MAX_VOCAB}\n(95%覆盖率)',
                 xy=(MAX_VOCAB - 2, 95), xytext=(MAX_VOCAB * 2, 80),
                 fontsize=11, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax1.set_xlabel('词表大小 (Vocabulary Size)', fontsize=13)
    ax1.set_ylabel('Token 覆盖率 (%)', fontsize=13)
    ax1.set_title('累积词频覆盖率曲线（线性轴）', fontsize=15)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 102)

    ax2 = axes[1]
    ax2.plot(vocab_sizes, coverage * 100, 'b-', linewidth=2)
    ax2.set_xscale('log')
    ax2.fill_between(vocab_sizes, coverage * 100, alpha=0.1, color='blue')
    for t, c in zip(thresholds, colors):
        ax2.axhline(y=t * 100, color=c, linestyle=':', alpha=0.6)
    ax2.plot(MAX_VOCAB - 2, 0.95 * 100, 'ro', markersize=10, zorder=5)
    ax2.annotate(f'{MAX_VOCAB}', xy=(MAX_VOCAB - 2, 95), xytext=(MAX_VOCAB * 3, 82),
                 fontsize=11, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax2.set_xlabel('词表大小 (对数轴)', fontsize=13)
    ax2.set_ylabel('Token 覆盖率 (%)', fontsize=13)
    ax2.set_title('累积词频覆盖率曲线（对数轴）', fontsize=15)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(0, 102)

    plt.tight_layout()
    fig.savefig(output_dir / "step5_coverage_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 覆盖率曲线已保存: step5_coverage_curve.png")

    result = {
        "总Token数": total_tokens,
        "不重复词汇数": total_types,
        "覆盖率阈值分析": threshold_info,
        "选定的MAX_VOCAB": MAX_VOCAB,
    }
    with open(output_dir / "step5_vocab_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"   📄 词表分析已保存: step5_vocab_analysis.json\n")
    return MAX_VOCAB, word_counts


# ============================================================
# 第 6 步：词表构建与整数编码（含第二排序）
# ============================================================
def step6_build_vocab_and_encode(train_tokens, test_tokens, word_counts,
                                 max_vocab, max_len, output_dir: Path):
    print("🔢 [第6步] 词表构建与整数编码...")
    PAD_ID, UNK_ID = 0, 1

    # 词频降序 + 字典序升序（第二排序，保证可复现性）
    sorted_vocab = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    top_words = sorted_vocab[:max_vocab - 2]

    word2idx = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for idx, (word, count) in enumerate(top_words, start=2):
        word2idx[word] = idx

    print(f"   词表大小: {len(word2idx)} (含 <PAD> 和 <UNK>)")
    print(f"   前20个高频词:")
    for i, (word, count) in enumerate(top_words[:20]):
        print(f"     {i + 2:5d}: '{word}' (词频: {count:,})")

    # 整数编码
    def encode_tokens(tokens_list, w2i, unk_id):
        return [[w2i.get(tok, unk_id) for tok in tokens] for tokens in tokens_list]

    train_encoded_raw = encode_tokens(train_tokens, word2idx, UNK_ID)
    test_encoded_raw = encode_tokens(test_tokens, word2idx, UNK_ID)

    # 截断与填充
    def pad_truncate_sequences(sequences, ml, pad_id):
        result = np.zeros((len(sequences), ml), dtype=np.int32)
        for i, seq in enumerate(sequences):
            truncated = seq[:ml]
            result[i, :len(truncated)] = truncated
        return result

    train_encoded = pad_truncate_sequences(train_encoded_raw, max_len, PAD_ID)
    test_encoded = pad_truncate_sequences(test_encoded_raw, max_len, PAD_ID)

    # 编码质量统计
    train_total = sum(len(t) for t in train_tokens)
    train_unk = sum(1 for seq in train_encoded_raw for tok in seq if tok == UNK_ID)
    test_total = sum(len(t) for t in test_tokens)
    test_unk = sum(1 for seq in test_encoded_raw for tok in seq if tok == UNK_ID)

    encode_stats = {
        "词表大小": len(word2idx),
        "MAX_LEN": max_len,
        "训练集": {
            "样本数": len(train_encoded),
            "总Token数": train_total,
            "<UNK>数": train_unk,
            "<UNK>比例": f"{train_unk / train_total * 100:.2f}%",
            "编码后形状": list(train_encoded.shape),
        },
        "测试集": {
            "样本数": len(test_encoded),
            "总Token数": test_total,
            "<UNK>数": test_unk,
            "<UNK>比例": f"{test_unk / test_total * 100:.2f}%",
            "编码后形状": list(test_encoded.shape),
        },
        "第二排序说明": "词频相同时按字典序(a-z)升序排列，保证跨环境可复现性",
    }

    print(f"\n   编码质量统计:")
    print(f"     训练集 <UNK> 比例: {train_unk / train_total * 100:.2f}% "
          f"({train_unk:,}/{train_total:,})")
    print(f"     测试集 <UNK> 比例: {test_unk / test_total * 100:.2f}% "
          f"({test_unk:,}/{test_total:,})")
    print(f"     训练集编码后形状: {train_encoded.shape}")
    print(f"     测试集编码后形状: {test_encoded.shape}")

    # 编码示例
    print(f"\n   编码示例 (第1条训练样本, 前15个Token):")
    sample_tokens = train_tokens[0][:15]
    sample_ids = train_encoded[0][:15]
    for tok, tid in zip(sample_tokens, sample_ids):
        oov_tag = '(OOV)' if tid == UNK_ID and tok not in word2idx else ''
        print(f"     '{tok:15s}' → {tid} {oov_tag}")

    # 保存结果
    with open(output_dir / "word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)
    print(f"\n   📄 词表已保存: word2idx.json")

    with open(output_dir / "step6_encode_stats.json", "w", encoding="utf-8") as f:
        json.dump(encode_stats, f, ensure_ascii=False, indent=2)
    print(f"   📄 编码统计已保存: step6_encode_stats.json")

    np.save(output_dir / "train_encoded.npy", train_encoded)
    np.save(output_dir / "test_encoded.npy", test_encoded)
    print(f"   💾 训练集编码数据: train_encoded.npy {train_encoded.shape}")
    print(f"   💾 测试集编码数据: test_encoded.npy  {test_encoded.shape}")

    # 词频分布可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1 = axes[0]
    top30 = top_words[:30]
    words_30 = [w for w, c in top30]
    counts_30 = [c for w, c in top30]
    bars = ax1.barh(range(len(words_30) - 1, -1, -1), counts_30,
                    color='steelblue', edgecolor='white')
    ax1.set_yticks(range(len(words_30) - 1, -1, -1))
    ax1.set_yticklabels(words_30, fontsize=9)
    ax1.set_xlabel('词频', fontsize=12)
    ax1.set_title('Top 30 高频词', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    for bar, count in zip(bars, counts_30):
        ax1.text(bar.get_width() + max(counts_30) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{count:,}', va='center', fontsize=8)

    ax2 = axes[1]
    ranks = np.arange(1, len(sorted_vocab) + 1)
    freqs = [c for w, c in sorted_vocab]
    ax2.loglog(ranks, freqs, 'b.', markersize=1, alpha=0.5)
    ax2.axvline(x=max_vocab - 2, color='red', linestyle='--', linewidth=2,
                label=f'MAX_VOCAB-2 = {max_vocab - 2}')
    ax2.set_xlabel('词排名 (Rank, 对数轴)', fontsize=12)
    ax2.set_ylabel('词频 (Frequency, 对数轴)', fontsize=12)
    ax2.set_title("词频分布 (Zipf's Law 验证)", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(output_dir / "step6_word_frequency.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 词频分布图已保存: step6_word_frequency.png")
    print(f"\n   ✅ 第6步全部完成！\n")
    return train_encoded, test_encoded, word2idx


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 65)
    print("    ACL IMDb 数据集预处理流水线 (6步完整版)")
    print("=" * 65)
    start_time = time.time()

    # 第 0 步：加载数据（含自动下载 + 进度条）
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(DATASET_DIR)

    # 第 1 步：清洗 HTML
    train_texts, test_texts = step1_clean_html(train_texts, test_texts)

    # 第 2 步：小写化
    train_texts, test_texts = step2_lowercase(train_texts, test_texts)

    # 第 3 步：分词
    train_tokens, test_tokens = step3_tokenize(train_texts, test_texts)

    # 第 4 步：长度分析与截断/填充
    MAX_LEN, length_stats = step4_length_analysis(train_tokens, OUTPUT_DIR)

    # 第 5 步：覆盖率曲线确定词表大小
    MAX_VOCAB, word_counts = step5_vocab_size(train_tokens, OUTPUT_DIR)

    # 第 6 步：词表构建与整数编码
    train_encoded, test_encoded, word2idx = step6_build_vocab_and_encode(
        train_tokens, test_tokens, word_counts, MAX_VOCAB, MAX_LEN, OUTPUT_DIR
    )

    # 保存标签
    np.save(OUTPUT_DIR / "train_labels.npy", np.array(train_labels, dtype=np.int32))
    np.save(OUTPUT_DIR / "test_labels.npy", np.array(test_labels, dtype=np.int32))

    # 生成最终报告
    elapsed = time.time() - start_time
    report = {
        "流水线总结": {
            "MAX_LEN": MAX_LEN,
            "MAX_VOCAB": MAX_VOCAB,
            "词表实际大小": len(word2idx),
            "训练集编码形状": list(train_encoded.shape),
            "测试集编码形状": list(test_encoded.shape),
            "总耗时(秒)": round(elapsed, 1),
        },
        "输出文件清单": [f.name for f in sorted(OUTPUT_DIR.iterdir())],
        "数据隔离声明": "所有统计量（长度分布、词频、覆盖率）均严格仅基于训练集计算，测试集在预处理阶段完全不可见。",
        "可复现性声明": "词表构建采用字典序作为第二排序关键字，确保跨环境可复现。",
    }
    with open(OUTPUT_DIR / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 完成
    print("=" * 65)
    print("  🎉 全部 6 步预处理完成！")
    print(f"  ⏱️  总耗时: {elapsed:.1f} 秒")
    print(f"  📁 输出目录: {OUTPUT_DIR}")
    print(f"  📄 输出文件:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        if size_kb > 1024:
            print(f"     {f.name:40s} ({size_kb / 1024:.1f} MB)")
        else:
            print(f"     {f.name:40s} ({size_kb:.1f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()