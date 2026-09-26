"""评价与可视化工具：混淆矩阵、逐类召回率、训练曲线、典型错误样本。

线性基线和三模型比较共用这里的函数，保证两边画出的图口径一致。
所有图都保存到 outputs/figures/ 下，标签统一用英文，规避 matplotlib 中文字体缺失问题。
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")   # 无界面后端，批量出图不需要弹窗
import matplotlib.pyplot as plt
import numpy as np
import torch

import config


def predict_model(model, loader):
    """在整个数据集上跑一遍前向，收集预测结果。

    参数: model - 训练好的模型；loader - 数据加载器（必须 shuffle=False）
    返回: (y_true, y_pred, probs)
          y_true/y_pred 为形状 (N,) 的 int64 数组，probs 为形状 (N, 10) 的概率
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs)
            prob = torch.softmax(logits, dim=1)     # 转成概率，便于展示置信度
            all_true.append(labels.numpy())
            all_pred.append(logits.argmax(dim=1).numpy())
            all_prob.append(prob.numpy())
    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_prob))


def confusion_matrix_np(y_true, y_pred, num_classes=config.NUM_CLASSES):
    """手算混淆矩阵，避免额外依赖 sklearn。

    参数: y_true - 真实标签；y_pred - 预测标签；num_classes - 类别数
    返回: 形状 (C, C) 的 int64 矩阵，cm[i, j] = 真实为 i 但预测为 j 的样本数
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # bincount 把 (真实*C + 预测) 的一维编码直接统计成频次，再 reshape 回二维
    idx = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    cm += counts.reshape(num_classes, num_classes)
    return cm


def per_class_recall(cm):
    """从混淆矩阵算逐类召回率。

    召回率 = 该类被正确预测的样本数 / 该类真实样本总数。
    若某类在测试集中一个样本都没有，分母为 0，约定返回 0 避免除零。

    参数: cm - 混淆矩阵
    返回: 形状 (C,) 的 float 数组
    """
    total = cm.sum(axis=1)
    correct = np.diag(cm)
    return np.where(total > 0, correct / np.maximum(total, 1), 0.0)


def plot_curves(histories, save_path, title, metric="loss"):
    """把多个模型的训练/验证曲线画在一张图上，便于横向比较。

    参数: histories - dict，{模型名: history字典}
          save_path - 图片保存路径
          title - 图标题
          metric - 'loss' 画损失曲线，'acc' 画准确率曲线
    返回: 无
    """
    train_key = "train_loss" if metric == "loss" else "train_acc"
    val_key = "val_loss" if metric == "loss" else "val_acc"
    ylabel = "Loss" if metric == "loss" else "Accuracy"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, hist in histories.items():
        epochs = range(1, len(hist[train_key]) + 1)
        axes[0].plot(epochs, hist[train_key], label=name)
        axes[1].plot(epochs, hist[val_key], label=name)

    axes[0].set_title(f"{title} - Train")
    axes[1].set_title(f"{title} - Validation")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, save_path, title, normalize=False):
    """画混淆矩阵热图。

    参数: cm - 混淆矩阵；save_path - 保存路径；title - 图标题
          normalize - True 时按行归一化，看的是"每个真实类被分到各预测类的比例"
    返回: 无
    """
    mat = cm.astype(np.float64)
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = mat / np.maximum(row_sum, 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # 每格写上数值：归一化时保留两位小数，否则写整数计数
    thresh = mat.max() / 2.0
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            txt = f"{mat[i, j]:.2f}" if normalize else f"{int(mat[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if mat[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_class_recall(recall, save_path, title):
    """画逐类召回率柱状图。

    参数: recall - 形状 (C,) 的召回率数组；save_path - 保存路径；title - 图标题
    返回: 无
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(config.NUM_CLASSES), recall, color="#4C78A8")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for b, r in zip(bars, recall):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.01, f"{r:.2f}",
                ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_bar(names, values, save_path, title, ylabel, ylim=None, fmt="{:.4f}"):
    """画一组柱状图，用于横向对比不同模型/不同配置的单个指标。

    参数: names - 横轴标签列表；values - 对应数值列表；save_path - 保存路径
          title - 图标题；ylabel - 纵轴名称；ylim - 纵轴范围 (low, high)，None 表示自动
          fmt - 柱顶数值的格式串
    返回: 无
    """
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4.5))
    colors = ["#9AA0A6", "#F2A93B", "#4C78A8", "#54A24B", "#E45756"]
    bars = ax.bar(names, values, color=[colors[i % len(colors)] for i in range(len(names))])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def top_confusion_pairs(cm, k=4):
    """找出混淆最严重的若干"真实类 -> 预测类"组合。

    只看非对角元素，按样本数从多到少排序。

    参数: cm - 混淆矩阵；k - 返回多少个组合
    返回: 列表，每项 (真实类下标, 预测类下标, 样本数)
    """
    pairs = []
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda t: -t[2])
    return pairs[:k]


def plot_error_cases(X_test_raw, y_true, y_pred, probs, save_path, k_pairs=4,
                     per_pair=6):
    """画典型错误样本：按"真实类 -> 预测类"分组展示被分错的图片。

    每行是一个混淆组合，每列是一张错分图片，标题给出真实标签、预测标签和预测置信度。

    参数: X_test_raw - 未归一化的测试集张量 (N, 784)，用原始像素方便看清内容
          y_true / y_pred - 测试集上的真实与预测标签
          probs - 预测概率，用于显示置信度
          save_path - 保存路径
          k_pairs - 展示多少个混淆组合
          per_pair - 每个组合最多展示几张图
    返回: 展示的组合列表，每项 (真实类名, 预测类名, 样本数)
    """
    cm = confusion_matrix_np(y_true, y_pred)
    pairs = top_confusion_pairs(cm, k_pairs)
    if not pairs:
        return []

    # 画布刻意取小、字号取大：这张图要缩到报告里约 12cm 宽，若按 12 英寸画布出图，
    # 6.5pt 的标题缩到纸面上只剩 2.7pt，根本看不清。缩小画布保持字号相对更大。
    fig, axes = plt.subplots(len(pairs), per_pair,
                             figsize=(1.6 * per_pair, 2.4 * len(pairs)))
    # 只有一行/一列时 subplots 返回的不是二维数组，统一成二维方便索引
    axes = np.atleast_2d(axes)

    shown = []
    for row, (ti, pi, cnt) in enumerate(pairs):
        # 找出所有"真实为 ti 且预测为 pi"的样本下标
        idxs = np.where((y_true == ti) & (y_pred == pi))[0][:per_pair]
        for col in range(per_pair):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(idxs):
                continue
            n = int(idxs[col])
            img = X_test_raw[n].view(28, 28).numpy()
            ax.imshow(img, cmap="gray")
            # 类名不截断：T-shirt/top 与 Shirt 是两个不同的类，截断后容易看混
            ax.set_title(f"T:{config.CLASS_NAMES[ti]}\n"
                         f"P:{config.CLASS_NAMES[pi]}\n"
                         f"conf={probs[n, pi]:.2f}", fontsize=11)
        shown.append((config.CLASS_NAMES[ti], config.CLASS_NAMES[pi], cnt))

    fig.suptitle("Typical misclassified test images (T=true, P=predicted)",
                 fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return shown


def write_csv(path, header, rows):
    """把表格写成 CSV（utf-8-sig，Excel 打开不乱码）。

    参数: path - 保存路径；header - 表头列表；rows - 每行数据的列表
    返回: 无
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_history_json(path, histories):
    """把训练曲线数据存成 json，方便报告里引用具体数值。

    参数: path - 保存路径；histories - {模型名: history字典}
    返回: 无
    """
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)
