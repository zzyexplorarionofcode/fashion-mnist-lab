"""线性 Softmax 基线：训练、画训练/验证曲线、在测试集上做完整评价。

产出：
    outputs/figures/linear_curves.png          训练与验证曲线
    outputs/figures/linear_cm.png              混淆矩阵（计数）
    outputs/figures/linear_cm_norm.png         混淆矩阵（按行归一化）
    outputs/figures/linear_recall.png          逐类召回率柱状图
    outputs/figures/linear_errors.png          典型错误样本
    outputs/per_class_recall_linear.csv        逐类召回率表
    outputs/linear_summary.json                测试准确率等汇总
"""

import json
import os

import numpy as np
import torch

import config
from data_utils import get_dataloaders, reset_loader_seed, set_seed
from evaluate import (confusion_matrix_np, per_class_recall, plot_confusion_matrix,
                      plot_curves, plot_error_cases, plot_per_class_recall,
                      predict_model, save_history_json, write_csv)
from models import LinearSoftmax, count_parameters
from trainer import evaluate, train_model


def main():
    """入口：完整跑一遍线性 Softmax 基线实验。

    参数: 无
    返回: 无
    """
    config.setup_console()
    config.ensure_dirs()
    set_seed(config.SEED)

    print("=" * 60)
    print("线性 Softmax 基线")
    print(config.model_config_text())
    print("=" * 60)

    data = get_dataloaders()

    # 建模前必须重置种子：load_arrays 遍历 DataLoader 时会消耗全局 torch 随机源，
    # 不重置的话这里的初始化状态和 train_compare.py 里的不同，
    # 同一个"线性模型"在两个脚本里会得到不同的初始权重和不同的结果。
    set_seed(config.SEED)
    reset_loader_seed(data.train_loader, config.SEED)

    model = LinearSoftmax()
    print(f"参数量: {count_parameters(model)}")

    history, diverged = train_model(model, data.train_loader, data.val_loader,
                                    log_prefix="[linear] ")
    if diverged:
        print("警告: 线性基线训练过程中损失发散")

    # ---------- 曲线 ----------
    hist_name = "Linear Softmax"
    plot_curves({hist_name: history}, os.path.join(config.FIG_DIR, "linear_curves_loss.png"),
                "Linear Softmax", metric="loss")
    plot_curves({hist_name: history}, os.path.join(config.FIG_DIR, "linear_curves_acc.png"),
                "Linear Softmax", metric="acc")
    save_history_json(os.path.join(config.OUTPUT_DIR, "history_linear.json"),
                      {hist_name: history})

    # ---------- 测试集评价 ----------
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, data.test_loader, criterion)
    print(f"\n测试集: loss={test_loss:.4f} acc={test_acc:.4f}")

    y_true, y_pred, probs = predict_model(model, data.test_loader)
    cm = confusion_matrix_np(y_true, y_pred)
    recall = per_class_recall(cm)

    plot_confusion_matrix(cm, os.path.join(config.FIG_DIR, "linear_cm.png"),
                          f"Linear Softmax Confusion Matrix (test acc={test_acc:.4f})")
    plot_confusion_matrix(cm, os.path.join(config.FIG_DIR, "linear_cm_norm.png"),
                          "Linear Softmax Confusion Matrix (row-normalized)",
                          normalize=True)
    plot_per_class_recall(recall, os.path.join(config.FIG_DIR, "linear_recall.png"),
                          f"Linear Softmax Per-class Recall (test acc={test_acc:.4f})")

    shown = plot_error_cases(data.X_test_raw, y_true, y_pred, probs,
                             os.path.join(config.FIG_DIR, "linear_errors.png"))

    # ---------- 保存数值结果 ----------
    write_csv(os.path.join(config.OUTPUT_DIR, "per_class_recall_linear.csv"),
              ["class", "recall", "correct", "total"],
              [[config.CLASS_NAMES[i], f"{recall[i]:.4f}",
                int(cm[i, i]), int(cm[i].sum())] for i in range(config.NUM_CLASSES)])

    print("\n逐类召回率:")
    for i in range(config.NUM_CLASSES):
        print(f"  {config.CLASS_NAMES[i]:<12} {recall[i]:.4f} "
              f"({cm[i, i]}/{cm[i].sum()})")
    print(f"宏平均召回率 = {recall.mean():.4f}")

    print("\n最严重的混淆组合 (真实 -> 预测):")
    for name_t, name_p, cnt in shown:
        print(f"  {name_t} -> {name_p}: {cnt}")

    summary = {
        "model": "LinearSoftmax",
        "config": config.model_config_text(),
        "num_params": count_parameters(model),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "macro_recall": float(recall.mean()),
        "per_class_recall": {config.CLASS_NAMES[i]: float(recall[i])
                             for i in range(config.NUM_CLASSES)},
        "top_confusions": [{"true": t, "pred": p, "count": c} for t, p, c in shown],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_acc": float(np.max(history["val_acc"])),
        "total_train_time_sec": float(np.sum(history["epoch_time"])),
    }
    with open(os.path.join(config.OUTPUT_DIR, "linear_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[线性基线] 结果已保存到 {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
