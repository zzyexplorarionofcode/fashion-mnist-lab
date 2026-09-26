"""受控比较：线性 Softmax / 双层无激活 / 双层 + ReLU。

受控的含义体现在以下几点，三者完全一致：
    - 同一份训练/验证划分（读同一个 split_indices.npz）
    - 同一个数据打乱顺序（每个模型训练前重置 loader 的 generator）
    - 同一个优化器与超参（SGD + momentum，同一学习率、批大小、epoch 数）
    - 同一套归一化参数（训练集均值方差）
    - 同一个随机种子
唯一变量是网络结构本身。模型②和③的隐藏层宽度也相同（HIDDEN_DIM）。

产出：
    outputs/compare_table.csv            三模型性能表
    outputs/figures/compare_curves_*.png 三模型训练/验证曲线
    outputs/figures/compare_test_acc.png 测试准确率对比柱状图
    outputs/figures/compare_cm_*.png     每个模型的混淆矩阵
    outputs/figures/compare_recall_*.png 每个模型的逐类召回率
"""

import json
import os

import numpy as np
import torch

import config
from data_utils import get_dataloaders, reset_loader_seed, set_seed
from evaluate import (confusion_matrix_np, per_class_recall, plot_bar,
                      plot_confusion_matrix, plot_curves, plot_per_class_recall,
                      predict_model, save_history_json, write_csv)
from models import build_model, count_parameters
from trainer import evaluate, train_model

# 三个模型的显示名与构建名，顺序即报告中的①②③
MODELS = [
    ("1-Linear", "linear"),
    ("2-TwoLinear-NoAct", "noact"),
    ("3-MLP-ReLU", "relu"),
]


def main():
    """入口：依次训练三个模型并汇总对比结果。

    参数: 无
    返回: 无
    """
    config.setup_console()
    config.ensure_dirs()
    set_seed(config.SEED)

    print("=" * 60)
    print("受控比较：线性 / 双层无激活 / 双层+ReLU")
    print(config.model_config_text())
    print("=" * 60)

    data = get_dataloaders()
    train_loader, val_loader, test_loader = (data.train_loader, data.val_loader,
                                             data.test_loader)
    criterion = torch.nn.CrossEntropyLoss()

    histories, rows, test_accs = {}, [], []

    for display, name in MODELS:
        print(f"\n{'-' * 60}\n训练模型: {display}  ({name})\n{'-' * 60}")

        # 每个模型开始前重置种子和打乱顺序，保证三者起点与数据顺序一致
        set_seed(config.SEED)
        reset_loader_seed(train_loader, config.SEED)

        model = build_model(name)
        n_params = count_parameters(model)
        print(f"参数量: {n_params}")

        history, diverged = train_model(model, train_loader, val_loader,
                                        log_prefix=f"[{display}] ")
        histories[display] = history

        test_loss, test_acc = evaluate(model, test_loader, criterion)
        test_accs.append(test_acc)
        print(f"{display} 测试集: loss={test_loss:.4f} acc={test_acc:.4f}")

        # 该模型的混淆矩阵与逐类召回率
        y_true, y_pred, _ = predict_model(model, test_loader)
        cm = confusion_matrix_np(y_true, y_pred)
        recall = per_class_recall(cm)
        plot_confusion_matrix(
            cm, os.path.join(config.FIG_DIR, f"compare_cm_{name}.png"),
            f"{display} Confusion Matrix (test acc={test_acc:.4f})")
        plot_per_class_recall(
            recall, os.path.join(config.FIG_DIR, f"compare_recall_{name}.png"),
            f"{display} Per-class Recall (test acc={test_acc:.4f})")

        rows.append([
            display, n_params,
            f"{history['train_loss'][-1]:.4f}",
            f"{history['val_loss'][-1]:.4f}",
            f"{float(np.max(history['val_acc'])):.4f}",
            f"{test_loss:.4f}",
            f"{test_acc:.4f}",
            f"{float(np.sum(history['epoch_time'])):.1f}",
            "是" if diverged else "否",
        ])

    # ---------- 汇总输出 ----------
    plot_curves(histories, os.path.join(config.FIG_DIR, "compare_curves_loss.png"),
                "Three Models", metric="loss")
    plot_curves(histories, os.path.join(config.FIG_DIR, "compare_curves_acc.png"),
                "Three Models", metric="acc")
    save_history_json(os.path.join(config.OUTPUT_DIR, "history_compare.json"), histories)

    write_csv(
        os.path.join(config.OUTPUT_DIR, "compare_table.csv"),
        ["model", "num_params", "final_train_loss", "final_val_loss",
         "best_val_acc", "test_loss", "test_acc", "train_time_sec", "diverged"],
        rows)

    # 测试准确率对比柱状图
    plot_bar([m[0] for m in MODELS], test_accs,
             os.path.join(config.FIG_DIR, "compare_test_acc.png"),
             f"Test Accuracy Comparison  (hidden={config.HIDDEN_DIM})",
             "Test Accuracy", ylim=(0.80, 0.95))

    print("\n" + "=" * 60)
    print("三模型性能表")
    print("=" * 60)
    header = ["模型", "参数量", "末轮训练loss", "末轮验证loss", "最优验证acc",
              "测试loss", "测试acc", "耗时(s)", "发散"]
    print("".join(f"{h:<18}" for h in header))
    for r in rows:
        print("".join(f"{str(v):<18}" for v in r))

    # 用测试准确率差值和参数量倍数，给出可控的结论性提示
    print("\n关键对比:")
    print(f"  ②无激活 vs ①线性: 参数量 {rows[1][1] / rows[0][1]:.1f} 倍，"
          f"测试准确率 {float(rows[1][6]) - float(rows[0][6]):+.4f}")
    print(f"  ③有ReLU vs ②无激活: 参数量相同，"
          f"测试准确率 {float(rows[2][6]) - float(rows[1][6]):+.4f}")

    with open(os.path.join(config.OUTPUT_DIR, "compare_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "config": config.model_config_text(),
            "table": [dict(zip(["model", "num_params", "final_train_loss",
                                "final_val_loss", "best_val_acc", "test_loss",
                                "test_acc", "train_time_sec", "diverged"], r))
                      for r in rows],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[受控比较] 结果已保存到 {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
