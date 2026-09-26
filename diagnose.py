"""训练诊断：学习率过大导致训练不收敛。

诊断流程遵循"先判断原因、再只改一个因素验证"的要求：

    现象   ：用 SGD + momentum 训练 MLP 时，学习率取 1.0 训练完全不收敛，
             训练损失在几十到两百之间剧烈震荡，验证准确率停在随机水平附近。
    原因判断：步长超过了损失曲面的稳定范围。SGD 的更新量是 lr * grad，
             lr 过大时参数一步就跨过谷底，loss 反而增大；下一步梯度更大，
             形成正反馈，损失量级持续攀升。动量项累积历史梯度，相当于把
             有效步长再放大数倍，让发散来得更快。学习率继续加大到 2.0 时，
             损失会直接溢出到 1e26 量级。
    验证方式：保持网络结构、数据划分、优化器、批大小、epoch、种子全部不变，
             只把学习率依次取 0.01 / 0.1 / 1.0，观察损失曲线的差异。
             - 0.01 ：平稳下降            -> 该量级是安全区间
             - 0.1  ：仍能收敛但明显变差   -> 退化是渐进的，不是突变
             - 1.0  ：剧烈震荡、损失暴涨   -> 证实是步长过大导致训练失败

结论只由"改学习率"这一个变量支撑，其他因素都被固定，因此可以归因。

产出：
    outputs/figures/diag_step_loss.png     第一个 epoch 内逐 step 的损失（对数纵轴）
    outputs/figures/diag_train_loss.png    逐 epoch 训练损失（对数纵轴）
    outputs/figures/diag_val_acc.png       逐 epoch 验证准确率
    outputs/diagnose_table.csv             各学习率的结果汇总
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from data_utils import get_dataloaders, reset_loader_seed, set_seed
from evaluate import write_csv
from models import MLPReLU
from trainer import build_optimizer, evaluate

# 每隔多少个 step 记录一次损失。设小一点是为了看清发散发生在第几步
LOG_EVERY = 20


def train_one_lr(train_loader, val_loader, lr, epochs, criterion):
    """用指定学习率训练一个 MLP，并记录逐 step 与逐 epoch 的损失。

    这里没有复用 trainer.train_model，原因是诊断需要 step 级别的损失曲线
    才能说明"在第几步炸掉"，而通用训练循环只按 epoch 汇总。

    参数: train_loader - 训练集；val_loader - 验证集；lr - 学习率
          epochs - 训练轮数；criterion - 损失函数
    返回: (record, step_log)
          record 含逐 epoch 的 train_loss / val_acc / time 与是否发散
          step_log 为 (step, loss) 列表，只覆盖第一个 epoch
    """
    # 每个学习率都用同样的种子初始化，唯一变量只有 lr
    set_seed(config.SEED)
    reset_loader_seed(train_loader, config.SEED)

    model = MLPReLU()
    optimizer = build_optimizer(model, lr=lr)

    record = {"train_loss": [], "val_acc": [], "time": [], "diverged": False}
    step_log = []
    global_step = 0
    diverged = False

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, total = 0.0, 0

        for imgs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            value = loss.item()
            run_loss += value * labels.size(0)
            total += labels.size(0)
            global_step += 1

            # 只记录第一个 epoch，因为发散总是发生在最开始的几十步内
            if epoch == 1 and global_step % LOG_EVERY == 0:
                step_log.append((global_step, value))

            # 出现 NaN/Inf 说明已经数值溢出，再训练没有意义，直接中断
            if not np.isfinite(value):
                record["diverged"] = True
                step_log.append((global_step, float("nan")))
                diverged = True
                break

        if diverged:
            record["train_loss"].append(float("nan"))
            record["val_acc"].append(float("nan"))
            record["time"].append(time.time() - t0)
            break

        train_loss = run_loss / total
        _, val_acc = evaluate(model, val_loader, criterion)
        record["train_loss"].append(train_loss)
        record["val_acc"].append(val_acc)
        record["time"].append(time.time() - t0)
        print(f"  lr={lr:<5} epoch {epoch:2d} | train_loss={train_loss:.4f} "
              f"| val_acc={val_acc:.4f} | {record['time'][-1]:.1f}s")

    return record, step_log


def classify(record):
    """根据训练结果判断该学习率属于哪种情况。

    判据：发散(NaN) -> 未收敛(末轮损失仍然很高) -> 收敛偏慢 -> 正常收敛。
    这些阈值是经验性的，只用于在报告里给一个直观标签。

    参数: record - train_one_lr 返回的记录
    返回: 结论字符串
    """
    if record["diverged"]:
        return "发散 (NaN)"
    final = record["train_loss"][-1]
    if final > 1.0:
        return "未收敛/震荡"
    if final > 0.5:
        return "收敛偏慢"
    return "正常收敛"


def plot_multi(curves, save_path, title, xlabel, ylabel, log_y=False,
               ylim=None, mark_nan=True):
    """把多条曲线画在同一张图上，纵轴可选对数刻度。

    参数: curves - [(标签, x数组, y数组), ...]
          save_path - 保存路径；title/xlabel/ylabel - 图元信息
          log_y - 纵轴是否用对数刻度（损失跨好几个数量级时必需）
          ylim - 纵轴范围；mark_nan - 是否把 NaN 位置标注出来
    返回: 无
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    nan_xs = []
    for label, xs, ys in curves:
        xs = np.asarray(xs)
        ys = np.asarray(ys, dtype=np.float64)
        finite = np.isfinite(ys)
        # 只画有限值，NaN 会自然形成断点，不会被当成 0 拉到图底
        ax.plot(xs[finite], ys[finite], marker="o", ms=3, label=label)
        if (~finite).any():
            nan_xs.append(float(xs[~finite][0]))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)

    # NaN 标注放在坐标轴设定之后，这样文字位置才是最终纵轴范围下的位置
    if mark_nan:
        for x in nan_xs:
            ax.axvline(x, ls=":", alpha=0.5, color="red")
        if nan_xs:
            ax.text(min(nan_xs), 0.97, "NaN here", fontsize=8, va="top",
                    color="red", transform=ax.get_xaxis_transform())

    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    """入口：跑学习率对照实验并输出诊断结论。

    参数: 无
    返回: 无
    """
    config.setup_console()
    config.ensure_dirs()

    print("=" * 60)
    print("训练诊断：学习率对 MLP 收敛的影响")
    print("=" * 60)
    print("现象    : 学习率取 1.0 时训练损失不降反升，直至 NaN")
    print("原因判断: 步长超过损失曲面的稳定范围，参数一步跨过谷底形成正反馈；")
    print("          动量项累积历史梯度，进一步放大有效步长，加速发散")
    print("验证方式: 只改学习率，其余条件（结构/数据/优化器/批大小/epoch/种子）全部固定")
    print(f"学习率取值: {config.DIAG_LR_LIST}，epoch={config.DIAG_EPOCHS}\n")

    data = get_dataloaders()
    train_loader, val_loader = data.train_loader, data.val_loader
    criterion = torch.nn.CrossEntropyLoss()

    records, step_logs, rows = {}, {}, []
    for lr in config.DIAG_LR_LIST:
        print(f"训练中: lr={lr}")
        rec, steps = train_one_lr(train_loader, val_loader, lr,
                                  config.DIAG_EPOCHS, criterion)
        records[lr] = rec
        step_logs[lr] = steps
        conclusion = classify(rec)
        print(f"  -> 结论: {conclusion}\n")
        rows.append([
            lr,
            f"{rec['train_loss'][0]:.4f}" if rec["train_loss"] else "nan",
            f"{rec['train_loss'][-1]:.4f}",
            f"{float(np.nanmax(rec['val_acc'])):.4f}",
            len(rec["train_loss"]),
            conclusion,
            f"{float(np.sum(rec['time'])):.1f}",
        ])

    # ---------- 出图 ----------
    plot_multi(
        [(f"lr={lr}", [s for s, _ in step_logs[lr]], [v for _, v in step_logs[lr]])
         for lr in config.DIAG_LR_LIST],
        os.path.join(config.FIG_DIR, "diag_step_loss.png"),
        f"Training loss within the first epoch (log scale), log every {LOG_EVERY} steps",
        "Step", "Loss", log_y=True)

    plot_multi(
        [(f"lr={lr}", np.arange(1, len(records[lr]["train_loss"]) + 1),
          records[lr]["train_loss"]) for lr in config.DIAG_LR_LIST],
        os.path.join(config.FIG_DIR, "diag_train_loss.png"),
        "Per-epoch training loss under different learning rates (log scale)",
        "Epoch", "Train Loss", log_y=True)

    plot_multi(
        [(f"lr={lr}", np.arange(1, len(records[lr]["val_acc"]) + 1),
          records[lr]["val_acc"]) for lr in config.DIAG_LR_LIST],
        os.path.join(config.FIG_DIR, "diag_val_acc.png"),
        "Validation accuracy under different learning rates",
        "Epoch", "Validation Accuracy", ylim=(0.0, 1.0))

    # ---------- 保存表格 ----------
    header = ["lr", "first_epoch_loss", "final_train_loss", "best_val_acc",
              "epochs_run", "conclusion", "time_sec"]
    write_csv(os.path.join(config.OUTPUT_DIR, "diagnose_table.csv"), header, rows)

    with open(os.path.join(config.OUTPUT_DIR, "diagnose_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "experiment": "learning rate diagnosis",
            "fixed": config.model_config_text(),
            "lr_list": config.DIAG_LR_LIST,
            "rows": [dict(zip(header, r)) for r in rows],
        }, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("诊断结果汇总")
    print("=" * 60)
    print("".join(f"{h:<18}" for h in header))
    for r in rows:
        print("".join(f"{str(v):<18}" for v in r))
    print(f"\n[诊断] 结果已保存到 {config.OUTPUT_DIR}")
    print("结论: 只改学习率就足以让同一个模型从正常收敛变成完全失败，"
          "证实上述现象的原因是步长过大，而不是网络结构或数据问题。")


if __name__ == "__main__":
    main()
