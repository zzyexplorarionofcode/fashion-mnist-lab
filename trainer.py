"""统一的训练与评估循环。

三个模型（线性 Softmax、双层无激活、双层 + ReLU）全部走这一个函数，
"受控比较"里的"其余训练条件尽量保持一致"就是靠这个共享实现来保证的：
优化器、损失函数、学习率、批大小、epoch 数、评估时机都写死在同一份代码里。
"""

import time

import torch
import torch.nn as nn

import config


def evaluate(model, loader, criterion):
    """在给定数据集上算平均损失和准确率（不更新参数）。

    参数: model - 待评估模型；loader - 数据加载器；criterion - 损失函数
    返回: (平均损失, 准确率)，均为 float
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def build_optimizer(model, lr=config.LR, momentum=config.MOMENTUM,
                    weight_decay=config.WEIGHT_DECAY):
    """构建优化器。所有实验统一用带动量的 SGD，只有学习率在诊断实验里被改动。

    参数: model - 待优化模型；lr - 学习率；momentum - 动量系数
          weight_decay - L2 权重衰减
    返回: torch.optim.Optimizer
    """
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                           weight_decay=weight_decay)


def train_model(model, train_loader, val_loader, epochs=config.EPOCHS,
                lr=config.LR, momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY, log_prefix="",
                verbose=True):
    """训练模型并逐 epoch 记录训练/验证曲线。

    每个 epoch 结束后在验证集上评估一次，用于画曲线和后续做 Early Stopping。
    若训练损失出现 NaN/Inf（学习率过大时会发生），立刻中断并记录，
    避免后续画出无意义的曲线。

    参数: model - 待训练模型；train_loader - 训练集；val_loader - 验证集
          epochs - 训练轮数；lr - 学习率；momentum - 动量
          weight_decay - L2 权重衰减；log_prefix - 日志前缀
          verbose - 是否打印每个 epoch 的信息
    返回: (history, diverged)
          history 为字典，含 train_loss / train_acc / val_loss / val_acc / epoch_time
          diverged 为 bool，表示是否因为损失发散而提前中断
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr, momentum, weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_time": []}
    diverged = False

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = run_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 损失发散时立刻停下：再往下训练参数已经是 NaN，曲线没有意义
        if not torch.isfinite(torch.tensor(train_loss)):
            history["train_loss"].append(float("nan"))
            history["train_acc"].append(float("nan"))
            history["val_loss"].append(float("nan"))
            history["val_acc"].append(float("nan"))
            history["epoch_time"].append(time.time() - t0)
            diverged = True
            if verbose:
                print(f"{log_prefix}epoch {epoch:2d} | 训练损失发散(NaN)，中断")
            break

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(time.time() - t0)

        if verbose:
            print(f"{log_prefix}epoch {epoch:2d} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"{history['epoch_time'][-1]:.1f}s")

    return history, diverged
