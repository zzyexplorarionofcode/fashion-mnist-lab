"""数据加载、固定划分与数据管道。

两个关键设计：

1) 固定划分：训练/验证的划分索引只生成一次并落盘到 outputs/split_indices.npz，
   之后线性基线、三模型比较、诊断实验全部读同一份索引。三个模型的训练集和
   验证集完全相同，"比较是否受控"才有前提。

2) 一次性堆叠成张量：torchvision 的数据集每取一个样本都要把 PIL 图转一次张量，
   在 50000 样本 / 每 epoch 391 个 batch 的训练下，这个开销比前向反向还大。
   这里在启动时把整个数据集读进内存（float32，约 220MB），训练时只做切片和
   归一化，单 epoch 耗时能降一个数量级。
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import config


@dataclass
class DataBundle:
    """打包数据集、加载器与归一化参数，避免各脚本里出现超长元组解包。

    字段说明:
        train_loader / val_loader / test_loader - 三个 DataLoader
        X_train_raw / y_train - 完整官方训练集（60000 张，像素 0~1，未归一化）
        X_test_raw / y_test   - 官方测试集（10000 张，未归一化）
        train_idx / val_idx   - 固定划分的索引
        mean / std            - 训练集统计出的归一化参数
    """
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    X_train_raw: torch.Tensor
    y_train: torch.Tensor
    X_test_raw: torch.Tensor
    y_test: torch.Tensor
    train_idx: np.ndarray
    val_idx: np.ndarray
    mean: float
    std: float


def set_seed(seed=config.SEED):
    """设置所有随机源的种子，保证实验可复现。

    参数: seed - 随机种子
    返回: 无
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU 环境下用不到 cudnn，保留是为了换到 GPU 机器上也能复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _stack_dataset(dataset):
    """把一个 torchvision 数据集整体堆叠成张量。

    参数: dataset - torchvision 数据集（transform 为 ToTensor）
    返回: (X, y)，X 形状 (N, 784) 的 float32，y 形状 (N,) 的 int64
    """
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)
    xs, ys = [], []
    for imgs, labels in loader:
        xs.append(imgs.view(imgs.size(0), -1))
        ys.append(labels)
    return torch.cat(xs), torch.cat(ys)


def load_arrays():
    """加载官方 Fashion-MNIST 并整体转成张量（只做 ToTensor，不归一化、不增强）。

    download=True 只在本地缺文件时才真正联网。

    参数: 无
    返回: (X_train, y_train, X_test, y_test)，像素取值范围 0~1
    """
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(
        root=config.DATA_DIR, train=True, download=True, transform=tf
    )
    test_set = datasets.FashionMNIST(
        root=config.DATA_DIR, train=False, download=True, transform=tf
    )
    X_train, y_train = _stack_dataset(train_set)
    X_test, y_test = _stack_dataset(test_set)
    return X_train, y_train, X_test, y_test


def get_split_indices(total_size):
    """取得固定的训练/验证划分索引，首次运行时随机生成并落盘。

    参数: total_size - 官方训练集样本总数
    返回: (train_idx, val_idx)，两个 int64 的 NumPy 数组
    """
    if os.path.exists(config.SPLIT_FILE):
        data = np.load(config.SPLIT_FILE)
        return data["train_idx"], data["val_idx"]

    # 用独立的 Generator，避免划分结果受其他随机调用影响
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(total_size)
    # 排序只为索引好看，不改变划分本身
    train_idx = np.sort(perm[: config.TRAIN_SIZE])
    val_idx = np.sort(perm[config.TRAIN_SIZE : config.TRAIN_SIZE + config.VAL_SIZE])

    config.ensure_dirs()
    np.savez(config.SPLIT_FILE, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


def get_dataloaders(normalize=True):
    """构建训练集、验证集、测试集三个 DataLoader。

    归一化参数只用训练集的均值和标准差，且在三个模型之间共用，
    避免"每个模型各归一化一套"这种隐性变量。

    参数: normalize - 是否按训练集统计量做标准化
    返回: DataBundle
    """
    set_seed(config.SEED)
    X_train, y_train, X_test, y_test = load_arrays()
    train_idx, val_idx = get_split_indices(X_train.size(0))

    if normalize:
        mean = float(X_train[train_idx].mean().item())
        std = float(X_train[train_idx].std().item())
    else:
        mean, std = 0.0, 1.0

    # 只对需要的切片做归一化，避免复制整份数据
    train_ds = TensorDataset((X_train[train_idx] - mean) / std, y_train[train_idx])
    val_ds = TensorDataset((X_train[val_idx] - mean) / std, y_train[val_idx])
    test_ds = TensorDataset((X_test - mean) / std, y_test)

    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    return DataBundle(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        X_train_raw=X_train, y_train=y_train,
        X_test_raw=X_test, y_test=y_test,
        train_idx=train_idx, val_idx=val_idx, mean=mean, std=std,
    )


def reset_loader_seed(loader, seed=config.SEED):
    """重置 DataLoader 内部随机数生成器的状态。

    受控比较的关键一步：打乱顺序由 loader 里的 generator 决定，
    而 generator 的推进状态会在多个模型之间连续累积。若不重置，
    模型②看到的批次顺序会接着模型①继续抽，三个模型的训练数据顺序就不一致了。
    每次开始训练一个模型前调用本函数，即可让三者看到完全相同的打乱顺序。

    参数: loader - 需要重置的 DataLoader；seed - 重置到的种子
    返回: 无
    """
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(seed)


def get_numpy_batch(X_train, y_train, indices, batch_size, seed, normalize=True,
                    mean=0.0, std=1.0):
    """从训练集中取一个固定的 mini-batch，转成 NumPy float64。

    专供手写 MLP 的梯度检查使用：梯度检查对数值精度极敏感，
    所以强制 float64，并且固定采样结果（同一个 seed 每次取到同一批数据）。

    参数: X_train - 完整训练集张量 (N, 784)；y_train - 对应标签
          indices - 可采样的索引数组；batch_size - 批大小；seed - 采样种子
          normalize/mean/std - 与 PyTorch 侧保持一致的归一化参数
    返回: (X, y)，X 形状 (N, 784) 的 float64，y 形状 (N,) 的 int64
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.asarray(indices), size=batch_size, replace=False)
    X = X_train[pick].numpy().astype(np.float64)
    y = y_train[pick].numpy().astype(np.int64)
    if normalize:
        X = (X - mean) / std
    return X, y
