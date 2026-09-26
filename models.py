"""受控比较用的三个 PyTorch 模型。

三个模型的唯一区别是网络结构本身，训练条件（数据划分、优化器、学习率、
批大小、epoch、种子）在所有脚本里保持一致。
"""

import torch.nn as nn

import config


class LinearSoftmax(nn.Module):
    """模型①：单层线性分类器 + Softmax。

    结构：784 -> 10 的全连接层。
    配合 nn.CrossEntropyLoss 使用（该损失内部已含 log_softmax，所以 forward 输出 logits）。
    """

    def __init__(self, in_dim=config.IN_DIM, num_classes=config.NUM_CLASSES):
        """参数: in_dim - 输入维度；num_classes - 类别数"""
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        """前向传播。
        参数: x - 形状 (N, 1, 28, 28) 的图像批
        返回: 形状 (N, 10) 的 logits
        """
        x = x.view(x.size(0), -1)   # 展平成 (N, 784)
        return self.fc(x)


class TwoLinearNoAct(nn.Module):
    """模型②：两个线性层，中间不加任何激活函数。

    结构：784 -> H -> 10，两层之间没有 ReLU。
    数学上两个线性变换复合仍是线性变换（W2·W1 等价于单个矩阵），
    所以这个模型的表达能力与模型①完全相同，它存在的意义就是验证
    "单纯堆叠线性层、不加非线性，不会带来任何增益"。
    """

    def __init__(self, in_dim=config.IN_DIM, hidden=config.HIDDEN_DIM,
                 num_classes=config.NUM_CLASSES):
        """参数: in_dim - 输入维度；hidden - 隐藏层宽度；num_classes - 类别数"""
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        """前向传播（两层线性，无激活）。
        参数: x - 形状 (N, 1, 28, 28) 的图像批
        返回: 形状 (N, 10) 的 logits
        """
        x = x.view(x.size(0), -1)
        return self.fc2(self.fc1(x))


class MLPReLU(nn.Module):
    """模型③：两个线性层 + ReLU 激活的单隐藏层 MLP。

    结构：784 -> H -> ReLU -> 10。
    与模型②的唯一区别就是中间多了 ReLU，用来验证非线性激活是否带来真实增益。
    """

    def __init__(self, in_dim=config.IN_DIM, hidden=config.HIDDEN_DIM,
                 num_classes=config.NUM_CLASSES):
        """参数: in_dim - 输入维度；hidden - 隐藏层宽度；num_classes - 类别数"""
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        """前向传播（线性 -> ReLU -> 线性）。
        参数: x - 形状 (N, 1, 28, 28) 的图像批
        返回: 形状 (N, 10) 的 logits
        """
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


def count_parameters(model):
    """统计模型的可训练参数量，用于性能表对比。

    参数: model - nn.Module
    返回: 参数量（Python int）
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name):
    """按名称构建模型，三模型共用同一套构建入口方便统一训练。

    参数: name - 'linear' | 'noact' | 'relu'
    返回: 对应的 nn.Module 实例
    """
    if name == "linear":
        return LinearSoftmax()
    if name == "noact":
        return TwoLinearNoAct()
    if name == "relu":
        return MLPReLU()
    raise ValueError(f"未知模型名称: {name}")
