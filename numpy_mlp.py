"""纯 NumPy 实现的单隐藏层 MLP 前向与反向传播（不使用任何自动求导）。

网络结构：
    X -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> Softmax -> 交叉熵损失

约定：
    N = 批大小，D = 输入维度(784)，H = 隐藏层宽度，C = 类别数(10)
    W1: (D, H)   b1: (H,)
    W2: (H, C)   b2: (C,)

全部计算使用 float64。梯度检查对数值精度极敏感，float32 下中心差分的
舍入误差会淹没真实的梯度误差，导致检查结果没有意义。
"""

import numpy as np


def init_params(in_dim, hidden_dim, num_classes, seed):
    """按 He 初始化生成网络参数。

    He 初始化：标准差取 sqrt(2 / fan_in)，配合 ReLU 使用，能避免深层前向
    信号逐层衰减，也让初始梯度处在合适的量级。

    参数: in_dim - 输入维度；hidden_dim - 隐藏层宽度
          num_classes - 类别数；seed - 随机种子
    返回: 参数字典 {'W1', 'b1', 'W2', 'b2'}
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, num_classes))
    b2 = np.zeros(num_classes)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def softmax(z):
    """数值稳定的 softmax（按行归一化）。

    减去每行最大值是为了防止 exp 溢出：softmax 的结果对整体平移不变，
    所以减最大值不改变数学结果，但能避免大 logits 导致 inf。

    参数: z - 形状 (N, C) 的 logits
    返回: 形状 (N, C) 的概率分布，每行和为 1
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X, y):
    """前向传播：计算损失并缓存反向传播需要的中间量。

    参数: params - 参数字典；X - 输入 (N, D)；y - 真实标签 (N,)
    返回: (loss, cache)
          loss 为标量交叉熵（对 batch 取平均）
          cache 含 X / Z1 / A1 / P / y，供 backward 复用，避免重复计算
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N = X.shape[0]

    # 第一层线性变换 + ReLU
    Z1 = X @ W1 + b1
    A1 = np.maximum(Z1, 0.0)

    # 第二层线性变换 + softmax
    Z2 = A1 @ W2 + b2
    P = softmax(Z2)

    # 交叉熵：取出每个样本真实类别对应的预测概率，取负对数后对 batch 求平均
    # 加 1e-12 只是防止 log(0) 产生 inf，对梯度的影响在 1e-12 量级，可忽略
    loss = -np.log(P[np.arange(N), y] + 1e-12).mean()

    cache = {"X": X, "Z1": Z1, "A1": A1, "P": P, "y": y}
    return loss, cache


def backward(params, cache):
    """反向传播：解析法计算损失对四组参数的梯度。

    推导要点（自上而下逐层回传）：
        1) softmax + 交叉熵合并求导后形式非常简洁：dZ2 = (P - onehot(y)) / N
           其中除以 N 来自 loss 对 batch 取平均。
        2) 第二层是线性层：dW2 = A1^T @ dZ2，db2 = dZ2 按样本求和
        3) 回传到隐藏层：dA1 = dZ2 @ W2^T
        4) 穿过 ReLU：dZ1 = dA1 * (Z1 > 0)，即正值处梯度原样通过，负值处截断为 0
        5) 第一层同第二层：dW1 = X^T @ dZ1，db1 = dZ1 按样本求和

    参数: params - 参数字典（只用 W2）；cache - forward 返回的缓存
    返回: 与 params 同键的梯度字典
    """
    X, Z1, A1, P, y = cache["X"], cache["Z1"], cache["A1"], cache["P"], cache["y"]
    W2 = params["W2"]
    N = X.shape[0]

    # 第 1 步：softmax + 交叉熵的合并导数
    dZ2 = P.copy()
    dZ2[np.arange(N), y] -= 1.0
    dZ2 /= N

    # 第 2 步：第二层参数梯度
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    # 第 3 步：误差回传到隐藏层激活
    dA1 = dZ2 @ W2.T

    # 第 4 步：穿过 ReLU。用严格大于 0，Z1 恰好等于 0 的位置取次梯度 0
    dZ1 = dA1 * (Z1 > 0)

    # 第 5 步：第一层参数梯度
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def predict(params, X):
    """用当前参数做预测（前向取 argmax），供检查代码使用。

    参数: params - 参数字典；X - 输入 (N, D)
    返回: 形状 (N,) 的预测类别
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    A1 = np.maximum(X @ W1 + b1, 0.0)
    Z2 = A1 @ W2 + b2
    return np.argmax(Z2, axis=1)
