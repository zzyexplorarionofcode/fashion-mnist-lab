"""梯度检查：用中心差分的数值梯度验证 NumPy 反向传播的解析梯度。

数值梯度公式（中心差分，误差 O(eps^2)，比前向差分精确得多）：
    g_num[i] = (L(theta + eps*e_i) - L(theta - eps*e_i)) / (2*eps)

相对误差判据：
    rel = |g_num - g_ana| / max(1e-8, |g_num| + |g_ana|)
    一般 rel < 1e-6 即可认为反向传播实现正确。

两个必须注意的实现细节：
    1) 全参数逐个检查不可行：MLP 参数量约 20 万，每个参数要 2 次前向，
       总共 40 万次前向传播。所以按层随机抽样，每层只查若干个参数。
    2) ReLU 在 0 处的不可导点会破坏数值梯度：如果某个隐藏单元的预激活
       Z1[:, h] 有样本落在 0 附近，把参数扰动 eps 后该样本的 ReLU 开关会翻转，
       数值梯度就会失真。所以抽样时主动避开这类不稳定的隐藏单元。
"""

import csv
import os

import numpy as np

import config
from data_utils import get_numpy_batch, get_split_indices, load_arrays, set_seed
from numpy_mlp import backward, forward, init_params

# 判定隐藏单元是否"远离 ReLU 拐点"的阈值。
# 扰动 eps=1e-4 引起 Z1 的变化量约 eps*|X| ~ 1e-4，取 1e-2 留了 100 倍余量。
KINK_MARGIN = 1e-2


def numerical_grad(params, X, y, key, idx, eps):
    """用中心差分计算某个参数元素上的数值梯度。

    做法是临时把该参数加上/减去 eps 各算一次损失，再还原原值，
    避免污染后续的解析梯度计算。

    参数: params - 参数字典；X, y - 固定的 mini-batch
          key - 参数名（'W1'/'b1'/'W2'/'b2'）
          idx - 参数元素的索引元组
          eps - 差分步长
    返回: 数值梯度（标量 float）
    """
    orig = params[key][idx]
    params[key][idx] = orig + eps
    loss_plus, _ = forward(params, X, y)
    params[key][idx] = orig - eps
    loss_minus, _ = forward(params, X, y)
    params[key][idx] = orig          # 还原，保证解析梯度用的是原始参数
    return (loss_plus - loss_minus) / (2.0 * eps)


def relative_error(ana, num):
    """计算解析梯度与数值梯度的相对误差。

    分母加 |g_num| + |g_ana| 并设下限 1e-8，是为了在梯度本身接近 0 时
    不会因为除以极小数而得到虚假的巨大误差。

    参数: ana - 解析梯度；num - 数值梯度
    返回: 相对误差（float）
    """
    return abs(ana - num) / max(1e-8, abs(ana) + abs(num))


def pick_safe_units(Z1, num_wanted, rng):
    """挑出远离 ReLU 拐点的隐藏单元下标。

    参数: Z1 - 形状 (N, H) 的隐藏层预激活；num_wanted - 需要的单元个数
          rng - NumPy 随机数生成器
    返回: (safe_units, num_excluded)
          safe_units 为可安全检查的隐藏单元下标数组
          num_excluded 为因靠近拐点被排除的单元数
    """
    H = Z1.shape[1]
    # 每个隐藏单元在所有样本上的最小 |Z1|，越大说明离 ReLU 拐点越远
    min_abs = np.abs(Z1).min(axis=0)
    safe = np.where(min_abs > KINK_MARGIN)[0]
    num_excluded = H - len(safe)

    if len(safe) == 0:
        raise RuntimeError("所有隐藏单元都靠近 ReLU 拐点，无法做梯度检查")

    # 需要多少取多少；不够就全部用上
    take = min(num_wanted, len(safe))
    chosen = rng.choice(safe, size=take, replace=False)
    return np.sort(chosen), num_excluded


def check_gradients(params, X, y, eps=config.GRAD_CHECK_EPS,
                    per_layer=config.GRAD_CHECK_PER_LAYER,
                    seed=config.GRAD_CHECK_SEED):
    """对四组参数分层抽样做中心差分梯度检查。

    参数: params - 参数字典；X, y - 固定的 mini-batch
          eps - 差分步长；per_layer - 每层抽样参数个数；seed - 采样种子
    返回: (records, summary, num_excluded)
          records 为明细列表，每项 (参数名, 索引, 解析梯度, 数值梯度, 相对误差)
          summary 为每层的最大相对误差字典
          num_excluded 为因靠近 ReLU 拐点被排除的隐藏单元数
    """
    rng = np.random.default_rng(seed)

    # 先算一次解析梯度，并保留中间量用于筛选安全的隐藏单元
    _, cache = forward(params, X, y)
    grads = backward(params, cache)
    Z1 = cache["Z1"]
    H, C = params["W2"].shape
    D = params["W1"].shape[0]

    safe_units, num_excluded = pick_safe_units(Z1, per_layer, rng)

    # 为四组参数分别构造待检查的索引列表
    # W1/b1/W2 的梯度都要穿过 ReLU，所以隐藏单元下标必须从安全集合里取
    jobs = []
    for _ in range(per_layer):
        h = int(rng.choice(safe_units))
        jobs.append(("W1", (int(rng.integers(D)), h)))
    for h in safe_units[:per_layer]:
        jobs.append(("b1", (int(h),)))
    for _ in range(per_layer):
        h = int(rng.choice(safe_units))
        jobs.append(("W2", (h, int(rng.integers(C)))))
    for j in range(min(per_layer, C)):
        jobs.append(("b2", (int(j),)))

    records = []
    for key, idx in jobs:
        ana = float(grads[key][idx])
        num = numerical_grad(params, X, y, key, idx, eps)
        records.append((key, idx, ana, num, relative_error(ana, num)))

    summary = {}
    for key, _, _, _, rel in records:
        summary[key] = max(summary.get(key, 0.0), rel)
    return records, summary, num_excluded


def main():
    """入口：取固定 mini-batch，跑梯度检查，打印并保存结果。

    参数: 无
    返回: 无
    """
    config.setup_console()
    config.ensure_dirs()
    set_seed(config.SEED)

    X_train, y_train, _, _ = load_arrays()
    train_idx, _ = get_split_indices(X_train.size(0))

    # 固定的一个 mini-batch，float64。归一化参数与 PyTorch 侧保持一致
    mean = float(X_train[train_idx].mean().item())
    std = float(X_train[train_idx].std().item())
    X, y = get_numpy_batch(X_train, y_train, train_idx, config.GRAD_CHECK_BATCH,
                           config.GRAD_CHECK_SEED, mean=mean, std=std)
    print(f"[梯度检查] 固定 mini-batch: X{X.shape} y{y.shape} dtype={X.dtype}")

    params = init_params(config.IN_DIM, config.HIDDEN_DIM, config.NUM_CLASSES,
                         config.SEED)
    for k, v in params.items():
        params[k] = v.astype(np.float64)

    loss, _ = forward(params, X, y)
    print(f"[梯度检查] 当前损失 = {loss:.6f}")

    records, summary, num_excluded = check_gradients(params, X, y)
    print(f"[梯度检查] eps={config.GRAD_CHECK_EPS}, 每层抽样 "
          f"{config.GRAD_CHECK_PER_LAYER} 个参数, "
          f"因靠近 ReLU 拐点排除 {num_excluded} 个隐藏单元")

    print("\n--- 明细 ---")
    print(f"{'参数':<4} {'索引':<12} {'解析梯度':>16} {'数值梯度':>16} {'相对误差':>12}")
    for key, idx, ana, num, rel in records:
        print(f"{key:<4} {str(idx):<12} {ana:>16.10f} {num:>16.10f} {rel:>12.3e}")

    print("\n--- 各层最大相对误差 ---")
    for key in ["W1", "b1", "W2", "b2"]:
        print(f"{key:<4} max_rel_err = {summary[key]:.3e}")

    overall = max(summary.values())
    print(f"\n总体最大相对误差 = {overall:.3e}  "
          f"-> {'通过 (<1e-6)' if overall < 1e-6 else '未通过，需要排查'}")

    csv_path = os.path.join(config.OUTPUT_DIR, "grad_check.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "index", "analytic", "numerical", "rel_error"])
        for key, idx, ana, num, rel in records:
            writer.writerow([key, str(idx), f"{ana:.12e}", f"{num:.12e}", f"{rel:.6e}"])
        writer.writerow([])
        writer.writerow(["summary", "max_rel_error", "", "", ""])
        for key in ["W1", "b1", "W2", "b2"]:
            writer.writerow([key, f"{summary[key]:.6e}", "", "", ""])
        writer.writerow(["overall", f"{overall:.6e}", "", "", ""])
    print(f"[梯度检查] 结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
