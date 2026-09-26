"""全局配置。

所有脚本统一从这里读取超参数，保证"受控比较"的前提成立：
三个模型的数据划分、优化器、学习率、批大小、epoch 数、随机种子全部一致。
"""

import os
import sys


def setup_console():
    """把标准输出改成 UTF-8，避免 Windows 中文控制台打印中文时乱码。

    参数: 无
    返回: 无
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        # 老版本 Python 或输出被重定向到不支持重配置的对象时，忽略即可
        pass


# ---------------- 随机种子 ----------------
# 固定种子保证划分、初始化、DataLoader 打乱顺序全部可复现
SEED = 42

# ---------------- 路径 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")           # Fashion-MNIST 原始数据
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")      # 结果（表格、json）
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")       # 图（曲线、混淆矩阵、错误样本）
SPLIT_FILE = os.path.join(OUTPUT_DIR, "split_indices.npz")  # 固定的训练/验证划分索引

# ---------------- 数据划分 ----------------
# 官方训练集 60000 张，固定切成 50000 训练 + 10000 验证；官方测试集 10000 张只用于最终评价
TRAIN_SIZE = 50000
VAL_SIZE = 10000

# ---------------- 训练超参数（三模型共用） ----------------
BATCH_SIZE = 128
EPOCHS = 30
# 学习率经实测选定：0.01 时三个模型都能稳定收敛；
# 0.1 明显变差，1.0 直接震荡发散（见 diagnose.py 的对照实验）
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0       # 受控比较中保持 0，避免引入额外变量
HIDDEN_DIM = 256         # 模型②③共用的隐藏层宽度
IN_DIM = 784             # 28x28 展平
NUM_CLASSES = 10

# ---------------- 学习率诊断实验 ----------------
# 只改学习率，网络结构、数据划分、优化器、批大小、epoch、种子全部与主实验一致
DIAG_LR_LIST = [0.01, 0.1, 1.0]
DIAG_EPOCHS = 20

# ---------------- 梯度检查 ----------------
GRAD_CHECK_BATCH = 64        # 固定 mini-batch 大小
GRAD_CHECK_EPS = 1e-4        # 中心差分步长
GRAD_CHECK_PER_LAYER = 20    # 每层随机抽多少个参数做检查
GRAD_CHECK_SEED = 0          # 采样种子

# ---------------- 类别名称（Fashion-MNIST 官方 10 类） ----------------
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def ensure_dirs():
    """创建所有输出目录（已存在则忽略）。
    参数: 无
    返回: 无
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def model_config_text():
    """生成一段描述当前超参的文本，便于写入结果文件做记录。
    参数: 无
    返回: 字符串
    """
    return (
        f"seed={SEED}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, "
        f"optimizer=SGD(lr={LR}, momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}), "
        f"hidden_dim={HIDDEN_DIM}, split={TRAIN_SIZE}/{VAL_SIZE}"
    )
