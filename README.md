# Fashion-MNIST 十分类实验

在 Fashion-MNIST 上完成十分类任务，共四个实验：NumPy 手写反向传播与梯度检查、线性 Softmax 基线、
三模型受控比较、训练诊断（学习率）。完整实验报告见 `report/`。

## 1. 环境

实测通过的版本：

| 依赖 | 版本 |
|---|---|
| Python | 3.13.3 |
| numpy | 2.1.1 |
| torch | 2.12.0+cpu |
| torchvision | 0.27.0+cpu |
| matplotlib | 3.10.7 |

全部实验在 **CPU** 上完成，无需 GPU（也没有用到 CUDA）。安装依赖：

```bash
pip install numpy torch torchvision matplotlib
```

数据集：首次运行会通过 torchvision 自动下载 Fashion-MNIST 到 `data/`（约 30MB）。
若网络不通，可手动下载 4 个文件放到 `data/FashionMNIST/raw/`：
`train-images-idx3-ubyte.gz`、`train-labels-idx1-ubyte.gz`、
`t10k-images-idx3-ubyte.gz`、`t10k-labels-idx1-ubyte.gz`。
下载源为 `http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/`。

## 2. 目录结构

```
fashion-mnist-lab/
├── config.py          全局配置：随机种子、划分比例、超参数、路径
├── data_utils.py      数据加载、固定划分、DataLoader 构建
├── models.py          三个 PyTorch 模型（线性 / 双层无激活 / 双层+ReLU）
├── numpy_mlp.py       纯 NumPy 的 forward 与 backward
├── grad_check.py      实验一：中心差分梯度检查
├── train_linear.py    实验二：线性 Softmax 基线
├── train_compare.py   实验三：三模型受控比较
├── diagnose.py        实验四：学习率诊断
├── trainer.py         三模型共用的训练与评估循环
├── evaluate.py        混淆矩阵、逐类召回率、曲线、典型错误可视化
├── run_all.py         一键跑通全部实验
├── report/            实验报告：PDF 成品 + HTML 源文件
├── data/              Fashion-MNIST 数据集（自动下载，未入库）
└── outputs/           所有结果：图、CSV、JSON
    └── figures/
```

## 3. 运行方式

一键跑完全部实验（约 4 分钟）：

```bash
python run_all.py
```

也可以单独运行某一部分，输出与一键运行完全一致：

```bash
python grad_check.py       # 梯度检查，<10 秒
python train_linear.py     # 线性 Softmax 基线，约 30 秒
python train_compare.py    # 三模型受控比较，约 2 分钟
python diagnose.py         # 学习率诊断，约 100 秒
```

各脚本之间没有依赖顺序要求，但**第一次运行**需要先让任意一个脚本完成数据下载。

## 4. 配置说明

所有超参数集中在 `config.py`，改一处即可全局生效：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `SEED` | 42 | 全局随机种子 |
| `TRAIN_SIZE` / `VAL_SIZE` | 50000 / 10000 | 官方训练集的固定划分 |
| `BATCH_SIZE` | 128 | 批大小 |
| `EPOCHS` | 30 | 主实验训练轮数 |
| `LR` | 0.01 | SGD 学习率 |
| `MOMENTUM` | 0.9 | SGD 动量 |
| `WEIGHT_DECAY` | 0.0 | L2 正则，受控比较中保持为 0 |
| `HIDDEN_DIM` | 256 | 模型②③共用的隐藏层宽度 |
| `DIAG_LR_LIST` | [0.01, 0.1, 1.0] | 诊断实验的学习率取值 |
| `DIAG_EPOCHS` | 20 | 诊断实验训练轮数 |
| `GRAD_CHECK_EPS` | 1e-4 | 中心差分步长（选取依据见第 7 节） |
| `GRAD_CHECK_PER_LAYER` | 20 | 每层抽样的参数个数 |

**学习率为何取 0.01**：实测 SGD + momentum 0.9 下，0.01 时三个模型都能稳定收敛；
0.1 明显变差；1.0 直接震荡不收敛。详见 `diagnose.py` 的对照实验。

## 5. 随机种子与可复现性

复现性由三处共同保证：

1. **固定数据划分**：`outputs/split_indices.npz` 保存训练/验证的索引，首次运行生成后不再变动。
   删除该文件会重新划分，三模型仍共用同一份新划分，比较依然受控，但数值结果会变。
2. **建模前重置种子**：每个模型构建之前都调用 `set_seed(SEED)`。
   注意 `load_arrays()` 遍历 DataLoader 时会消耗全局 torch 随机源，因此**不能在脚本开头设一次种子就算了**，
   否则同一模型在不同脚本里会得到不同的初始权重。
3. **重置 DataLoader 的打乱顺序**：`reset_loader_seed()` 在每次训练前把 generator 拨回种子位置。
   否则三个模型会依次接着抽批次顺序，训练数据顺序不一致，比较就不受控了。

在相同环境下重复运行，结果应完全一致。

## 6. 输出文件清单

| 文件 | 对应实验 | 内容 |
|---|---|---|
| `outputs/grad_check.csv` | 梯度检查 | 解析梯度、数值梯度、相对误差明细 |
| `outputs/linear_summary.json` | 线性基线 | 测试准确率、逐类召回率、混淆组合 |
| `outputs/figures/linear_curves_*.png` | 线性基线 | 训练/验证的 loss 与 accuracy 曲线 |
| `outputs/figures/linear_cm*.png` | 线性基线 | 混淆矩阵（计数 / 行归一化） |
| `outputs/figures/linear_recall.png` | 线性基线 | 逐类召回率柱状图 |
| `outputs/figures/linear_errors.png` | 线性基线 | 典型错误样本 |
| `outputs/per_class_recall_linear.csv` | 线性基线 | 逐类召回率表 |
| `outputs/compare_table.csv` | 受控比较 | 三模型性能表 |
| `outputs/figures/compare_curves_*.png` | 受控比较 | 三模型曲线叠在一起 |
| `outputs/figures/compare_test_acc.png` | 受控比较 | 测试准确率对比柱状图 |
| `outputs/figures/compare_cm_*.png` | 受控比较 | 每个模型的混淆矩阵 |
| `outputs/figures/compare_recall_*.png` | 受控比较 | 每个模型的逐类召回率 |
| `outputs/diagnose_table.csv` | 训练诊断 | 各学习率的结果汇总 |
| `outputs/figures/diag_step_loss.png` | 训练诊断 | 首个 epoch 内逐 step 损失 |
| `outputs/figures/diag_train_loss.png` | 训练诊断 | 逐 epoch 训练损失 |
| `outputs/figures/diag_val_acc.png` | 训练诊断 | 逐 epoch 验证准确率 |

## 7. 实测结果摘要

### 梯度检查

在固定 mini-batch（64 个样本，float64）上，按层各抽样 20 个参数做中心差分，
损失值 3.918401，步长 eps=1e-4：

| 参数 | 最大相对误差 |
|---|---|
| W1 | 2.72e-09 |
| b1 | 4.83e-09 |
| W2 | 5.92e-09 |
| b2 | 2.31e-08 |
| **总体** | **2.31e-08** |

判据为相对误差 < 1e-6，全部通过，说明 `numpy_mlp.py` 的解析梯度与数值梯度一致。

需要说明的一点：抽样时有 86 个（共 256 个）隐藏单元被主动排除。
这些单元存在 `|Z1| < 1e-2` 的样本，把参数扰动 eps 后该样本的 ReLU 开关会翻转，
中心差分就不再是真实梯度的近似。排除它们后剩下的检查才有意义。

### 差分步长 h 的选取

中心差分的误差由两部分叠加：截断误差随 h 减小而减小（约 `h²|f‴|/6`），
舍入误差随 h 减小而增大（约 `ε|L|/(2h)`），因此存在一个最优步长。
在同一个固定 mini-batch、同一批参数上把 h 从 1e-3 扫到 1e-8，实测得到清晰的 V 形：

| 步长 h | 最大绝对误差 | 最大相对误差 | 主导因素 |
|---|---|---|---|
| 1e-3 | 1.18e-07 | 2.44e-06 | 截断误差 |
| **1e-4（本实验采用）** | **1.29e-09** | **2.31e-08** | 截断误差 |
| 3e-5 | 2.27e-10 | 6.59e-09 | 两者相当（最优） |
| 3e-6 | 2.61e-10 | 9.06e-08 | 舍入误差 |
| 1e-7 | 5.11e-09 | 4.44e-06 | 舍入误差 |
| 1e-8 | 3.38e-08 | 2.27e-05 | 舍入误差 |

- 左侧下降段：1e-3 → 1e-4 步长缩小 10 倍，误差缩小约 92 倍，符合 `O(h²)`。
- 右侧上升段：到 1e-7 时实测 5.11e-09，由 `ε|L|/(2h)` 算得 4.35e-09，符合 `O(1/h)`。
- 实测最优在 h=3e-5（2.27e-10），理论式 `h* ≈ (3ε|L|/|f‴|)^(1/3)` 预测 1.4e-5，同量级。

**为什么仍取 1e-4**：它落在 V 形左侧的平坦段，绝对误差 1.29e-09 只比最优值高约 6 倍，
却远离右侧舍入误差急剧上升的区域，是更稳妥的取值。

顺带一个反面教材：若把 dtype 换成 float32（`ε ≈ 1.2e-7`），同样 h=1e-4 下舍入误差升到
约 2.3e-3，比 1e-6 的判据本身还高三个数量级，测出的「误差」将全是浮点噪声。
**梯度检查必须在 float64 下做。**

### 三模型性能表

| 模型 | 参数量 | 末轮训练loss | 末轮验证loss | 最优验证acc | 测试acc |
|---|---|---|---|---|---|
| ① 线性 Softmax | 7,850 | 0.3806 | 0.4542 | 0.8523 | 0.8319 |
| ② 双层线性（无激活） | 203,530 | 0.3707 | 0.4605 | 0.8572 | 0.8305 |
| ③ 双层线性 + ReLU | 203,530 | 0.1192 | 0.3373 | 0.8965 | 0.8820 |

两个关键对照：

- **② vs ①**：参数量是①的 25.9 倍，测试准确率反而低 0.14 个百分点。
  两层线性之间不加激活时，`W2·W1` 仍是一个线性变换，表达能力与单层完全相同，
  多出来的参数没有任何作用 —— 这条对照直接说明"堆叠线性层不等于变深"。
- **③ vs ②**：参数量完全相同，仅多一个 ReLU，测试准确率提升 5.15 个百分点。
  增益完全来自非线性激活，而非参数容量。

另外③的训练损失（0.1192）明显低于验证损失（0.3373），已经出现过拟合迹象。

### 线性基线的逐类召回率（测试集，acc=0.8319）

召回率最低的四类：Shirt 0.512、T-shirt/top 0.705、Pullover 0.710、Coat 0.847。
混淆集中在 Shirt → Coat(196)、T-shirt/top → Shirt(192)、Pullover → Coat(188)、Shirt → Pullover(138)，
即上衣类（T-shirt/top、Shirt、Pullover、Coat）之间互相混淆 —— 这几类在 28×28 灰度图下轮廓确实接近。
其余五类（Sandal、Sneaker、Ankle boot、Bag、Trouser）的召回率都在 0.91 以上，最高的是 Trouser 0.956。

### 训练诊断（学习率）

现象：学习率取 1.0 时训练完全不收敛，损失在几十到两百之间剧烈震荡。
原因判断：步长超过损失曲面的稳定范围，参数一步跨过谷底形成正反馈；动量项进一步放大有效步长。
验证：固定网络结构、数据划分、优化器、批大小、epoch、种子，只改学习率。

| 学习率 | 首个epoch损失 | 末轮训练损失 | 最优验证acc | 结论 |
|---|---|---|---|---|
| 0.01 | 0.5616 | 0.1677 | 0.8965 | 正常收敛 |
| 0.1 | 0.5556 | 0.2346 | 0.8753 | 收敛但明显更差 |
| 1.0 | 69.99 | 2.3081 | 0.1374 | 未收敛/震荡 |

学习率 1.0 时首个 epoch 的损失就已经冲到 69.99，第一个 epoch 内在 0.5 ~ 1800 之间反复震荡；
从第 7 个 epoch 起损失停在 2.303，正好是 `log(10)`，等价于对 10 个类均匀猜测，
验证准确率也只有约 10%，即模型完全没有学到任何东西。
继续加大到 2.0 时损失会直接溢出到 1e26 量级。

结论：只改学习率这一个因素就足以让同一个模型从正常收敛变成完全失败，
证实原因是步长过大，而不是网络结构或数据问题。

## 8. 常见问题

**Q: 运行时报数据集下载失败？**
A: 见第 1 节，手动下载 4 个 `.gz` 文件放入 `data/FashionMNIST/raw/` 即可，脚本检测到文件存在就不会联网。

**Q: 想换隐藏层宽度做容量实验？**
A: 改 `config.py` 里的 `HIDDEN_DIM` 即可，模型②③会同时改变，宽度仍然一致，比较依然受控。

**Q: 想复现完全相同的数值？**
A: 不要删除 `outputs/split_indices.npz`，并保持 `config.py` 的种子与超参不变。
