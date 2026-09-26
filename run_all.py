"""一键跑通全部实验。

依次执行：梯度检查 -> 线性基线 -> 三模型受控比较 -> 学习率诊断。
各部分也可以单独运行对应的脚本，输出是一样的。

用法：
    python run_all.py
"""

import time

import config
import diagnose
import grad_check
import train_compare
import train_linear


def main():
    """按顺序执行四个实验，并打印每一步的耗时。

    参数: 无
    返回: 无
    """
    config.setup_console()
    config.ensure_dirs()

    steps = [
        ("1/4 梯度检查", grad_check.main),
        ("2/4 线性 Softmax 基线", train_linear.main),
        ("3/4 三模型受控比较", train_compare.main),
        ("4/4 学习率诊断", diagnose.main),
    ]

    timings = []
    for name, fn in steps:
        print("\n" + "#" * 60)
        print(f"# {name}")
        print("#" * 60)
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        timings.append((name, elapsed))
        print(f"\n>>> {name} 完成，耗时 {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("全部实验完成，各部分耗时：")
    for name, elapsed in timings:
        print(f"  {name:<24} {elapsed:>8.1f}s")
    print(f"  {'合计':<24} {sum(t for _, t in timings):>8.1f}s")
    print(f"\n所有结果已写入: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
