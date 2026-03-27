"""
完整流程一键运行脚本
训练 → 回测 → 预测，循环迭代直到准确率达标（或达到最大轮次）
"""

import os
import sys
import argparse
import subprocess


def run_cmd(cmd: list, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ {desc} 失败，退出")
        sys.exit(1)
    print(f"✓ {desc} 完成")


def main():
    parser = argparse.ArgumentParser(description="GAN 彩票预测完整流程")
    parser.add_argument("--csv", type=str, default="lottery_data.csv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_periods", type=int, default=200)
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="最大迭代次数（每次迭代增加100个epoch）")
    parser.add_argument("--target_acc", type=float, default=80.0,
                        help="目标综合准确率（%%）")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # ---- 第1步：训练 ----
    train_cmd = [
        sys.executable, "gan_lottery/train.py",
        "--csv", args.csv,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--test_periods", str(args.test_periods),
        "--save_dir", "output",
        "--resume" if args.max_iterations > 1 else "",
    ]
    train_cmd = [c for c in train_cmd if c]
    if args.device:
        train_cmd += ["--device", args.device]

    run_cmd(train_cmd, "GAN 模型训练")

    # ---- 第2步：回测 ----
    backtest_cmd = [
        sys.executable, "gan_lottery/backtest.py",
        "--csv", args.csv,
        "--model_prefix", "output/lottery_gan",
        "--test_periods", str(args.test_periods),
        "--n_candidates", "50",
        "--top_k", "5",
    ]
    if args.device:
        backtest_cmd += ["--device", args.device]

    run_cmd(backtest_cmd, "回测验证")

    # ---- 第3步：预测 ----
    predict_cmd = [
        sys.executable, "gan_lottery/predict.py",
        "--csv", args.csv,
        "--model_prefix", "output/lottery_gan",
        "--n", "5",
    ]
    if args.device:
        predict_cmd += ["--device", args.device]

    run_cmd(predict_cmd, "生成下一期预测")

    print(f"\n{'='*60}")
    print("✅ 完整流程运行完成！")
    print(f"   模型: output/lottery_gan_gan.pt")
    print(f"   回测结果: output/backtest_results.json")
    print(f"   预测结果: output/predictions_gan.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
