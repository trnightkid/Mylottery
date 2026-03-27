"""
GAN 预测脚本
用训练好的 GAN 模型对下一期（或多期）做预测，
结合统计特征生成候选号码，并输出推荐

用法：
  python predict.py                           # 预测下一期
  python predict.py --period 26030            # 预测指定期号
  python predict.py --n 10                    # 生成10个候选
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gan_model import LotteryGAN
from dataset import LotteryDataset


def generate_predictions(
    gan: LotteryGAN,
    dataset: LotteryDataset,
    n_predictions: int = 5,
    n_candidates: int = 50,
    device: str = None,
) -> list:
    """
    生成下一期（或多期）的预测号码

    Args:
        gan: 训练好的模型
        dataset: 数据集
        n_predictions: 生成几组不同预测
        n_candidates: 每组取多少个候选
        device: 设备
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gan.eval()
    latest_idx = len(dataset) - 1
    latest_period = dataset.records[latest_idx]['period']

    results = []

    for pred_round in range(n_predictions):
        # 用最新一期及之前的数据构建条件
        cond = dataset.get_condition_vector(latest_idx, window=200)
        cond_t = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)

        candidates = []
        with torch.no_grad():
            for _ in range(n_candidates):
                z = torch.randn(1, gan.noise_dim, device=device)
                red_idx, blue_idx = gan.generate(1, cond_t)
                red = (red_idx.cpu().numpy()[0] + 1).tolist()
                blue = blue_idx.cpu().item() + 1
                red_sorted = sorted(red)
                candidates.append({
                    'red': red_sorted,
                    'blue': blue,
                    'red_str': ' '.join(f"{r:02d}" for r in red_sorted),
                    'blue_str': f"{blue:02d}",
                })

        # 统计分析：选最"均衡"的
        # 标准：号码分布均匀、奇偶比例适中、区间分布合理
        def score_candidate(c):
            r = c['red']
            # 区间分布（1-11, 12-22, 23-33）各至少1个
            zone1 = sum(1 for x in r if 1 <= x <= 11)
            zone2 = sum(1 for x in r if 12 <= x <= 22)
            zone3 = sum(1 for x in r if 23 <= x <= 33)
            zone_score = min(zone1, zone2, zone3)  # 各区间至少1个

            # 奇偶
            odd = sum(1 for x in r if x % 2 == 1)
            even = 6 - odd
            odd_even_score = min(odd, even)  # 奇偶均衡

            # 和值（60-120 区间较常见）
            s = sum(r)
            sum_score = 1 if 60 <= s <= 120 else 0

            return zone_score * 3 + odd_even_score * 2 + sum_score

        candidates.sort(key=score_candidate, reverse=True)

        best = candidates[0]
        results.append({
            'prediction_round': pred_round + 1,
            'based_on_period': latest_period,
            'recommended': best,
            'alternatives': candidates[1:6],
        })

    return results, latest_period


def main():
    parser = argparse.ArgumentParser(description="GAN 双色球预测")
    parser.add_argument("--csv", type=str, default="lottery_data.csv",
                        help="历史数据 CSV")
    parser.add_argument("--model_prefix", type=str, default="output/lottery_gan",
                        help="模型前缀路径")
    parser.add_argument("--n", type=int, default=5,
                        help="生成多少组预测")
    parser.add_argument("--n_candidates", type=int, default=50,
                        help="每组取多少个候选中选最优")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/predictions_gan.json")

    args = parser.parse_args()

    if device := args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据集
    if not os.path.exists(args.csv):
        print(f"❌ 数据文件不存在: {args.csv}")
        print("请确保 lottery_data.csv 在当前目录")
        return

    dataset = LotteryDataset(args.csv)
    print(f"✓ 数据加载完成: {len(dataset)} 期")
    print(f"  最新期号: {dataset.records[-1]['period']}")
    print(f"  最新开奖: 红球 {dataset.records[-1]['red']}  蓝球 {dataset.records[-1]['blue']}\n")

    # 加载模型
    model_path = f"{args.model_prefix}_gan.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行: python train.py --csv lottery_data.csv --epochs 300")
        return

    cond_dim = 99
    gan = LotteryGAN(noise_dim=64, cond_dim=cond_dim, device=device)
    gan.load(args.model_prefix)
    print(f"✓ 模型加载成功\n")

    # 生成预测
    results, latest_period = generate_predictions(
        gan, dataset,
        n_predictions=args.n,
        n_candidates=args.n_candidates,
        device=device,
    )

    # 输出
    print(f"{'='*50}")
    print(f"  GAN 预测结果（基于第 {latest_period} 期数据）")
    print(f"{'='*50}\n")

    for r in results:
        print(f"【推荐 #{r['prediction_round']}】")
        print(f"  红球: {r['recommended']['red_str']}")
        print(f"  蓝球: {r['recommended']['blue_str']}")
        print(f"  完整: {r['recommended']['red_str']} + {r['recommended']['blue_str']}")
        print()

        if r['alternatives']:
            print(f"  备选:")
            for alt in r['alternatives']:
                print(f"    {alt['red_str']} + {alt['blue_str']}")
            print()

    # 保存 JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            'latest_period': latest_period,
            'predictions': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ 预测结果已保存: {args.output}")


if __name__ == "__main__":
    main()
