"""
预测脚本 v2 - 配合 gan_model_v2.py
生成下一期预测号码
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gan_model_v2 import LotteryGANV2
from dataset import LotteryDataset


def get_latest_condition(dataset: LotteryDataset, window: int = 200):
    """获取最新一期（第 len-1 期）的条件向量"""
    last_idx = len(dataset) - 1
    cond = dataset.get_condition_vector(last_idx, window=window)
    return torch.from_numpy(cond).float().unsqueeze(0)  # (1, 99)


def predict(csv_path: str = "lottery_data.csv",
            model_path_prefix: str = "output/lottery_gan_v2",
            n_samples: int = 16,
            output_path: str = "output/gan_prediction_v2.json",
            device: str = None) -> dict:
    """
    生成最新一期预测
    
    Args:
        csv_path: 历史数据路径
        model_path_prefix: 模型路径前缀
        n_samples: 生成多少个候选用于投票
        output_path: 输出路径
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 加载数据集 ----
    dataset = LotteryDataset(csv_path)
    latest_period = dataset.records[-1]['period']
    latest_date = dataset.records[-1].get('draw_date', 'N/A')
    print(f"最新一期: {latest_period}, 数据日期: {latest_date}")

    # ---- 加载模型 ----
    gan = LotteryGANV2(noise_dim=128, cond_dim=99, device=device)
    if os.path.exists(f"{model_path_prefix}_gan.pt"):
        gan.load(model_path_prefix)
        print(f"✓ 模型已加载: {model_path_prefix}")
    else:
        print(f"⚠ 模型文件不存在: {model_path_prefix}_gan.pt")
        return None

    # ---- 获取条件向量 ----
    cond = get_latest_condition(dataset, window=200).to(device)

    # ---- 统计热冷号 ----
    hot_reds = []
    for rec in dataset.records[-20:]:  # 近20期
        hot_reds.extend(rec['red'])
    from collections import Counter
    red_counter = Counter(hot_reds)
    hot = sorted([(n, c) for n, c in red_counter.items() if c >= 2], key=lambda x: -x[1])
    cold = sorted([(n, c) for n, c in red_counter.items() if c <= 1], key=lambda x: x[1])

    # ---- GAN 多样本生成 ----
    all_red_votes = []
    all_blue_votes = []

    print(f"\n生成 {n_samples} 个候选号码...")
    for i in range(n_samples):
        with torch.no_grad():
            # 温度从低到高变化（第一个确定，后续更随机）
            temp = 0.8 if i == 0 else 1.2
            fake_red, fake_blue = gan.generate(1, cond, temperature=temp)
            reds = (fake_red[0].cpu().numpy() + 1).tolist()
            blue = fake_blue[0].cpu().item() + 1
            all_red_votes.extend(reds)
            all_blue_votes.append(blue)
        
        if i < 5:
            print(f"  样本{i+1}: 红球{sorted(reds)} 蓝球{blue}")

    # ---- 红球投票 ----
    red_counts = Counter(all_red_votes)
    # 选出得票最高的6个
    top6 = sorted(red_counts.items(), key=lambda x: -x[1])[:6]
    plan_a_red = [r for r, _ in top6]  # 热号优先（GAN投票最高的）
    
    # Plan B: 混合方案（GAN + 冷号）
    top_reds_set = set(r for r, _ in top6)
    cold_pool = [n for n, _ in cold if n not in top_reds_set][:3]
    plan_b_red = sorted(top6[:3], key=lambda x: x[1]) + cold_pool
    plan_b_red = sorted(plan_b_red)[:6]

    # Plan C: 纯热号
    plan_c_red = [n for n, _ in hot[:6]]

    # Plan D: 综合评分（热号权重0.6 + GAN权重0.4）
    gan_scores = {r: c / n_samples for r, c in red_counts.items()}
    hot_scores = {n: c / 20 for n, c in red_counter.items()}
    combined = {}
    for n in range(1, 34):
        combined[n] = gan_scores.get(n, 0) * 0.4 + hot_scores.get(n, 0) * 0.6
    plan_d_red = sorted(combined.items(), key=lambda x: -x[1])[:6]
    plan_d_red = [n for n, _ in plan_d_red]

    # ---- 蓝球投票 ----
    blue_counts = Counter(all_blue_votes)
    plan_a_blue = blue_counts.most_common(1)[0][0]

    # 蓝球热号
    recent_blues = [r['blue'] for r in dataset.records[-20:]]
    blue_counter = Counter(recent_blues)
    hot_blue = blue_counter.most_common(1)[0][0] if blue_counter else 4

    # GAN 投票第2热门
    plan_b_blue = blue_counts.most_common(2)[1][0] if len(blue_counts) > 1 else plan_a_blue

    recommendations = {
        "plan_A": {
            "name": "GAN投票优先",
            "red": sorted(plan_a_red),
            "blue": int(plan_a_blue),
        },
        "plan_B": {
            "name": "冷热混合",
            "red": sorted(plan_b_red[:6]),
            "blue": int(hot_blue),
        },
        "plan_C": {
            "name": "近期热号",
            "red": sorted([n for n, _ in hot[:6]]),
            "blue": int(hot_blue),
        },
        "plan_D": {
            "name": "综合评分",
            "red": sorted(plan_d_red),
            "blue": int(plan_a_blue),
        },
    }

    result = {
        "target_period": int(latest_period) + 1,
        "target_date": "2026-04-02",  # 双色球每周二、五开奖
        "latest_period": int(latest_period),
        "latest_date": latest_date,
        "recommendations": recommendations,
        "red_vote_stats": {str(k): v for k, v in red_counts.most_common(10)},
        "blue_vote_stats": {str(k): v for k, v in blue_counts.most_common()},
        "red_hot_cold": {
            "hot": [n for n, _ in hot[:10]],
            "warm": [n for n, _ in sorted(red_counter.items(), key=lambda x: -x[1])[10:20]],
            "cold": [n for n, _ in cold[:5]],
        },
        "gan_blue_distribution": {str(k): v for k, v in blue_counts.most_common()},
        "stats": {
            "total_records": len(dataset),
            "n_samples": n_samples,
            "window": 200,
        }
    }

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 预测结果已保存: {output_path}")
    print(f"\n🌟 第 {result['target_period']} 期预测（{result['target_date']} 开奖）:")
    for plan_name, plan_data in recommendations.items():
        print(f"  {plan_name} [{plan_data['name']}]: "
              f"红球 {' '.join(f'{n:02d}' for n in plan_data['red'])} + 蓝球 {plan_data['blue']:02d}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="lottery_data.csv")
    parser.add_argument("--model", type=str, default="output/lottery_gan_v2")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=str, default="output/gan_prediction_v2.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    predict(args.csv, args.model, args.samples, args.output, args.device)
