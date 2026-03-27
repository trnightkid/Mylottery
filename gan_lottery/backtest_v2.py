"""
回测脚本 v2 - 配合 gan_model_v2.py
用新的 LotteryGANV2 模型对200期历史数据进行回测
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gan_model_v2 import LotteryGANV2
from dataset import LotteryDataset, prepare_backtest_data


def backtest(csv_path: str, model_path_prefix: str = "output/lottery_gan_v2",
              test_periods: int = 200, n_samples: int = 8, device: str = None) -> dict:
    """
    回测 GAN v2 模型
    
    Args:
        csv_path: 历史数据路径
        model_path_prefix: 模型路径前缀
        test_periods: 测试期数
        n_samples: 每期生成多少个候选（用于投票）
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 加载数据 ----
    (train_red, train_blue, train_cond), (test_red, test_blue, test_cond), dataset = \
        prepare_backtest_data(csv_path, test_periods=test_periods)

    # ---- 加载模型 ----
    gan = LotteryGANV2(noise_dim=128, cond_dim=99, device=device)
    if os.path.exists(f"{model_path_prefix}_gan.pt"):
        gan.load(model_path_prefix)
        print(f"✓ 模型已加载: {model_path_prefix}")
    else:
        print(f"⚠ 模型文件不存在: {model_path_prefix}_gan.pt，跳过 GAN 预测")
        gan = None

    # ---- 回测每一期 ----
    results = []
    total = test_red.size(0)
    
    for i in range(total):
        period = dataset.records[dataset.__len__() - total + i]['period']
        real_red = test_red[i].cpu().numpy() + 1  # 转1-33
        real_blue = test_blue[i].cpu().numpy() + 1  # 转1-16
        cond = test_cond[i:i+1].to(device)

        # ---- GAN 预测（多样本投票）----
        if gan is not None:
            all_red_votes = []
            all_blue_votes = []
            for _ in range(n_samples):
                with torch.no_grad():
                    fake_red, fake_blue = gan.generate(1, cond, temperature=1.0)
                    all_red_votes.extend((fake_red[0].cpu().numpy() + 1).tolist())
                    all_blue_votes.append(fake_blue[0].cpu().item() + 1)
            
            # 红球投票（选最常出现的6个）
            red_counts = {}
            for r in all_red_votes:
                red_counts[r] = red_counts.get(r, 0) + 1
            top_reds = sorted(red_counts.items(), key=lambda x: -x[1])[:6]
            pred_red = [r for r, _ in top_reds]
            
            # 蓝球投票
            blue_counts = {}
            for b in all_blue_votes:
                blue_counts[b] = blue_counts.get(b, 0) + 1
            pred_blue = max(blue_counts.items(), key=lambda x: x[1])[0]
        else:
            # Fallback：使用热号
            pred_red = [2, 3, 6, 9, 10, 13]
            pred_blue = 4

        # ---- 计算中奖 ----
        red_hit = len(set(real_red) & set(pred_red))
        blue_hit = (int(real_blue) == pred_blue)

        if red_hit == 6 and blue_hit:
            level = "一等奖"
        elif red_hit == 6:
            level = "二等奖（红球全中）"
        elif red_hit == 5 and blue_hit:
            level = "三等奖"
        elif red_hit == 5 or (red_hit == 4 and blue_hit):
            level = "四等奖"
        elif red_hit == 4 or (red_hit == 3 and blue_hit):
            level = "五等奖"
        elif red_hit == 3 and blue_hit:
            level = "四等奖（红球3+蓝球）"
        elif red_hit == 2 and blue_hit:
            level = "六等奖（红球2+蓝球1）"
        elif red_hit == 1 and blue_hit:
            level = "六等奖（红球1+蓝球1）"
        elif blue_hit:
            level = "六等奖（仅蓝球）"
        else:
            level = "未中奖"

        results.append({
            'period': int(period),
            'real_red': real_red.tolist(),
            'real_blue': int(real_blue),
            'pred_red': sorted(pred_red),
            'pred_blue': int(pred_blue),
            'red_hit': int(red_hit),
            'blue_hit': bool(blue_hit),
            'level': level,
        })

        if i < 10 or i % 50 == 0:
            print(f"期号 {period}: 红球命中{red_hit}个, 蓝球{'✓' if blue_hit else '✗'}, {level}")

    # ---- 统计 ----
    total_red_hits = {str(i): 0 for i in range(7)}
    for r in results:
        total_red_hits[str(r['red_hit'])] += 1
    
    blue_hit_count = sum(1 for r in results if r['blue_hit'])
    blue_hit_rate = blue_hit_count / total * 100

    # 奖项统计
    award_stats = {}
    for r in results:
        level = r['level']
        award_stats[level] = award_stats.get(level, 0) + 1

    # 关键指标
    hit_3plus = sum(total_red_hits[str(i)] for i in range(3, 7))
    hit_4plus = sum(total_red_hits[str(i)] for i in range(4, 7))
    hit_5plus = sum(total_red_hits[str(i)] for i in range(5, 7))
    avg_hit = sum(int(r['red_hit']) for r in results) / total

    key_metrics = {
        'red_avg_hit': round(avg_hit, 2),
        'hit_3plus_rate': round(hit_3plus / total * 100, 1),
        'hit_4plus_rate': round(hit_4plus / total * 100, 1),
        'hit_5plus_rate': round(hit_5plus / total * 100, 1),
        'blue_hit_rate': round(blue_hit_rate, 1),
        'comprehensive_accuracy': round((avg_hit / 6 * 50 + blue_hit_rate / 100 * 50), 1),
    }

    backtest_result = {
        'total_periods': total,
        'red_hit_counts': total_red_hits,
        'blue_hit_count': blue_hit_count,
        'blue_hit_rate': round(blue_hit_rate, 1),
        'award_stats': award_stats,
        'key_metrics': key_metrics,
        'period_results': results,
    }

    # 保存
    output_path = os.path.join(os.path.dirname(model_path_prefix), "backtest_results_v2.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(backtest_result, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 回测结果已保存: {output_path}")
    print(f"\n📊 回测结果摘要：")
    print(f"   红球平均命中: {key_metrics['red_avg_hit']}个")
    print(f"   红球3+命中率: {key_metrics['hit_3plus_rate']}%")
    print(f"   红球4+命中率: {key_metrics['hit_4plus_rate']}%")
    print(f"   蓝球命中率: {key_metrics['blue_hit_rate']}%")
    print(f"   综合准确率: {key_metrics['comprehensive_accuracy']}%")

    return backtest_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="lottery_data.csv")
    parser.add_argument("--model", type=str, default="output/lottery_gan_v2")
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    backtest(args.csv, args.model, args.periods, args.samples, args.device)
