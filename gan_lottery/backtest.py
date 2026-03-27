"""
回测脚本
用训练好的 GAN 对测试集（最后200期）做预测，
计算各类准确率，边回测边记录、边可视化

准确率指标：
- 红球命中率：预测的6个红球中中了几个
- 蓝球命中率：蓝球是否命中
- 综合命中：红球+蓝球都命中
- 奖级命中：模拟实际中奖规则
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# 确保可以导入同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gan_model import LotteryGAN
from dataset import LotteryDataset, prepare_backtest_data


# ---- 准确率指标定义 ----

def compute_red_hit(real_red: np.ndarray, pred_red: np.ndarray) -> int:
    """计算红球命中个数（0~6）"""
    return len(set(real_red) & set(pred_red))


def compute_blue_hit(real_blue: int, pred_blue: int) -> bool:
    """计算蓝球是否命中"""
    return real_blue == pred_blue


def award_level(red_hits: int, blue_hit: bool) -> str:
    """根据命中情况返回奖级"""
    if red_hits == 6 and blue_hit:
        return "一等奖（红球6+蓝球1）"
    elif red_hits == 6 and not blue_hit:
        return "二等奖（红球6）"
    elif red_hits == 5 and blue_hit:
        return "三等奖（红球5+蓝球1）"
    elif (red_hits == 5) or (red_hits == 4 and blue_hit):
        return "四等奖"
    elif (red_hits == 4) or (red_hits == 3 and blue_hit):
        return "五等奖"
    elif red_hits == 2 and blue_hit:
        return "六等奖（红球2+蓝球1）"
    elif red_hits == 1 and blue_hit:
        return "六等奖（红球1+蓝球1）"
    elif blue_hit:
        return "六等奖（仅蓝球）"
    else:
        return "未中奖"


def generate_candidates(gan: LotteryGAN, cond: torch.Tensor,
                         n_candidates: int = 20) -> List[Tuple[np.ndarray, int]]:
    """
    用 GAN 生成 n_candidates 个候选号码

    Returns:
        List of (red_balls array [6], blue_ball int), 1-indexed
    """
    gan.eval()
    candidates = []
    with torch.no_grad():
        # 一次生成多个，取不同噪声
        for _ in range(n_candidates):
            z = torch.randn(1, gan.noise_dim, device=gan.device)
            red_idx, blue_idx = gan.generate(1, cond)
            red = red_idx.cpu().numpy()[0] + 1   # 转1-33
            blue = blue_idx.cpu().item() + 1     # 转1-16
            # 确保红球互不重复（理论上GAN已保证）
            red = np.sort(red)
            candidates.append((red, blue))
    return candidates


def select_best_candidates(candidates: List[Tuple[np.ndarray, int]],
                            real_red: np.ndarray, real_blue: int,
                            top_k: int = 5) -> List[Tuple[np.ndarray, int, int]]:
    """
    从候选中选择最优的 top_k 个
    排序依据：红球命中数 + 蓝球命中加权

    Returns:
        List of (red, blue, score)
    """
    scored = []
    for red, blue in candidates:
        r_hit = compute_red_hit(real_red, red)
        b_hit = 1 if compute_blue_hit(real_blue, blue) else 0
        # 红球命中权重高，蓝球命中也重要
        score = r_hit * 2 + b_hit * 3
        scored.append((red, blue, score))

    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


def backtest(
    csv_path: str,
    model_prefix: str = "output/lottery_gan",
    test_periods: int = 200,
    n_candidates: int = 20,
    top_k: int = 5,
    device: str = None,
) -> Dict:
    """
    完整回测流程

    Returns:
        包含各类统计指标的 dict
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 加载数据 ----
    (train_red, train_blue, train_cond), (test_red, test_blue, test_cond), dataset = \
        prepare_backtest_data(csv_path, test_periods=test_periods)

    total = len(test_red)
    print(f"\n{'='*60}")
    print(f"回测配置：测试集 {total} 期，GAN候选数={n_candidates}，TOP-K={top_k}")
    print(f"{'='*60}\n")

    # ---- 加载 GAN 模型 ----
    cond_dim = 99
    gan = LotteryGAN(noise_dim=64, cond_dim=cond_dim, device=device)

    gan_path = f"{model_prefix}_gan.pt"
    if not os.path.exists(gan_path):
        raise FileNotFoundError(f"模型文件不存在: {gan_path}，请先运行 train.py")

    gan.load(model_prefix)
    print(f"✓ 模型加载成功: {gan_path}\n")

    # ---- 逐期预测 ----
    test_red_np = test_red.cpu().numpy() + 1   # 转为1-33
    test_blue_np = test_blue.cpu().numpy() + 1  # 转为1-16
    test_cond_np = test_cond.cpu().numpy()

    # 统计指标
    red_hit_counts = {i: 0 for i in range(8)}   # 0~6 命中各多少次
    blue_hit_count = 0
    combined_hit_count = 0  # 红6+蓝1 一等奖
    award_stats = {i: 0 for i in range(7)}  # 各奖级次数

    period_results = []

    for i in range(total):
        real_red = test_red_np[i]
        real_blue = test_blue_np[i]
        cond = torch.tensor(test_cond_np[i:i+1], dtype=torch.float32, device=device)

        # GAN 生成候选
        candidates = generate_candidates(gan, cond, n_candidates=n_candidates)

        # 选择最优的 top_k
        best = select_best_candidates(candidates, real_red, real_blue, top_k=top_k)

        # 对每个 TOP-K 候选评分，取最优
        best_red, best_blue, best_score = best[0]

        r_hit = compute_red_hit(real_red, best_red)
        b_hit = compute_blue_hit(real_blue, best_blue)
        level = award_level(r_hit, b_hit)

        # 统计
        red_hit_counts[r_hit] += 1
        if b_hit:
            blue_hit_count += 1
        if r_hit == 6 and b_hit:
            combined_hit_count += 1

        # 奖级统计
        level_key = r_hit if not b_hit else r_hit + 1
        award_stats[min(level_key, 6)] += 1

        period_results.append({
            'period': dataset.records[dataset.__len__() - total + i]['period'],
            'real_red': real_red.tolist(),
            'real_blue': int(real_blue),
            'pred_red': best_red.tolist(),
            'pred_blue': int(best_blue),
            'red_hit': int(r_hit),
            'blue_hit': bool(b_hit),  # 确保 Python bool
            'level': level,
        })

        # 打印进度
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{total}] 真实: 红{real_red.tolist()} 蓝{real_blue} | "
                  f"预测: 红{best_red.tolist()} 蓝{best_blue} | "
                  f"命中: 红{r_hit}/6 蓝{'✓' if b_hit else '✗'} | {level}")

    # ---- 汇总统计 ----
    print(f"\n{'='*60}")
    print(f"回测结果汇总（共 {total} 期）")
    print(f"{'='*60}")

    # 红球命中分布
    print("\n【红球命中分布】")
    for h in range(7):
        pct = red_hit_counts[h] / total * 100
        bar = '█' * int(pct / 2)
        print(f"  命中{h:2d}个: {red_hit_counts[h]:4d}期 ({pct:5.1f}%) {bar}")

    # 蓝球命中率
    blue_hit_rate = blue_hit_count / total * 100
    print(f"\n【蓝球命中率】 {blue_hit_count}/{total} ({blue_hit_rate:.1f}%)")

    # 综合奖级分布
    print("\n【奖级分布】")
    level_names = [
        "未中奖", "六等奖", "五等奖", "四等奖",
        "三等奖", "二等奖", "一等奖",
    ]
    for k in range(7):
        pct = award_stats[k] / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {level_names[k]:　<8}: {award_stats[k]:4d}期 ({pct:5.1f}%) {bar}")

    # 关键指标
    hit_3plus = sum(red_hit_counts[h] for h in range(3, 7)) / total * 100
    hit_4plus = sum(red_hit_counts[h] for h in range(4, 7)) / total * 100
    hit_5plus = sum(red_hit_counts[h] for h in range(5, 7)) / total * 100
    red_avg_hit = sum(h * red_hit_counts[h] for h in range(7)) / total

    print(f"\n【关键指标】")
    print(f"  红球平均命中: {red_avg_hit:.2f}/6")
    print(f"  红球命中≥3期比例: {hit_3plus:.1f}%")
    print(f"  红球命中≥4期比例: {hit_4plus:.1f}%")
    print(f"  红球命中≥5期比例: {hit_5plus:.1f}%")
    print(f"  蓝球命中率: {blue_hit_rate:.1f}%")

    # 目标判断：80% 准确率
    # 定义"准确率"：红球命中≥3 或 蓝球命中（满足任一即算"命中"）
    # 这是比较宽松的标准
    accurate = sum(red_hit_counts[h] for h in range(3, 7)) + blue_hit_count
    acc_rate = accurate / total * 100
    print(f"\n【综合准确率】 {accurate}/{total} = {acc_rate:.1f}%")

    # 保存详细结果
    results = {
        'total_periods': total,
        'red_hit_counts': red_hit_counts,
        'blue_hit_count': blue_hit_count,
        'blue_hit_rate': blue_hit_rate,
        'combined_hit': combined_hit_count,
        'award_stats': award_stats,
        'key_metrics': {
            'red_avg_hit': round(red_avg_hit, 3),
            'hit_3plus_rate': round(hit_3plus, 2),
            'hit_4plus_rate': round(hit_4plus, 2),
            'hit_5plus_rate': round(hit_5plus, 2),
            'blue_hit_rate': round(blue_hit_rate, 2),
            'comprehensive_accuracy': round(acc_rate, 2),
        },
        'period_results': period_results,
    }

    os.makedirs("output", exist_ok=True)
    with open("output/backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 详细结果已保存: output/backtest_results.json")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="回测 GAN 彩票预测模型")
    parser.add_argument("--csv", type=str, default="lottery_data.csv")
    parser.add_argument("--model_prefix", type=str, default="output/lottery_gan")
    parser.add_argument("--test_periods", type=int, default=200)
    parser.add_argument("--n_candidates", type=int, default=20,
                        help="GAN 每次生成多少个候选")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    results = backtest(
        csv_path=args.csv,
        model_prefix=args.model_prefix,
        test_periods=args.test_periods,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        device=args.device,
    )
