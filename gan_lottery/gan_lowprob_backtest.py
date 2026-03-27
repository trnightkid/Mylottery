"""
GAN 二次筛选 + 回测分析
========================================
核心思路（用户定义）：
  GAN 的低概率号码 = 模型认为"最不应该出现"的号码
  这些号码并非要被排除，而是要单独分析它们的历史实际表现
  规律：低概率号码如果大量命中 = 说明出号有反规律可循

二次筛选流程：
  1. GAN 生成 N 个候选，按低概率得分排序
  2. 提取低/中/高三档候选
  3. 结合 PageRank 打分做二次筛选
  4. 输出：各档位的实际命中率、各档位+PageRank组合的效果
"""

import sys, torch, json, os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gan_model import LotteryGAN
from dataset import LotteryDataset


def score_by_low_probability(candidates_red: np.ndarray, ds: LotteryDataset) -> np.ndarray:
    """
    对每个候选红球号码打分（越低=越低概率）
    得分 = 候选内所有号码的历史频率之和（越低越"不应该出"）
    返回每个候选的得分，排序后低分档在前面
    """
    scores = []
    for reds in candidates_red:  # reds: (6,) array of ball numbers 1-33
        # 频率求和，越低表示越"不该出"
        score = sum(ds.red_freq[r - 1] for r in reds)
        scores.append(score)
    return np.array(scores)


def gan_backtest_full(ds: LotteryDataset, gan: LotteryGAN,
                      n_candidates: int = 50,
                      test_start: int = None,
                      test_end: int = None) -> dict:
    """
    完整 GAN 回测 + 低概率号码分析
    返回详细统计数据
    """
    total = len(ds.records)
    if test_start is None:
        test_start = max(0, total - 200)
    if test_end is None:
        test_end = total

    results = {
        'test_periods': test_end - test_start,
        'gan': {},
        'low_prob': {},    # 低概率号码分析
        'award': {},       # 奖级分布
        'by_bin': {},      # 按概率分档的命中率
    }

    # ---- 统计变量 ----
    # GAN 第1候选
    gan1_red_hits = []
    gan1_blue_hits = []
    gan1_comprehensive = []

    # 低/中/高三档（各取10个候选）
    low_red_hits = []
    mid_red_hits = []
    high_red_hits = []
    low_blue_hits = []
    mid_blue_hits = []
    high_blue_hits = []

    # 低概率号码中实际命中的统计
    low_hit_count = []  # 低概率候选里命中的个数

    period_results = []

    for i in range(test_start, test_end):
        period = ds.records[i]['period']
        real_red = np.array([r for r in ds.records[i]['red']])
        real_blue = ds.records[i]['blue']

        cond = ds.get_condition_vector(i, window=200)
        cond_t = torch.tensor(cond, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            candidates_red = []
            candidates_blue = []
            for _ in range(n_candidates):
                z = torch.randn(1, 64)
                r_idx, b_idx = gan.generate(1, cond_t)
                candidates_red.append(r_idx.cpu().numpy()[0] + 1)
                candidates_blue.append(b_idx.cpu().item() + 1)

        candidates_red = np.array(candidates_red)  # (N, 6)
        candidates_blue = np.array(candidates_blue)

        # ---- GAN 第1候选 ----
        pred1_red = candidates_red[0]
        pred1_blue = candidates_blue[0]
        r_hit1 = len(set(real_red) & set(pred1_red))
        b_hit1 = (real_blue == pred1_blue)
        gan1_red_hits.append(r_hit1)
        gan1_blue_hits.append(b_hit1)
        gan1_comprehensive.append(r_hit1 >= 3 or b_hit1)

        # ---- 低概率分档 ----
        lp_scores = score_by_low_probability(candidates_red, ds)
        sorted_idx = np.argsort(lp_scores)  # 低分在前=低概率

        low_idx = sorted_idx[:10]      # 最低概率的10个
        mid_idx = sorted_idx[20:30]    # 中间概率的10个
        high_idx = sorted_idx[-10:]   # 最高概率的10个（相对"该出的"）

        # 每档各选第1名
        low_red_hits.append(len(set(real_red) & set(candidates_red[low_idx[0]])))
        mid_red_hits.append(len(set(real_red) & set(candidates_red[mid_idx[0]])))
        high_red_hits.append(len(set(real_red) & set(candidates_red[high_idx[0]])))
        low_blue_hits.append(real_blue in candidates_blue[low_idx])
        mid_blue_hits.append(real_blue in candidates_blue[mid_idx])
        high_blue_hits.append(real_blue in candidates_blue[high_idx])

        # 低概率候选里命中了几个
        low_hit = sum(1 for idx in low_idx if len(set(real_red) & set(candidates_red[idx])) >= 1)
        low_hit_count.append(low_hit)

        # 综合各档位（只看低概率的第1名）
        low_comp = low_red_hits[-1] >= 3 or low_blue_hits[-1]
        mid_comp = mid_red_hits[-1] >= 3 or mid_blue_hits[-1]
        high_comp = high_red_hits[-1] >= 3 or high_blue_hits[-1]

        period_results.append({
            'period': int(period),
            'real_red': real_red.tolist(),
            'real_blue': int(real_blue),
            'gan1_pred_red': pred1_red.tolist(),
            'gan1_pred_blue': int(pred1_blue),
            'gan1_red_hit': int(r_hit1),
            'gan1_blue_hit': bool(b_hit1),
            'gan1_comprehensive': bool(gan1_comprehensive[-1]),
            'low_prob_red_hit': int(low_red_hits[-1]),
            'low_prob_blue_hit': bool(low_blue_hits[-1]),
            'low_prob_comprehensive': bool(low_comp),
            'low_bin_avg_hit': round(np.mean([len(set(real_red) & set(candidates_red[idx])) for idx in low_idx]), 2),
        })

    # ---- 汇总统计 ----
    n = len(gan1_red_hits)
    comp_all = sum(gan1_comprehensive)
    comp_low = sum(low_comp for lp in period_results)
    comp_mid = sum(mid_comp for lp in period_results)
    comp_high = sum(high_comp for lp in period_results)

    print(f"\n{'='*60}")
    print(f"  GAN 回测 + 低概率分析（{n}期）")
    print(f"{'='*60}")

    print(f"\n【GAN 第1候选 vs 低概率档】")
    print(f"{'指标':<20} {'GAN第1':>8} {'低概率':>8} {'中等':>8} {'高概率':>8}")
    print(f"{'-'*60}")
    r_gan1 = np.mean(gan1_red_hits)
    r_low = np.mean(low_red_hits)
    r_mid = np.mean(mid_red_hits)
    r_high = np.mean(high_red_hits)
    print(f"{'红球均命中':<20} {r_gan1:>8.2f} {r_low:>8.2f} {r_mid:>8.2f} {r_high:>8.2f}")

    b_gan1 = np.mean(gan1_blue_hits) * 100
    b_low = np.mean(low_blue_hits) * 100
    b_mid = np.mean(mid_blue_hits) * 100
    b_high = np.mean(high_blue_hits) * 100
    print(f"{'蓝球命中率':<20} {b_gan1:>7.1f}% {b_low:>7.1f}% {b_mid:>7.1f}% {b_high:>7.1f}%")

    print(f"{'综合准确率(红≥3/蓝)':<20} {comp_all/n*100:>7.1f}% {comp_low/n*100:>7.1f}% {comp_mid/n*100:>7.1f}% {comp_high/n*100:>7.1f}%")

    # 低概率号码里命中的平均个数
    print(f"\n【低概率候选分析】")
    avg_low_hit = np.mean(low_hit_count)
    print(f"  低概率档(10个候选)中平均命中: {avg_low_hit:.2f} 个号码")
    print(f"  说明: 如果 avg < 1，说明低概率号码普遍不中")

    # 红球各档位命中分布
    print(f"\n【红球命中分布 - 各档位】")
    for label, hits_list in [('低概率', low_red_hits), ('中等', mid_red_hits), ('高概率', high_red_hits)]:
        h = np.array(hits_list)
        print(f"  {label}:", '  '.join(f"命中{i}个:{((h==i).sum()/n*100):.1f}%" for i in range(7)))

    # GAN 第1候选完整奖级
    award_counts = {i: 0 for i in range(7)}
    award_names = ['未中奖', '六等奖', '五等奖', '四等奖', '三等奖', '二等奖', '一等奖']
    for i, pr in enumerate(period_results):
        r_h = pr['gan1_red_hit']
        b_h = 1 if pr['gan1_blue_hit'] else 0
        if r_h == 6 and b_h: award_counts[6] += 1
        elif r_h == 6: award_counts[5] += 1
        elif r_h == 5 and b_h: award_counts[4] += 1
        elif r_h == 5 or (r_h == 4 and b_h): award_counts[3] += 1
        elif r_h == 4 or (r_h == 3 and b_h): award_counts[2] += 1
        elif r_h == 2 and b_h: award_counts[1] += 1
        elif r_h <= 1 and b_h: award_counts[1] += 1
        else: award_counts[0] += 1

    print(f"\n【奖级分布 - GAN第1候选】")
    for k, name in enumerate(award_names):
        pct = award_counts[k] / n * 100
        print(f"  {name}: {award_counts[k]:3d}期 ({pct:5.1f}%) {'█' * int(pct/2)}")

    # 保存
    out = {
        'test_periods': n,
        'gan1': {
            'red_avg': round(float(np.mean(gan1_red_hits)), 3),
            'blue_hit_rate': round(float(np.mean(gan1_blue_hits)) * 100, 2),
            'red_ge3_rate': round(sum(1 for h in gan1_red_hits if h >= 3) / n * 100, 2),
            'comprehensive': round(comp_all / n * 100, 2),
        },
        'by_bin': {
            'low': {
                'red_avg': round(float(np.mean(low_red_hits)), 3),
                'blue_hit_rate': round(float(np.mean(low_blue_hits)) * 100, 2),
                'comprehensive': round(comp_low / n * 100, 2),
            },
            'mid': {
                'red_avg': round(float(np.mean(mid_red_hits)), 3),
                'blue_hit_rate': round(float(np.mean(mid_blue_hits)) * 100, 2),
                'comprehensive': round(comp_mid / n * 100, 2),
            },
            'high': {
                'red_avg': round(float(np.mean(high_red_hits)), 3),
                'blue_hit_rate': round(float(np.mean(high_blue_hits)) * 100, 2),
                'comprehensive': round(comp_high / n * 100, 2),
            },
        },
        'low_prob_avg_hit_in_bin': round(float(np.mean(low_hit_count)), 2),
        'award_counts_gan1': award_counts,
        'award_names': award_names,
        'period_results': period_results,
    }

    os.makedirs('output', exist_ok=True)
    with open('output/gan_lowprob_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 详细结果已保存: output/gan_lowprob_analysis.json")
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='lottery_data.csv')
    parser.add_argument('--model_prefix', default='output/lottery_gan')
    parser.add_argument('--test_periods', type=int, default=200)
    parser.add_argument('--n_candidates', type=int, default=50)
    args = parser.parse_args()

    ds = LotteryDataset(args.csv)
    gan = LotteryGAN(noise_dim=64, cond_dim=99, device='cpu')
    gan.load(args.model_prefix)
    gan.eval()

    total = len(ds.records)
    test_start = total - args.test_periods

    print(f"数据: {total}期, 测试: {test_start}~{total-1} ({args.test_periods}期)")
    print(f"GAN候选数: {args.n_candidates}")

    out = gan_backtest_full(
        ds, gan,
        n_candidates=args.n_candidates,
        test_start=test_start,
        test_end=total,
    )
