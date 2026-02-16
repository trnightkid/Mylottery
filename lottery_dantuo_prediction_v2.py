"""
双色球预测分析 - 胆拖投注预测优化版（蒙特卡洛采样 + 精英选择）
"""
import pymysql
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============== 配置区域 ==============
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'reven@0504',
    'database': 'lottery_db',
    'charset': 'utf8mb4'
}

OUTPUT_DIR = r"D:\Mydevelopment\MultiContentProject\Mylottery\dan_tuo_prediction"

# ============== 胆拖参数 ==============
N_DAN_RED = 4      # 红球胆码数量
N_TUO_HOT_RED = 3  # 红球拖码数量（高拟合区）
N_TUO_COLD_RED = 2 # 红球拖码数量（冷门区）

N_DAN_BLUE = 1     # 蓝球胆码数量
N_TUO_BLUE = 2     # 蓝球拖码数量

# ============== 优化参数 ==============
N_SAMPLES = 500        # 蒙特卡洛采样次数
N_TOP_SELECT = 10      # 最终选出TOP多少组
N_DISPLAY = 5          # 控制台显示前多少组


def load_data():
    """从数据库加载数据"""
    print("📥 正在从数据库加载数据...")

    conn = pymysql.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date
        FROM lottery_data 
        ORDER BY CAST(period AS UNSIGNED)
    """, conn)
    conn.close()

    print(f"   ✅ 加载完成: {len(df)} 条记录")
    print(f"   📅 {df['draw_date'].min()} ~ {df['draw_date'].max()}")

    return df


def fit_kde(data, x):
    """核密度估计"""
    kde = stats.gaussian_kde(data)
    pdf = kde(x)
    return pdf / pdf.sum()


def fit_beta(data, x):
    """Beta分布拟合"""
    normalized = (data - 1) / 32

    try:
        a, b, loc, scale = stats.beta.fit(normalized, floc=0, fscale=1)
        x_norm = (x - 1) / 32
        pdf = stats.beta.pdf(x_norm, a, b)
        return pdf / pdf.sum()
    except:
        return np.ones_like(x) / len(x)


def fit_trimodal(data, x):
    """手动实现三峰高斯混合模型"""
    n_samples = len(data)
    x = np.array(x)

    quantiles = np.percentile(data, [100 / (3 + 1) * i for i in range(1, 4)])
    means = sorted(quantiles.copy())
    stds = [np.std(data) / np.sqrt(3)] * 3
    weights = np.ones(3) / 3

    for _ in range(50):
        responsibilities = np.zeros((n_samples, 3))
        for k in range(3):
            responsibilities[:, k] = weights[k] * stats.norm.pdf(data, means[k], stds[k] + 0.1)
        responsibilities = responsibilities / (responsibilities.sum(axis=1, keepdims=True) + 1e-10)

        for k in range(3):
            nk = responsibilities[:, k].sum()
            weights[k] = nk / n_samples
            means[k] = (responsibilities[:, k] * data).sum() / (nk + 1e-10)
            var = (responsibilities[:, k] * (data - means[k]) ** 2).sum() / (nk + 1e-10)
            stds[k] = np.sqrt(var + 0.1)

    pdf = np.zeros_like(x)
    for k in range(3):
        pdf += weights[k] * stats.norm.pdf(x, means[k], stds[k] + 0.1)

    return pdf / (pdf.sum() + 1e-10)


def fit_distributions(df):
    """拟合分布，返回综合概率"""
    print("\n📊 正在拟合分布...")

    red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
    all_reds = np.array([df[col].values for col in red_cols]).flatten()

    x = np.linspace(1, 33, 1000)

    print("   1/4 核密度估计...")
    pdf_kde = fit_kde(all_reds, x)

    print("   2/4 Beta分布...")
    pdf_beta = fit_beta(all_reds, x)

    print("   3/4 三峰高斯混合...")
    pdf_gmm = fit_trimodal(all_reds, x)

    print("   4/4 频率分析...")
    freq_counts = Counter(all_reds)
    pdf_freq = np.zeros_like(x)
    for num in range(1, 34):
        idx = np.abs(x - num).argmin()
        pdf_freq[idx] = freq_counts.get(num, 0)
    pdf_freq = pdf_freq / (pdf_freq.sum() + 1e-10)

    print("   综合权重计算...")
    combined = (
            0.30 * pdf_kde +
            0.25 * pdf_beta +
            0.25 * pdf_gmm +
            0.20 * pdf_freq
    )
    combined = combined / combined.sum()

    red_probs = {}
    for num in range(1, 34):
        idx = np.abs(x - num).argmin()
        red_probs[num] = combined[idx]

    total = sum(red_probs.values())
    for num in red_probs:
        red_probs[num] /= total

    return red_probs, x, combined


def calculate_blue_probs(df):
    """计算蓝球概率（带拉普拉斯平滑）"""
    blue_counts = Counter(df['blue'].tolist())
    total = len(df)

    blue_probs = {}
    for num in range(1, 17):
        count = blue_counts.get(num, 0)
        blue_probs[num] = (count + 1) / (total + 16)

    return blue_probs


def weighted_random_choice(probs_dict, n, exclude=None):
    """根据权重随机抽取n个数字"""
    if exclude is None:
        exclude = set()

    items = [(k, v) for k, v in probs_dict.items() if k not in exclude]
    nums = [k for k, v in items]
    weights = np.array([v for k, v in items])

    if weights.sum() == 0:
        weights = np.ones(len(nums))

    weights = weights / weights.sum()

    selected = np.random.choice(nums, size=min(n, len(nums)), replace=False, p=weights)

    return list(selected)


def select_cold_numbers(probs_dict, n, exclude=None):
    """选取最冷门的n个数字"""
    if exclude is None:
        exclude = set()

    sorted_items = sorted(probs_dict.items(), key=lambda x: x[1])

    selected = []
    for num, prob in sorted_items:
        if num not in exclude:
            selected.append((num, prob))
            if len(selected) >= n:
                break

    return selected


def select_hot_numbers(probs_dict, n, exclude=None):
    """选取最热门的n个数字"""
    if exclude is None:
        exclude = set()

    sorted_items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

    selected = []
    for num, prob in sorted_items:
        if num not in exclude:
            selected.append((num, prob))
            if len(selected) >= n:
                break

    return selected


def build_red_dantuo_pool(red_probs, n_dan=4, n_tuo_hot=3, n_tuo_cold=2, seed=None):
    """构建红球胆拖号码池"""
    if seed is not None:
        np.random.seed(seed)

    hot_candidates = select_hot_numbers(red_probs, 12)
    hot_nums = [n for n, p in hot_candidates]
    hot_probs = {n: red_probs[n] for n in hot_nums}

    dan = weighted_random_choice(hot_probs, n_dan)
    dan_set = set(dan)

    tuo_hot = weighted_random_choice(hot_probs, n_tuo_hot, exclude=dan_set)

    cold_candidates = select_cold_numbers(red_probs, 12, exclude=dan_set)
    cold_nums = [n for n, p in cold_candidates]
    tuo_cold = weighted_random_choice({n: red_probs[n] for n in cold_nums}, n_tuo_cold)

    tuo = tuo_hot + tuo_cold
    tuo_set = set(tuo)

    dan = [d for d in dan if d not in tuo_set]
    tuo = [t for t in tuo if t not in dan_set]

    full_pool = sorted(dan + tuo)

    return dan, tuo, full_pool


def build_blue_dantuo_pool(blue_probs, n_dan=1, n_tuo=2, seed=None):
    """构建蓝球胆拖号码池"""
    if seed is not None:
        np.random.seed(seed)

    dan = weighted_random_choice(blue_probs, n_dan)
    dan_set = set(dan)

    tuo = weighted_random_choice(blue_probs, n_tuo, exclude=dan_set)

    full_pool = dan + tuo

    return dan, tuo, full_pool


def calculate_pool_statistics(dan, tuo, probs, total_nums):
    """计算号码池的统计特性"""
    expected_hits = sum(probs[n] for n in dan + tuo)
    dan_prob = sum(probs[n] for n in dan)
    tuo_prob = sum(probs[n] for n in tuo)
    theoretical = (len(dan) + len(tuo)) / total_nums
    coverage = expected_hits / theoretical if theoretical > 0 else 0

    return {
        'expected_hits': expected_hits,
        'dan_prob': dan_prob,
        'tuo_prob': tuo_prob,
        'coverage': coverage,
        'total_count': len(dan) + len(tuo)
    }


def monte_carlo_sampling(red_probs, blue_probs, n_samples=500):
    """蒙特卡洛采样 - 生成大量候选预测组"""
    all_predictions = []

    for i in range(n_samples):
        dan_red, tuo_red, full_pool_red = build_red_dantuo_pool(
            red_probs,
            n_dan=N_DAN_RED,
            n_tuo_hot=N_TUO_HOT_RED,
            n_tuo_cold=N_TUO_COLD_RED,
            seed=None  # 完全随机
        )

        dan_blue, tuo_blue, full_pool_blue = build_blue_dantuo_pool(
            blue_probs,
            n_dan=N_DAN_BLUE,
            n_tuo=N_TUO_BLUE,
            seed=None
        )

        red_stats = calculate_pool_statistics(dan_red, tuo_red, red_probs, 33)
        blue_stats = calculate_pool_statistics(dan_blue, tuo_blue, blue_probs, 16)

        prediction = {
            'sample_id': i,
            'dan_red': sorted(dan_red),
            'tuo_red': sorted(tuo_red),
            'red_pool': sorted(dan_red + tuo_red),
            'dan_blue': sorted(dan_blue),
            'tuo_blue': sorted(tuo_blue),
            'blue_pool': sorted(dan_blue + tuo_blue),
            'red_stats': red_stats,
            'blue_stats': blue_stats,
            'total_expected': red_stats['expected_hits'] + blue_stats['expected_hits']
        }
        all_predictions.append(prediction)

    return all_predictions


def elite_selection(all_predictions, top_k=10):
    """精英选择 - 从所有采样中挑选最优的top_k组"""
    sorted_predictions = sorted(
        all_predictions,
        key=lambda x: x['total_expected'],
        reverse=True
    )
    return sorted_predictions[:top_k]


def generate_optimized_predictions(df, n_samples=500, top_k=10):
    """生成优化后的预测结果"""
    print("\n" + "=" * 70)
    print("🎯 胆拖投注预测 - 优化版（蒙特卡洛采样 + 精英选择）")
    print("=" * 70)

    red_probs, x, pdf = fit_distributions(df)
    blue_probs = calculate_blue_probs(df)

    print("\n📈 红球拟合概率TOP15:")
    top15 = select_hot_numbers(red_probs, 15)
    for i, (num, prob) in enumerate(top15, 1):
        deviation = (prob - 1 / 33) / (1 / 33) * 100
        sign = '+' if deviation > 0 else ''
        print(f"   {i:2d}. {num:02d}: {prob:.5f} ({sign}{deviation:.1f}%)")

    print("\n📈 蓝球拟合概率TOP8:")
    top8 = select_hot_numbers(blue_probs, 8)
    for i, (num, prob) in enumerate(top8, 1):
        deviation = (prob - 1 / 16) / (1 / 16) * 100
        sign = '+' if deviation > 0 else ''
        print(f"   {i:2d}. {num:02d}: {prob:.5f} ({sign}{deviation:.1f}%)")

    print("\n📉 红球冷门TOP5:")
    cold5 = select_cold_numbers(red_probs, 5)
    for i, (num, prob) in enumerate(cold5, 1):
        deviation = (prob - 1 / 33) / (1 / 33) * 100
        sign = '+' if deviation > 0 else ''
        print(f"   {i:2d}. {num:02d}: {prob:.5f} ({sign}{deviation:.1f}%)")

    # 蒙特卡洛采样
    print(f"\n🎲 开始蒙特卡洛采样 ({n_samples}次)...")
    all_samples = monte_carlo_sampling(red_probs, blue_probs, n_samples=n_samples)

    # 统计采样结果
    all_expected = [p['total_expected'] for p in all_samples]
    print(f"   📊 采样统计:")
    print(f"      期望命中 - 最高: {max(all_expected):.3f}, "
          f"平均: {np.mean(all_expected):.3f}, "
          f"最低: {min(all_expected):.3f}")

    # 精英选择
    print(f"\n🏆 执行精英选择 (TOP {top_k})...")
    top_predictions = elite_selection(all_samples, top_k)

    best = top_predictions[0]
    worst = top_predictions[-1]
    print(f"   最佳期望: {best['total_expected']:.3f}")
    print(f"   最差期望: {worst['total_expected']:.3f}")
    print(f"   优化幅度: +{(best['total_expected'] - np.mean(all_expected)):.3f} "
          f"(相比平均)")

    # 显示TOP组
    display_count = min(N_DISPLAY, len(top_predictions))
    print(f"\n📋 TOP {display_count} 预测号码:")
    print("-" * 70)

    for i, pred in enumerate(top_predictions[:display_count], 1):
        print(f"\n【预测{i}】 (样本#{pred['sample_id']})")
        print(f"  🟥 红球胆码 ({len(pred['dan_red'])}个): ", end="")
        print(", ".join([f"{n:02d}" for n in pred['dan_red']]))
        print(f"  🟨 红球拖码 ({len(pred['tuo_red'])}个): ", end="")
        print(", ".join([f"{n:02d}" for n in pred['tuo_red']]))
        print(f"     └─ 热区: {[f'{n:02d}' for n in sorted(pred['tuo_red'][:N_TUO_HOT_RED])]}", end=" ")
        print(f"+ 冷区: {[f'{n:02d}' for n in sorted(pred['tuo_red'][N_TUO_HOT_RED:])]}")
        print(f"  🔵 蓝球胆码 ({len(pred['dan_blue'])}个): ", end="")
        print(", ".join([f"{n:02d}" for n in pred['dan_blue']]))
        print(f"  🟦 蓝球拖码 ({len(pred['tuo_blue'])}个): ", end="")
        print(", ".join([f"{n:02d}" for n in pred['tuo_blue']]))
        print(f"  📊 覆盖红球: {len(pred['red_pool'])}个 | 蓝球: {len(pred['blue_pool'])}个")
        print(f"  📈 期望命中: 红{pred['red_stats']['expected_hits']:.2f} + "
              f"蓝{pred['blue_stats']['expected_hits']:.2f} = "
              f"总计{pred['total_expected']:.2f}")

    return top_predictions, all_samples, red_probs, blue_probs, x, pdf


def plot_optimized_charts(predictions, all_samples, red_probs, blue_probs, x_vals, pdf, output_dir):
    """绘制优化后的可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 图表1：综合分析
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1：红球概率分布
    ax1 = axes1[0, 0]
    nums = list(range(1, 34))
    probs = [red_probs[n] for n in nums]

    colors = []
    for n in nums:
        if n in predictions[0]['dan_red']:
            colors.append('#FF4444')
        elif n in predictions[0]['tuo_red']:
            colors.append('#FFA500')
        else:
            colors.append('#CCCCCC')

    ax1.bar(nums, probs, color=colors, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=1 / 33, color='blue', linestyle='--', alpha=0.5, label='均匀分布')
    ax1.set_xlabel('Red Ball Number', fontsize=11)
    ax1.set_ylabel('Fitted Probability', fontsize=11)
    ax1.set_title('Red Ball Fitted Probability Distribution (Best Prediction)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, 34, 2))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF4444', label='Dan (Key)'),
        Patch(facecolor='#FFA500', label='Tuo (Extended)'),
        Patch(facecolor='#CCCCCC', label='Not Selected')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # 子图2：TOP预测的胆拖分布
    ax2 = axes1[0, 1]

    dan_counts = np.zeros(34)
    tuo_counts = np.zeros(34)

    for pred in predictions:
        for n in pred['dan_red']:
            dan_counts[n - 1] += 1
        for n in pred['tuo_red']:
            tuo_counts[n - 1] += 1

    x_pos = np.arange(34)
    width = 0.6

    ax2.bar(x_pos, dan_counts, width, label='Dan Count', color='#FF4444')
    ax2.bar(x_pos, tuo_counts, width, bottom=dan_counts, label='Tuo Count', color='#FFA500')
    ax2.set_xlabel('Red Ball Number', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'Dan/Tuo Distribution in TOP {len(predictions)}', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(i) for i in range(1, 35)])
    ax2.legend()

    # 子图3：蒙特卡洛采样分布
    ax3 = axes1[1, 0]

    all_expected = [p['total_expected'] for p in all_samples]

    ax3.hist(all_expected, bins=30, alpha=0.7, color='steelblue',
             label=f'All Samples (n={len(all_samples)})', edgecolor='white')
    ax3.axvline(x=np.mean(all_expected), color='blue', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(all_expected):.3f}')
    ax3.axvline(x=max(all_expected), color='green', linestyle='--',
                linewidth=2, label=f'Max: {max(all_expected):.3f}')

    for pred in predictions:
        ax3.axvline(x=pred['total_expected'], color='red', alpha=0.3, linewidth=1)

    ax3.set_xlabel('Total Expected Hits', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Monte Carlo Sampling Distribution', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')

    # 子图4：期望命中数对比
    ax4 = axes1[1, 1]

    groups = [f'TOP {p["sample_id"] % 10 + 1}' for p in predictions]
    red_expected = [p['red_stats']['expected_hits'] for p in predictions]
    blue_expected = [p['blue_stats']['expected_hits'] for p in predictions]

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, red_expected, width, label='Red Expected Hits', color='#FF6B6B')
    bars2 = ax4.bar(x + width / 2, blue_expected, width, label='Blue Expected Hits', color='#4ECDC4')

    ax4.set_ylabel('Expected Hits', fontsize=11)
    ax4.set_title('Expected Hits by Prediction (TOP Selected)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(groups, fontsize=9)
    ax4.legend()
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimized_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ optimized_analysis.png")

    # 图表2：优化效果对比
    fig5, ax = plt.subplots(figsize=(10, 6))

    all_expected = [p['total_expected'] for p in all_samples]

    n, bins, patches = ax.hist(all_expected, bins=40, alpha=0.7,
                                color='steelblue', edgecolor='white')

    top_threshold = predictions[-1]['total_expected']
    for i, patch in enumerate(patches):
        if bins[i] >= top_threshold:
            patch.set_facecolor('#FF6B6B')
            patch.set_alpha(0.8)

    ax.axvline(x=np.mean(all_expected), color='blue', linestyle='--',
               linewidth=2, label=f'Overall Mean: {np.mean(all_expected):.3f}')
    ax.axvline(x=top_threshold, color='red', linestyle='-',
               linewidth=2, label=f'TOP Selection Threshold: {top_threshold:.3f}')

    ax.axvspan(top_threshold, max(all_expected) + 0.1, alpha=0.1, color='red')

    ax.set_xlabel('Total Expected Hits', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Monte Carlo Optimization Effect (n={len(all_samples)}, TOP {len(predictions)})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    stats_text = (
        f"Overall Statistics:\n"
        f"Mean: {np.mean(all_expected):.3f}\n"
        f"Std: {np.std(all_expected):.3f}\n"
        f"Min: {min(all_expected):.3f}\n"
        f"Max: {max(all_expected):.3f}\n"
        f"\nTOP {len(predictions)} Selected:\n"
        f"Best: {predictions[0]['total_expected']:.3f}\n"
        f"Threshold: {top_threshold:.3f}\n"
        f"Improvement: +{(predictions[0]['total_expected'] - np.mean(all_expected)):.3f}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimization_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ optimization_effect.png")


def save_optimized_results(predictions, all_samples, red_probs, blue_probs, output_dir):
    """保存优化后的预测结果"""
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/optimized_predictions.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("         双色球胆拖投注预测结果 - 优化版\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"采样次数: {len(all_samples)}\n")
        f.write(f"选出组数: {len(predictions)}\n\n")

        # 采样统计
        all_expected = [p['total_expected'] for p in all_samples]
        f.write("一、采样统计\n")
        f.write("-" * 70 + "\n")
        f.write(f"  期望命中 - 最高: {max(all_expected):.3f}\n")
        f.write(f"           平均: {np.mean(all_expected):.3f}\n")
        f.write(f"           最低: {min(all_expected):.3f}\n")
        f.write(f"           标准差: {np.std(all_expected):.3f}\n\n")

        f.write("二、胆拖策略说明\n")
        f.write("-" * 70 + "\n")
        f.write(f"  红球：胆{N_DAN_RED}个 + 拖{N_TUO_HOT_RED}个(热区) + 拖{N_TUO_COLD_RED}个(冷区)\n")
        f.write(f"  蓝球：胆{N_DAN_BLUE}个 + 拖{N_TUO_BLUE}个\n")
        f.write(f"  原理：蒙特卡洛采样({len(all_samples)}次) + 精英选择(TOP {len(predictions)})\n\n")

        f.write("三、预测号码 (TOP SELECTED)\n")
        f.write("-" * 70 + "\n")

        for i, pred in enumerate(predictions, 1):
            f.write(f"\n【预测{i}】 样本ID: #{pred['sample_id']}\n")
            f.write(f"  🟥 红球胆码: {', '.join([f'{n:02d}' for n in pred['dan_red']])}\n")
            f.write(f"  🟨 红球拖码: {', '.join([f'{n:02d}' for n in pred['tuo_red']])}\n")
            f.write(f"     └─ 热区: {[f'{n:02d}' for n in sorted(pred['tuo_red'][:N_TUO_HOT_RED])]}\n")
            f.write(f"     └─ 冷区: {[f'{n:02d}' for n in sorted(pred['tuo_red'][N_TUO_HOT_RED:])]}\n")
            f.write(f"  🔵 蓝球胆码: {', '.join([f'{n:02d}' for n in pred['dan_blue']])}\n")
            f.write(f"  🟦 蓝球拖码: {', '.join([f'{n:02d}' for n in pred['tuo_blue']])}\n")
            f.write(f"  📊 红球池: {len(pred['red_pool'])}个 | 蓝球池: {len(pred['blue_pool'])}个\n")
            f.write(f"  📈 期望命中: 红{pred['red_stats']['expected_hits']:.2f} | "
                    f"蓝{pred['blue_stats']['expected_hits']:.2f} | "
                    f"总计{pred['total_expected']:.2f}\n")

        f.write("\n\n四、号码池汇总统计 (TOP 10)\n")
        f.write("-" * 70 + "\n")

        dan_counts = {i: 0 for i in range(1, 34)}
        tuo_counts = {i: 0 for i in range(1, 34)}

        for pred in predictions:
            for n in pred['dan_red']:
                dan_counts[n] += 1
            for n in pred['tuo_red']:
                tuo_counts[n] += 1

        f.write("\n  红球统计（各数字在TOP 10预测中出现的次数）:\n")
        f.write(f"  {'号码':^6} {'胆码次数':^10} {'拖码次数':^10} {'总次数':^10}\n")
        f.write("  " + "-" * 40 + "\n")
        for i in range(1, 34):
            dan_c = dan_counts[i]
            tuo_c = tuo_counts[i]
            total = dan_c + tuo_c
            f.write(f"  {i:^6} {dan_c:^10} {tuo_c:^10} {total:^10}\n")

        # ========== 修复部分：完整概率排名 ==========
        f.write("\n  完整概率排名（红球）:\n")
        sorted_reds = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)
        for i, (num, prob) in enumerate(sorted_reds, 1):
            deviation = (prob - 1 / 33) / (1 / 33) * 100
            sign = '+' if deviation > 0 else ''
            f.write(f"  {i:3d}. {num:02d}: {prob:.5f} ({sign}{deviation:.1f}%)\n")

        f.write("\n  完整概率排名（蓝球）:\n")
        sorted_blues = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)
        for i, (num, prob) in enumerate(sorted_blues, 1):
            deviation = (prob - 1 / 16) / (1 / 16) * 100
            sign = '+' if deviation > 0 else ''
            f.write(f"  {i:3d}. {num:02d}: {prob:.5f} ({sign}{deviation:.1f}%)\n")

        f.write("\n\n五、优化效果分析\n")
        f.write("-" * 70 + "\n")
        best_selected = predictions[0]['total_expected']
        worst_selected = predictions[-1]['total_expected']
        improvement = (best_selected - np.mean(all_expected)) / np.mean(all_expected) * 100

        f.write(f"  原始平均期望命中: {np.mean(all_expected):.3f}\n")
        f.write(f"  最佳选中期望命中: {best_selected:.3f}\n")
        f.write(f"  最差选中期望命中: {worst_selected:.3f}\n")
        f.write(f"  优化提升幅度: +{improvement:.2f}%\n")
        f.write(f"  超越样本比例: {(sum(1 for e in all_expected if e >= worst_selected) / len(all_expected) * 100):.1f}%\n")

    print(f"   ✅ optimized_predictions.txt")

    # 保存CSV格式
    csv_data = []
    for pred in predictions:
        row = {
            '预测组': f"TOP{predictions.index(pred) + 1}",
            '样本ID': pred['sample_id'],
            '红球胆码': ','.join([f'{n:02d}' for n in pred['dan_red']]),
            '红球拖码_热区': ','.join([f'{n:02d}' for n in sorted(pred['tuo_red'][:N_TUO_HOT_RED])]),
            '红球拖码_冷区': ','.join([f'{n:02d}' for n in sorted(pred['tuo_red'][N_TUO_HOT_RED:])]),
            '红球拖码_全部': ','.join([f'{n:02d}' for n in pred['tuo_red']]),
            '蓝球胆码': ','.join([f'{n:02d}' for n in pred['dan_blue']]),
            '蓝球拖码': ','.join([f'{n:02d}' for n in pred['tuo_blue']]),
            '红球池大小': len(pred['red_pool']),
            '蓝球池大小': len(pred['blue_pool']),
            '红球期望命中': round(pred['red_stats']['expected_hits'], 3),
            '蓝球期望命中': round(pred['blue_stats']['expected_hits'], 3),
            '总期望命中': round(pred['total_expected'], 3)
        }
        csv_data.append(row)

    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(f'{output_dir}/optimized_predictions.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ optimized_predictions.csv")

    # 保存全部采样数据
    sampling_stats = []
    for pred in all_samples:
        sampling_stats.append({
            '样本ID': pred['sample_id'],
            '红球胆码': ','.join([f'{n:02d}' for n in pred['dan_red']]),
            '红球拖码': ','.join([f'{n:02d}' for n in pred['tuo_red']]),
            '蓝球胆码': ','.join([f'{n:02d}' for n in pred['dan_blue']]),
            '蓝球拖码': ','.join([f'{n:02d}' for n in pred['tuo_blue']]),
            '红球期望命中': round(pred['red_stats']['expected_hits'], 3),
            '蓝球期望命中': round(pred['blue_stats']['expected_hits'], 3),
            '总期望命中': round(pred['total_expected'], 3)
        })

    df_sampling = pd.DataFrame(sampling_stats)
    df_sampling = df_sampling.sort_values('总期望命中', ascending=False)
    df_sampling.to_csv(f'{output_dir}/all_samples.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ all_samples.csv (全部采样数据)")

    return df_csv

def print_summary(predictions, all_samples):
    """打印优化总结"""
    print("\n" + "=" * 70)
    print("📊 优化总结")
    print("=" * 70)

    all_expected = [p['total_expected'] for p in all_samples]
    top_expected = [p['total_expected'] for p in predictions]

    print(f"\n  🎯 蒙特卡洛采样: {len(all_samples)} 次")
    print(f"  🏆 精英选择: TOP {len(predictions)} 组")
    print(f"\n  📈 期望命中统计:")
    print(f"     全部采样:")
    print(f"        - 最高: {max(all_expected):.3f}")
    print(f"        - 平均: {np.mean(all_expected):.3f}")
    print(f"        - 最低: {min(all_expected):.3f}")
    print(f"        - 标准差: {np.std(all_expected):.3f}")
    print(f"\n     TOP选中:")
    print(f"        - 最高: {max(top_expected):.3f}")
    print(f"        - 平均: {np.mean(top_expected):.3f}")
    print(f"        - 最低: {min(top_expected):.3f}")
    print(f"\n  📊 优化效果:")
    improvement = (predictions[0]['total_expected'] - np.mean(all_expected)) / np.mean(all_expected) * 100
    print(f"     相比平均提升: +{improvement:.2f}%")
    percentile = sum(1 for e in all_expected if e <= predictions[0]['total_expected']) / len(all_expected) * 100
    print(f"     超越样本比例: {percentile:.1f}%")

    print(f"\n  📁 输出文件:")
    print(f"     - optimized_predictions.txt (详细预测结果)")
    print(f"     - optimized_predictions.csv (预测结果表格)")
    print(f"     - all_samples.csv (全部采样数据)")
    print(f"     - optimized_analysis.png (综合分析图)")
    print(f"     - optimization_effect.png (优化效果图)")

    print("\n" + "=" * 70)


def main():
    """主函数"""
    print("=" * 70)
    print("🎯双色球胆拖投注预测 - 优化版")
    print("   蒙特卡洛采样 + 精英选择")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = datetime.now()

    # 1. 加载数据
    df = load_data()

    # 2. 生成优化预测
    top_predictions, all_samples, red_probs, blue_probs, x_vals, pdf = \
        generate_optimized_predictions(df, n_samples=N_SAMPLES, top_k=N_TOP_SELECT)

    # 3. 绘制图表
    print("\n📊 正在绘制可视化图表...")
    plot_optimized_charts(top_predictions, all_samples, red_probs, blue_probs, x_vals, pdf, OUTPUT_DIR)

    # 4. 保存结果
    print("\n💾 正在保存结果...")
    save_optimized_results(top_predictions, all_samples, red_probs, blue_probs, OUTPUT_DIR)

    # 5. 打印总结
    print_summary(top_predictions, all_samples)

    # 完成
    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print("✅ 预测完成!")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"⏱️  耗时: {duration:.2f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ============== UI兼容层 ==============
# 为lottery_ui提供兼容的函数接口

def generate_predictions(df, n=5, n_dan_blue=None, n_tuo_blue=None):
    """
    UI兼容的预测函数
    参数:
        df: 数据DataFrame
        n: 预测组数
        n_dan_blue: 蓝球胆码数（兼容参数）
        n_tuo_blue: 蓝球拖码数（兼容参数）
    返回:
        predictions, red_probs, blue_probs, x, pdf
    """
    global N_PREDICTIONS
    
    # 更新参数
    if n and n > 0:
        N_PREDICTIONS = n
    
    if n_dan_blue:
        N_DAN_BLUE = n_dan_blue
    if n_tuo_blue:
        N_TUO_BLUE = n_tuo_blue
    
    # 调用主预测函数
    top_predictions, all_samples, red_probs, blue_probs, x_vals, pdf = \
        generate_optimized_predictions(df, n_samples=N_SAMPLES, top_k=N_PREDICTIONS)
    
    # 转换为UI期望的格式
    result = []
    for pred in top_predictions:
        result.append({
            'red_dan': pred['dan_red'],
            'red_tuo': pred['tuo_red'],
            'blue_dan': pred['dan_blue'],
            'blue_tuo': pred['tuo_blue']
        })
    
    return result, red_probs, blue_probs, x_vals, pdf


def plot_prediction_pools(predictions, red_probs, blue_probs, x, pdf, output_dir):
    """
    UI兼容的绘图函数
    """
    # 转换为模块期望的格式
    top_preds = []
    for p in predictions:
        top_preds.append({
            'dan_red': p.get('red_dan', p.get('dan_red', [])),
            'tuo_red': p.get('red_tuo', p.get('tuo_red', [])),
            'dan_blue': p.get('blue_dan', p.get('dan_blue', [])),
            'tuo_blue': p.get('blue_tuo', p.get('tuo_blue', [])),
            'red_pool': p.get('red_dan', []) + p.get('red_tuo', []),
            'blue_pool': p.get('blue_dan', []) + p.get('blue_tuo', []),
            'red_stats': {'expected_hits': sum(red_probs.get(n, 0) for n in p.get('red_dan', []) + p.get('red_tuo', []))},
            'blue_stats': {'expected_hits': sum(blue_probs.get(n, 0) for n in p.get('blue_dan', []) + p.get('blue_tuo', []))},
            'sample_id': predictions.index(p),
            'total_expected': sum(red_probs.get(n, 0) for n in p.get('red_dan', []) + p.get('red_tuo', [])) + 
                            sum(blue_probs.get(n, 0) for n in p.get('blue_dan', []) + p.get('blue_tuo', []))
        })
    
    # 生成全部采样数据
    all_samples = top_preds.copy()
    
    # 调用实际绘图函数
    plot_optimized_charts(top_preds, all_samples, red_probs, blue_probs, x, pdf, output_dir)

