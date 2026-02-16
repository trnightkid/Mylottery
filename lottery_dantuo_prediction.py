"""
双色球预测分析 - 胆拖投注预测（无sklearn依赖）
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

# 胆拖参数
N_DAN_RED = 4  # 红球胆码数量
N_TUO_HOT_RED = 3  # 红球拖码数量（高拟合区）
N_TUO_COLD_RED = 2  # 红球拖码数量（冷门区）

N_DAN_BLUE = 1  # 蓝球胆码数量
N_TUO_BLUE = 2  # 蓝球拖码数量

N_PREDICTIONS = 5  # 生成多少组预测


# =====================================

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


def generate_predictions(df, n=5):
    """生成预测结果"""
    print("\n" + "=" * 70)
    print("🎯 胆拖投注预测")
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

    print(f"\n🎲 生成 {n} 组胆拖预测号码:")
    print("-" * 70)

    predictions = []

    for group in range(1, n + 1):
        dan_red, tuo_red, full_pool_red = build_red_dantuo_pool(
            red_probs,
            n_dan=N_DAN_RED,
            n_tuo_hot=N_TUO_HOT_RED,
            n_tuo_cold=N_TUO_COLD_RED,
            seed=42 + group
        )

        dan_blue, tuo_blue, full_pool_blue = build_blue_dantuo_pool(
            blue_probs,
            n_dan=N_DAN_BLUE,
            n_tuo=N_TUO_BLUE,
            seed=42 + group
        )

        red_stats = calculate_pool_statistics(dan_red, tuo_red, red_probs, 33)
        blue_stats = calculate_pool_statistics(dan_blue, tuo_blue, blue_probs, 16)

        prediction = {
            'group': group,
            'red_dan': sorted(dan_red),
            'red_tuo': sorted(tuo_red),
            'red_pool': sorted(dan_red + tuo_red),
            'blue_dan': sorted(dan_blue),
            'blue_tuo': sorted(tuo_blue),
            'blue_pool': sorted(dan_blue + tuo_blue),
            'red_stats': red_stats,
            'blue_stats': blue_stats,
            'seed': 42 + group
        }
        predictions.append(prediction)

        print(f"\n【预测{group}】")
        print(f"  🟥 红球胆码 ({len(dan_red)}个): ", end="")
        print(", ".join([f"{n:02d}" for n in sorted(dan_red)]))
        print(f"  🟨 红球拖码 ({len(tuo_red)}个): ", end="")
        print(", ".join([f"{n:02d}" for n in sorted(tuo_red)]))
        print(f"     └─ 热区: {[f'{n:02d}' for n in sorted(tuo_red[:N_TUO_HOT_RED])]}", end=" ")
        print(f"+ 冷区: {[f'{n:02d}' for n in sorted(tuo_red[N_TUO_HOT_RED:])]}")
        print(f"  🔵 蓝球胆码 ({len(dan_blue)}个): ", end="")
        print(", ".join([f"{n:02d}" for n in sorted(dan_blue)]))
        print(f"  🟦 蓝球拖码 ({len(tuo_blue)}个): ", end="")
        print(", ".join([f"{n:02d}" for n in sorted(tuo_blue)]))
        print(f"  📊 覆盖红球: {len(full_pool_red)}个 | 蓝球: {len(full_pool_blue)}个")
        print(f"  📈 红球期望命中: {red_stats['expected_hits']:.2f} | 蓝球: {blue_stats['expected_hits']:.2f}")

    return predictions, red_probs, blue_probs, x, pdf


def plot_prediction_pools(predictions, red_probs, blue_probs, x_vals, pdf, output_dir):
    """绘制胆拖号码池可视化"""
    os.makedirs(output_dir, exist_ok=True)

    # ===================== 图表1：综合分析 =====================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1：红球概率分布
    ax1 = axes1[0, 0]
    nums = list(range(1, 34))
    probs = [red_probs[n] for n in nums]

    colors = []
    for n in nums:
        if n in predictions[0]['red_dan']:
            colors.append('#FF4444')
        elif n in predictions[0]['red_tuo']:
            colors.append('#FFA500')
        else:
            colors.append('#CCCCCC')

    ax1.bar(nums, probs, color=colors, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=1 / 33, color='blue', linestyle='--', alpha=0.5, label='均匀分布')
    ax1.set_xlabel('Red Ball Number', fontsize=11)
    ax1.set_ylabel('Fitted Probability', fontsize=11)
    ax1.set_title('Red Ball Fitted Probability Distribution (Prediction 1)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, 34, 2))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF4444', label='Dan (Key)'),
        Patch(facecolor='#FFA500', label='Tuo (Extended)'),
        Patch(facecolor='#CCCCCC', label='Not Selected')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # 子图2：多组预测的胆拖分布
    ax2 = axes1[0, 1]

    dan_counts = np.zeros(34)
    tuo_counts = np.zeros(34)

    for pred in predictions:
        for n in pred['red_dan']:
            dan_counts[n - 1] += 1
        for n in pred['red_tuo']:
            tuo_counts[n - 1] += 1

    x_pos = np.arange(34)
    width = 0.6

    ax2.bar(x_pos, dan_counts, width, label='Dan Count', color='#FF4444')
    ax2.bar(x_pos, tuo_counts, width, bottom=dan_counts, label='Tuo Count', color='#FFA500')
    ax2.set_xlabel('Red Ball Number', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'Dan/Tuo Distribution in {len(predictions)} Predictions', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(i) for i in range(1, 35)])
    ax2.legend()

    # 子图3：蓝球概率
    ax3 = axes1[1, 0]
    blue_nums = list(range(1, 17))
    blue_probs_list = [blue_probs[n] for n in blue_nums]

    colors_blue = []
    for n in blue_nums:
        if n in predictions[0]['blue_dan']:
            colors_blue.append('#4444FF')
        elif n in predictions[0]['blue_tuo']:
            colors_blue.append('#44AAFF')
        else:
            colors_blue.append('#CCCCCC')

    ax3.bar(blue_nums, blue_probs_list, color=colors_blue)
    ax3.axhline(y=1 / 16, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Blue Ball Number', fontsize=11)
    ax3.set_ylabel('Fitted Probability', fontsize=11)
    ax3.set_title('Blue Ball Fitted Probability Distribution', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(1, 17))

    # 子图4：期望命中数对比
    ax4 = axes1[1, 1]

    groups = [f'Pred {p["group"]}' for p in predictions]
    red_expected = [p['red_stats']['expected_hits'] for p in predictions]
    blue_expected = [p['blue_stats']['expected_hits'] for p in predictions]

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, red_expected, width, label='Red Expected Hits', color='#FF6B6B')
    bars2 = ax4.bar(x + width / 2, blue_expected, width, label='Blue Expected Hits', color='#4ECDC4')

    ax4.set_ylabel('Expected Hits', fontsize=11)
    ax4.set_title('Expected Hits by Prediction', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(groups)
    ax4.legend()
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dan_tuo_pools.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ dan_tuo_pools.png")

    # ===================== 图表2：拟合曲线 =====================
    fig2, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(x_vals, pdf, alpha=0.3, color='steelblue')
    ax.plot(x_vals, pdf, color='steelblue', linewidth=2, label='Fitted Distribution')

    ax.axvline(x=11, color='red', linestyle='--', alpha=0.5, label='Zone Boundary')
    ax.axvline(x=22, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Red Ball Number', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Double Color Ball Red Ball Fitted Distribution Curve', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 33)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fitting_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ fitting_curve.png")

    # ===================== 图表3：红球热力图 =====================
    fig3, ax = plt.subplots(figsize=(16, 6))

    prob_matrix = np.zeros((33, 33))
    for i in range(1, 34):
        for j in range(1, 34):
            if i <= j:
                prob_matrix[i - 1, j - 1] = red_probs.get(i, 0) + red_probs.get(j, 0)

    im = ax.imshow(prob_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(33))
    ax.set_xticklabels(range(1, 34))
    ax.set_yticks(range(33))
    ax.set_yticklabels(range(1, 34))
    ax.set_xlabel('Red Ball Number (End)', fontsize=12)
    ax.set_ylabel('Red Ball Number (Start)', fontsize=12)
    ax.set_title('Red Ball Combination Probability Heatmap', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Combined Probability', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/red_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ red_heatmap.png")

    # ===================== 图表4：蓝球雷达图 =====================
    fig4, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    blue_nums = list(range(1, 17))
    blue_probs_list = [blue_probs[n] for n in blue_nums]

    max_prob = max(blue_probs_list)
    normalized_probs = [p / max_prob for p in blue_probs_list]

    angles = np.linspace(0, 2 * np.pi, 17, endpoint=True)

    ax.plot(angles, normalized_probs + [normalized_probs[0]], 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, normalized_probs + [normalized_probs[0]], alpha=0.25, color='steelblue')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(blue_nums)
    ax.set_title('Blue Ball Probability Radar Chart', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/blue_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ blue_radar.png")


def save_prediction_results(predictions, red_probs, blue_probs, output_dir):
    """保存预测结果"""
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/dan_tuo_predictions.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("              双色球胆拖投注预测结果\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"预测组数: {len(predictions)}\n\n")

        f.write("一、胆拖策略说明\n")
        f.write("-" * 70 + "\n")
        f.write(f"  红球：胆{N_DAN_RED}个 + 拖{N_TUO_HOT_RED}个(热区) + 拖{N_TUO_COLD_RED}个(冷区)\n")
        f.write(f"  蓝球：胆{N_DAN_BLUE}个 + 拖{N_TUO_BLUE}个\n")
        f.write("  原理：高拟合度区随机抽取胆码和热拖，冷门区抽取冷拖\n\n")

        f.write("二、预测号码\n")
        f.write("-" * 70 + "\n")

        for pred in predictions:
            f.write(f"\n【预测{pred['group']}】 种子: {pred['seed']}\n")
            f.write(f"  🟥 红球胆码: {', '.join([f'{n:02d}' for n in pred['red_dan']])}\n")
            f.write(f"  🟨 红球拖码: {', '.join([f'{n:02d}' for n in pred['red_tuo']])}\n")
            f.write(f"     └─ 热区: {[f'{n:02d}' for n in sorted(pred['red_tuo'][:N_TUO_HOT_RED])]}\n")
            f.write(f"     └─ 冷区: {[f'{n:02d}' for n in sorted(pred['red_tuo'][N_TUO_HOT_RED:])]}\n")
            f.write(f"  🔵 蓝球胆码: {', '.join([f'{n:02d}' for n in pred['blue_dan']])}\n")
            f.write(f"  🟦 蓝球拖码: {', '.join([f'{n:02d}' for n in pred['blue_tuo']])}\n")
            f.write(f"  📊 红球池: {len(pred['red_pool'])}个 | 蓝球池: {len(pred['blue_pool'])}个\n")
            f.write(
                f"  📈 期望命中: 红球{pred['red_stats']['expected_hits']:.2f} | 蓝球{pred['blue_stats']['expected_hits']:.2f}\n")

        f.write("\n\n三、号码池汇总统计\n")
        f.write("-" * 70 + "\n")

        dan_counts = {i: 0 for i in range(1, 34)}
        tuo_counts = {i: 0 for i in range(1, 34)}

        for pred in predictions:
            for n in pred['red_dan']:
                dan_counts[n] += 1
            for n in pred['red_tuo']:
                tuo_counts[n] += 1

        f.write("\n  红球统计（各数字在预测中出现的次数）:\n")
        f.write(f"  {'号码':^6} {'胆码次数':^10} {'拖码次数':^10} {'总次数':^10}\n")
        f.write("  " + "-" * 40 + "\n")
        for i in range(1, 34):
            dan_c = dan_counts[i]
            tuo_c = tuo_counts[i]
            total = dan_c + tuo_c
            f.write(f"  {i:^6} {dan_c:^10} {tuo_c:^10} {total:^10}\n")

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

    print(f"   ✅ dan_tuo_predictions.txt")

    # 保存CSV格式
    csv_data = []
    for pred in predictions:
        row = {
            '预测组': pred['group'],
            '红球胆码': ','.join([f'{n:02d}' for n in pred['red_dan']]),
            '红球拖码_热区': ','.join([f'{n:02d}' for n in sorted(pred['red_tuo'][:N_TUO_HOT_RED])]),
            '红球拖码_冷区': ','.join([f'{n:02d}' for n in sorted(pred['red_tuo'][N_TUO_HOT_RED:])]),
            '蓝球胆码': ','.join([f'{n:02d}' for n in pred['blue_dan']]),
            '蓝球拖码': ','.join([f'{n:02d}' for n in pred['blue_tuo']]),
            '红球池大小': len(pred['red_pool']),
            '蓝球池大小': len(pred['blue_pool']),
            '红球期望命中': round(pred['red_stats']['expected_hits'], 2),
            '蓝球期望命中': round(pred['blue_stats']['expected_hits'], 2)
        }
        csv_data.append(row)

    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(f'{output_dir}/dan_tuo_predictions.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ dan_tuo_predictions.csv")


def main():
    """主函数"""
    print("=" * 70)
    print("🎯双色球胆拖投注预测 - 基于分布拟合")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = datetime.now()

    # 1. 加载数据
    df = load_data()

    # 2. 生成预测（返回5个值）
    predictions, red_probs, blue_probs, x_vals, pdf = generate_predictions(df, N_PREDICTIONS)

    # 3. 绘制图表（传递x_vals和pdf）
    plot_prediction_pools(predictions, red_probs, blue_probs, x_vals, pdf, OUTPUT_DIR)

    # 4. 保存结果
    save_prediction_results(predictions, red_probs, blue_probs, OUTPUT_DIR)

    # 完成
    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print("✅ 预测完成!")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"⏱️  耗时: {duration:.2f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()
