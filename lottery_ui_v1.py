"""
双色球预测分析 - GUI界面版
功能：
  1. 爬取最新开奖数据
  2. 胆拖投注预测
  3. 结果展示 + 图片预览
"""
"""
双色球预测分析 - GUI界面版 (保留原设置 + 新增蓝球参数)
"""
import pymysql
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from datetime import datetime
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import requests
import threading
import csv
import os

warnings.filterwarnings('ignore')

# 设置matplotlib无GUI后端
plt.switch_backend('Agg')

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
CSV_FILE = r"D:\Mydevelopment\MultiContentProject\Mylottery\lottery_data_from_web.csv"

# 胆拖参数
N_DAN_RED = 3  # 红球胆码数量
N_TUO_HOT_RED = 3  # 红球拖码数量（高拟合区）
N_TUO_COLD_RED = 2  # 红球拖码数量（冷门区）

N_DAN_BLUE = 1  # 蓝球胆码数量
N_TUO_BLUE = 2  # 蓝球拖码数量

N_PREDICTIONS = 5  # 生成多少组预测


# ======================================


# ============== 数据爬取模块 ==============
def crawl_latest_data():

    """从官网爬取最新开奖数据"""
    try:
        url = "https://datachart.500star.com/ssq/history/history.shtml"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'

        # 🔍 调试：打印页面结构
        print("\n" + "="*60)
        print("页面结构分析:")
        print("="*60)

        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找可能的容器
        containers = soup.find_all(class_=lambda x: x and ('list' in str(x).lower() or 'data' in str(x).lower() or 'chart' in str(x).lower()))
        print(f"找到可能的容器: {len(containers)}")

        # 查找开奖号码元素
        ball_elements = soup.find_all(class_=lambda x: x and ('ball' in str(x).lower() or 'red' in str(x).lower() or 'blue' in str(x).lower()))
        print(f"找到球号元素: {len(ball_elements)}")
        for elem in ball_elements[:10]:
            print(f"  class={elem.get('class')}, text={elem.get_text().strip()}")

        # 直接打印前3000字符看看整体结构
        print("\n页面HTML前3000字符:")
        print(response.text[:3000])
        print("="*60)

        return None  # 先返回None，通过调试信息确定解析方式

    except Exception as e:
        print(f"爬取出错: {e}")
        return None


        # 查找开奖数据表格
        data_list = []
        rows = soup.select('table tr')

        for row in rows[1:6]:  # 取前5期
            cols = row.select('td')
            if len(cols) >= 8:
                period = cols[0].get_text().strip()
                red1 = int(cols[1].get_text().strip())
                red2 = int(cols[2].get_text().strip())
                red3 = int(cols[3].get_text().strip())
                red4 = int(cols[4].get_text().strip())
                red5 = int(cols[5].get_text().strip())
                red6 = int(cols[6].get_text().strip())
                blue = int(cols[7].get_text().strip())
                draw_date = cols[8].get_text().strip() if len(cols) > 8 else datetime.now().strftime('%Y-%m-%d')

                data_list.append({
                    'period': period,
                    'red1': red1, 'red2': red2, 'red3': red3,
                    'red4': red4, 'red5': red5, 'red6': red6,
                    'blue': blue,
                    'draw_date': draw_date
                })

        return data_list
    except Exception as e:
        # 备用方案：手动输入
        return None


def save_to_database(data_list):
    """保存数据到数据库"""
    if not data_list:
        return False

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    saved_count = 0
    for data in data_list:
        try:
            # 检查是否已存在
            cursor.execute("SELECT id FROM lottery_db WHERE period = %s", (data['period'],))
            if cursor.fetchone():
                continue

            cursor.execute("""
                INSERT INTO lottery_db 
                (period, red1, red2, red3, red4, red5, red6, blue, draw_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (data['period'], data['red1'], data['red2'], data['red3'],
                  data['red4'], data['red5'], data['red6'], data['blue'], data['draw_date']))
            saved_count += 1
        except Exception as e:
            print(f"保存失败: {data['period']}, {e}")

    conn.commit()
    cursor.close()
    conn.close()

    return saved_count


# ======================================


# ============== 预测核心模块 ==============
def load_data():
    """从数据库加载数据"""
    conn = pymysql.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date
        FROM lottery_db 
        ORDER BY CAST(period AS UNSIGNED)
    """, conn)
    conn.close()
    return df


def fit_kde(data, x):
    kde = stats.gaussian_kde(data)
    pdf = kde(x)
    return pdf / pdf.sum()


def fit_beta(data, x):
    normalized = (data - 1) / 32
    try:
        a, b, loc, scale = stats.beta.fit(normalized, floc=0, fscale=1)
        x_norm = (x - 1) / 32
        pdf = stats.beta.pdf(x_norm, a, b)
        return pdf / pdf.sum()
    except:
        return np.ones_like(x) / len(x)


def fit_trimodal(data, x):
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
    red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
    all_reds = np.array([df[col].values for col in red_cols]).flatten()

    x = np.linspace(1, 33, 1000)
    pdf_kde = fit_kde(all_reds, x)
    pdf_beta = fit_beta(all_reds, x)
    pdf_gmm = fit_trimodal(all_reds, x)

    freq_counts = Counter(all_reds)
    pdf_freq = np.zeros_like(x)
    for num in range(1, 34):
        idx = np.abs(x - num).argmin()
        pdf_freq[idx] = freq_counts.get(num, 0)
    pdf_freq = pdf_freq / (pdf_freq.sum() + 1e-10)

    combined = (0.30 * pdf_kde + 0.25 * pdf_beta + 0.25 * pdf_gmm + 0.20 * pdf_freq)
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
    blue_counts = Counter(df['blue'].tolist())
    total = len(df)

    blue_probs = {}
    for num in range(1, 17):
        count = blue_counts.get(num, 0)
        blue_probs[num] = (count + 1) / (total + 16)

    return blue_probs



def weighted_random_choice(probs_dict, n, exclude=None):
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
    if seed is not None:
        np.random.seed(seed)

    dan = weighted_random_choice(blue_probs, n_dan)
    dan_set = set(dan)
    tuo = weighted_random_choice(blue_probs, n_tuo, exclude=dan_set)
    full_pool = dan + tuo

    return dan, tuo, full_pool


def calculate_pool_statistics(dan, tuo, probs, total_nums):
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
    red_probs, x, pdf = fit_distributions(df)
    blue_probs = calculate_blue_probs(df)

    predictions = []

    for group in range(1, n + 1):
        dan_red, tuo_red, full_pool_red = build_red_dantuo_pool(
            red_probs, n_dan=N_DAN_RED, n_tuo_hot=N_TUO_HOT_RED,
            n_tuo_cold=N_TUO_COLD_RED, seed=42 + group
        )

        dan_blue, tuo_blue, full_pool_blue = build_blue_dantuo_pool(
            blue_probs, n_dan=N_DAN_BLUE, n_tuo=N_TUO_BLUE, seed=42 + group
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

    return predictions, red_probs, blue_probs, x, pdf


# ======================================


# ============== 绘图模块 ==============
def plot_prediction_pools(predictions, red_probs, blue_probs, x_vals, pdf, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 图1：综合分析
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

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
    ax1.axhline(y=1 / 33, color='blue', linestyle='--', alpha=0.5, label='Uniform')
    ax1.set_xlabel('Red Ball Number', fontsize=11)
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_title('Red Ball Probability Distribution (Pred 1)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, 34, 2))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF4444', label='Dan'),
        Patch(facecolor='#FFA500', label='Tuo'),
        Patch(facecolor='#CCCCCC', label='Not Selected')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

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
    ax2.set_title(f'Dan/Tuo in {len(predictions)} Predictions', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(i) for i in range(1, 35)])
    ax2.legend()

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
    ax3.set_ylabel('Probability', fontsize=11)
    ax3.set_title('Blue Ball Probability', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(1, 17))

    ax4 = axes1[1, 1]
    groups = [f'Pred {p["group"]}' for p in predictions]
    red_expected = [p['red_stats']['expected_hits'] for p in predictions]
    blue_expected = [p['blue_stats']['expected_hits'] for p in predictions]

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, red_expected, width, label='Red Expected', color='#FF6B6B')
    bars2 = ax4.bar(x + width / 2, blue_expected, width, label='Blue Expected', color='#4ECDC4')

    ax4.set_ylabel('Expected Hits', fontsize=11)
    ax4.set_title('Expected Hits by Prediction', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(groups)
    ax4.legend()
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dan_tuo_pools.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图2：拟合曲线
    fig2, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x_vals, pdf, alpha=0.3, color='steelblue')
    ax.plot(x_vals, pdf, color='steelblue', linewidth=2, label='Fitted')
    ax.axvline(x=11, color='red', linestyle='--', alpha=0.5, label='Zone')
    ax.axvline(x=22, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Red Ball Number', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Red Ball Fitted Curve', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 33)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fitting_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图3：红球热力图
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
    ax.set_xlabel('Red Ball (End)', fontsize=12)
    ax.set_ylabel('Red Ball (Start)', fontsize=12)
    ax.set_title('Red Ball Combination Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/red_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图4：蓝球雷达图
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
    ax.set_title('Blue Ball Radar', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/blue_radar.png', dpi=150, bbox_inches='tight')
    plt.close()


# ======================================


# ============== GUI界面类 ==============
class LotteryPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎱 双色球胆拖预测系统")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)

        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')

        # 主容器
        main_container = ttk.Frame(root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 右侧内容面板
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ========== 左侧控制面板 ==========
        # 标题
        title_label = ttk.Label(left_panel, text="🎱 双色球预测", font=('Microsoft YaHei', 16, 'bold'))
        title_label.pack(pady=(0, 15))

        # 数据库信息
        info_frame = ttk.LabelFrame(left_panel, text="📊 数据库状态", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.db_status_label = ttk.Label(info_frame, text="未连接", foreground="red")
        self.db_status_label.pack(anchor=tk.W)

        self.data_count_label = ttk.Label(info_frame, text="数据量: 0 条", foreground="gray")
        self.data_count_label.pack(anchor=tk.W)

        # 爬取数据按钮
        crawl_frame = ttk.LabelFrame(left_panel, text="🌐 数据爬取", padding="10")
        crawl_frame.pack(fill=tk.X, pady=(0, 10))

        self.crawl_btn = ttk.Button(crawl_frame, text="爬取最新开奖数据", command=self.crawl_data)
        self.crawl_btn.pack(fill=tk.X, pady=(0, 5))

        self.crawl_status = ttk.Label(crawl_frame, text="就绪", foreground="gray")
        self.crawl_status.pack(anchor=tk.W)

        # 预测参数
        param_frame = ttk.LabelFrame(left_panel, text="⚙️ 预测参数", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(param_frame, text="红球胆码数:").grid(row=0, column=0, sticky=tk.W)
        self.n_dan_red = ttk.Spinbox(param_frame, from_=1, to=6, width=8)
        self.n_dan_red.set(N_DAN_RED)
        self.n_dan_red.grid(row=0, column=1, padx=(5, 0))

        ttk.Label(param_frame, text="红球热拖数:").grid(row=1, column=0, sticky=tk.W)
        self.n_tuo_hot = ttk.Spinbox(param_frame, from_=1, to=6, width=8)
        self.n_tuo_hot.set(N_TUO_HOT_RED)
        self.n_tuo_hot.grid(row=1, column=1, padx=(5, 0))

        ttk.Label(param_frame, text="红球冷拖数:").grid(row=2, column=0, sticky=tk.W)
        self.n_tuo_cold = ttk.Spinbox(param_frame, from_=0, to=6, width=8)
        self.n_tuo_cold.set(N_TUO_COLD_RED)
        self.n_tuo_cold.grid(row=2, column=1, padx=(5, 0))

        ttk.Label(param_frame, text="预测组数:").grid(row=3, column=0, sticky=tk.W)
        self.n_predictions = ttk.Spinbox(param_frame, from_=1, to=10, width=8)
        self.n_predictions.set(N_PREDICTIONS)
        self.n_predictions.grid(row=3, column=1, padx=(5, 0))

        # 预测按钮
        predict_frame = ttk.LabelFrame(left_panel, text="🎯 开始预测", padding="10")
        predict_frame.pack(fill=tk.X, pady=(0, 10))

        self.predict_btn = ttk.Button(predict_frame, text="开始预测", command=self.run_prediction)
        self.predict_btn.pack(fill=tk.X)

        self.predict_status = ttk.Label(predict_frame, text="请先加载数据", foreground="gray")
        self.predict_status.pack(anchor=tk.W, pady=(5, 0))

        # 刷新数据库按钮
        self.refresh_btn = ttk.Button(left_panel, text="🔄 刷新数据库状态", command=self.refresh_db_status)
        self.refresh_btn.pack(fill=tk.X, pady=(5, 0))

        # ========== 右侧内容面板 ==========
        # 创建Notebook选项卡
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 选项卡1：预测结果
        self.result_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="📋 预测结果")
        self.setup_result_tab()

        # 选项卡2：图表展示
        self.chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_tab, text="📈 图表分析")
        self.setup_chart_tab()

        # 选项卡3：历史数据
        self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.history_tab, text="📜 历史数据")
        self.setup_history_tab()

        # 初始化数据库状态
        self.refresh_db_status()

    def setup_result_tab(self):
        """设置预测结果选项卡"""
        # 创建可滚动的文本框
        self.result_text = ScrolledText(self.result_tab, wrap=tk.WORD, font=('Consolas', 11))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 默认显示提示信息
        self.result_text.insert(tk.END, "=" * 60 + "\n")
        self.result_text.insert(tk.END, "🎱 双色球胆拖投注预测系统\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n\n")
        self.result_text.insert(tk.END, "使用说明：\n")
        self.result_text.insert(tk.END, "1. 点击左侧「爬取最新开奖数据」更新数据库\n")
        self.result_text.insert(tk.END, "2. 调整预测参数（可选）\n")
        self.result_text.insert(tk.END, "3. 点击「开始预测」生成预测结果\n")
        self.result_text.insert(tk.END, "4. 查看预测结果和图表分析\n")
        self.result_text.insert(tk.END, "\n注意：预测结果仅供参考，请理性购彩！\n")

    def setup_chart_tab(self):
        """设置图表选项卡"""
        # 创建图表选择框架
        chart_select_frame = ttk.Frame(self.chart_tab)
        chart_select_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(chart_select_frame, text="选择图表:").pack(side=tk.LEFT)

        self.chart_var = tk.StringVar(value='dan_tuo_pools')
        charts = [
            ('综合分析图', 'dan_tuo_pools'),
            ('拟合曲线图', 'fitting_curve'),
            ('红球热力图', 'red_heatmap'),
            ('蓝球雷达图', 'blue_radar')
        ]

        for text, value in charts:
            rb = ttk.Radiobutton(chart_select_frame, text=text, variable=self.chart_var,
                                 value=value, command=self.show_chart)
            rb.pack(side=tk.LEFT, padx=(0, 10))

        # 图片显示区域
        self.chart_frame = ttk.Frame(self.chart_tab)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 默认提示
        self.chart_label = ttk.Label(self.chart_frame, text="请先运行预测生成图表",
                                     font=('Microsoft YaHei', 14), foreground="gray")
        self.chart_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # 打开图片文件夹按钮
        self.open_folder_btn = ttk.Button(chart_select_frame, text="📂 打开图片文件夹",
                                          command=self.open_image_folder)
        self.open_folder_btn.pack(side=tk.RIGHT)

    def setup_history_tab(self):
        """设置历史数据选项卡"""
        # 搜索框
        search_frame = ttk.Frame(self.history_tab)
        search_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(search_frame, text="搜索期号:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame, width=20)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="搜索", command=self.search_history).pack(side=tk.LEFT)
        ttk.Button(search_frame, text="显示全部", command=self.show_all_history).pack(side=tk.LEFT, padx=5)

        # 历史数据表格
        columns = ('period', 'reds', 'blue', 'date')
        self.tree = ttk.Treeview(self.history_tab, columns=columns, show='headings', height=20)

        self.tree.heading('period', text='期号')
        self.tree.heading('reds', text='红球')
        self.tree.heading('blue', text='蓝球')
        self.tree.heading('date', text='开奖日期')

        self.tree.column('period', width=100, anchor='center')
        self.tree.column('reds', width=200, anchor='center')
        self.tree.column('blue', width=80, anchor='center')
        self.tree.column('date', width=120, anchor='center')

        # 滚动条
        scrollbar = ttk.Scrollbar(self.history_tab, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # 加载历史数据
        self.load_history_data()

    def load_history_data(self, keyword=''):
        """加载历史数据"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            conn = pymysql.connect(**DB_CONFIG)
            cursor = conn.cursor()

            if keyword:
                cursor.execute("""
                    SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date
                    FROM lottery_db 
                    WHERE period LIKE %s
                    ORDER BY CAST(period AS UNSIGNED) DESC
                    LIMIT 100
                """, (f'%{keyword}%'))
            else:
                cursor.execute("""
                    SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date
                    FROM lottery_db 
                    ORDER BY CAST(period AS UNSIGNED) DESC
                    LIMIT 100
                """)

            rows = cursor.fetchall()
            for row in rows:
                reds = f"{row[1]:02d} {row[2]:02d} {row[3]:02d} {row[4]:02d} {row[5]:02d} {row[6]:02d}"
                self.tree.insert('', tk.END, values=(row[0], reds, f"{row[7]:02d}", row[8]))

            cursor.close()
            conn.close()

        except Exception as e:
            messagebox.showerror("错误", f"加载历史数据失败：{e}")

    def search_history(self):
        keyword = self.search_entry.get().strip()
        self.load_history_data(keyword)

    def show_all_history(self):
        self.search_entry.delete(0, tk.END)
        self.load_history_data()

    def refresh_db_status(self):
        """刷新数据库状态"""
        try:
            conn = pymysql.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM lottery_db")
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()

            self.db_status_label.config(text="✅ 已连接", foreground="green")
            self.data_count_label.config(text=f"数据量: {count} 条")

        except Exception as e:
            self.db_status_label.config(text="❌ 连接失败", foreground="red")
            self.data_count_label.config(text=f"错误: {e}")

    def crawl_data(self):
        """爬取数据"""
        self.crawl_btn.config(state=tk.DISABLED)
        self.crawl_status.config(text="正在爬取...", foreground="blue")
        self.root.update()

        def do_crawl():
            try:
                data_list = crawl_latest_data()

                if data_list:
                    saved = save_to_database(data_list)
                    self.root.after(0, lambda: self.crawl_status.config(
                        text=f"✅ 成功获取 {len(data_list)} 期，保存 {saved} 条", foreground="green"))
                    self.root.after(0, self.refresh_db_status)
                    self.root.after(0, lambda: messagebox.showinfo("完成", f"成功获取 {len(data_list)} 期数据"))
                else:
                    # 尝试备用方案
                    self.root.after(0, lambda: self.crawl_status.config(
                        text="❌ 爬取失败，请手动输入", foreground="red"))
                    self.root.after(0, lambda: messagebox.showwarning("提示", "自动爬取失败，请检查网络连接"))

            except Exception as e:
                self.root.after(0, lambda: self.crawl_status.config(
                    text=f"❌ 错误: {e}", foreground="red"))
            finally:
                self.root.after(0, lambda: self.crawl_btn.config(state=tk.NORMAL))

        threading.Thread(target=do_crawl, daemon=True).start()

    def run_prediction(self):
        """运行预测"""
        # 获取参数
        try:
            n_dan_red = int(self.n_dan_red.get())
            n_tuo_hot = int(self.n_tuo_hot.get())
            n_tuo_cold = int(self.n_tuo_cold.get())
            n_predictions = int(self.n_predictions.get())
        except ValueError:
            messagebox.showerror("错误", "参数输入无效")
            return

        self.predict_btn.config(state=tk.DISABLED)
        self.predict_status.config(text="预测中...", foreground="blue")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "预测中，请稍候...\n")
        self.root.update()

        def do_prediction():
            try:
                # 加载数据
                df = load_data()

                if len(df) < 10:
                    self.root.after(0, lambda: self.predict_status.config(
                        text="❌ 数据量不足", foreground="red"))
                    self.root.after(0, lambda: self.result_text.insert(tk.END, "错误：数据库中数据不足，请先爬取数据！\n"))
                    return

                # 生成预测
                predictions, red_probs, blue_probs, x_vals, pdf = generate_predictions(df, n_predictions)

                # 绘制图表
                plot_prediction_pools(predictions, red_probs, blue_probs, x_vals, pdf, OUTPUT_DIR)

                # 显示结果
                result_text = self.result_text
                result_text.delete(1.0, tk.END)

                result_text.insert(tk.END, "=" * 70 + "\n")
                result_text.insert(tk.END, "              🎱 双色球胆拖投注预测结果\n")
                result_text.insert(tk.END, "=" * 70 + "\n\n")
                result_text.insert(tk.END, f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                result_text.insert(tk.END, f"📊 数据库数据: {len(df)} 条\n\n")

                result_text.insert(tk.END, "一、预测号码\n")
                result_text.insert(tk.END, "-" * 70 + "\n")

                for pred in predictions:
                    result_text.insert(tk.END, f"\n【预测{pred['group']}】\n")
                    result_text.insert(tk.END, f"  🟥 红球胆码: {', '.join([f'{n:02d}' for n in pred['red_dan']])}\n")
                    result_text.insert(tk.END, f"  🟨 红球拖码: {', '.join([f'{n:02d}' for n in pred['red_tuo']])}\n")
                    result_text.insert(tk.END,
                                       f"     └─ 热区: {[f'{n:02d}' for n in sorted(pred['red_tuo'][:n_tuo_hot])]}\n")
                    result_text.insert(tk.END,
                                       f"     └─ 冷区: {[f'{n:02d}' for n in sorted(pred['red_tuo'][n_tuo_hot:])]}\n")
                    result_text.insert(tk.END, f"  🔵 蓝球胆码: {', '.join([f'{n:02d}' for n in pred['blue_dan']])}\n")
                    result_text.insert(tk.END, f"  🟦 蓝球拖码: {', '.join([f'{n:02d}' for n in pred['blue_tuo']])}\n")
                    result_text.insert(tk.END,
                                       f"  📊 期望命中: 红球 {pred['red_stats']['expected_hits']:.2f} | 蓝球 {pred['blue_stats']['expected_hits']:.2f}\n")

                result_text.insert(tk.END, "\n\n二、概率排名TOP10\n")
                result_text.insert(tk.END, "-" * 70 + "\n")

                result_text.insert(tk.END, "\n红球TOP10:\n")
                top10 = select_hot_numbers(red_probs, 10)
                for i, (num, prob) in enumerate(top10, 1):
                    deviation = (prob - 1 / 33) / (1 / 33) * 100
                    sign = '+' if deviation > 0 else ''
                    result_text.insert(tk.END, f"  {i:2d}. {num:02d}: {prob:.5f} ({sign}{deviation:.1f}%)\n")

                result_text.insert(tk.END, "\n蓝球TOP5:\n")
                top5_blue = select_hot_numbers(blue_probs, 5)
                for i, (num, prob) in enumerate(top5_blue, 1):
                    deviation = (prob - 1 / 16) / (1 / 16) * 100
                    sign = '+' if deviation > 0 else ''
                    result_text.insert(tk.END, f"  {i:2d}. {num:02d}: {prob:.5f} ({sign}{deviation:.1f}%)\n")

                result_text.insert(tk.END, "\n\n三、输出文件\n")
                result_text.insert(tk.END, "-" * 70 + "\n")
                result_text.insert(tk.END, f"📁 {OUTPUT_DIR}\n")
                result_text.insert(tk.END, "  ├── dan_tuo_pools.png      (综合分析图)\n")
                result_text.insert(tk.END, "  ├── fitting_curve.png      (拟合曲线图)\n")
                result_text.insert(tk.END, "  ├── red_heatmap.png        (红球热力图)\n")
                result_text.insert(tk.END, "  ├── blue_radar.png         (蓝球雷达图)\n")
                result_text.insert(tk.END, "  ├── dan_tuo_predictions.txt\n")
                result_text.insert(tk.END, "  └── dan_tuo_predictions.csv\n")

                result_text.insert(tk.END, "\n\n⚠️ 风险提示\n")
                result_text.insert(tk.END, "-" * 70 + "\n")
                result_text.insert(tk.END, "预测结果仅供参考，请理性购彩！\n")
                result_text.insert(tk.END, "彩票具有随机性，任何预测方法都不能保证准确。\n")

                # 切换到结果选项卡
                self.root.after(0, lambda: self.notebook.select(self.result_tab))

                self.root.after(0, lambda: self.predict_status.config(
                    text="✅ 预测完成", foreground="green"))

                # 显示第一张图
                self.root.after(0, lambda: self.show_chart())

            except Exception as e:
                import traceback
                self.root.after(0, lambda: self.predict_status.config(
                    text=f"❌ 错误: {e}", foreground="red"))
                self.root.after(0, lambda: self.result_text.insert(tk.END, f"\n错误: {e}\n{traceback.format_exc()}"))
            finally:
                self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

        threading.Thread(target=do_prediction, daemon=True).start()

    def show_chart(self):
        """显示选中的图表"""
        chart_name = self.chart_var.get()
        chart_path = f"{OUTPUT_DIR}/{chart_name}.png"

        if not os.path.exists(chart_path):
            self.chart_label.config(text="请先运行预测生成图表", foreground="gray")
            self.chart_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            return

        # 清除旧图片
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # 加载新图片
        try:
            img = Image.open(chart_path)

            # 计算合适的尺寸
            frame_width = self.chart_frame.winfo_width() - 20
            frame_height = self.chart_frame.winfo_height() - 20

            if frame_width <= 0:
                frame_width = 800
            if frame_height <= 0:
                frame_height = 500

            # 按比例缩放
            img_width, img_height = img.size
            ratio = min(frame_width / img_width, frame_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            img = img.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            # 显示图片
            label = ttk.Label(self.chart_frame, image=photo)
            label.image = photo  # 保持引用
            label.pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            self.chart_label.config(text=f"加载图片失败: {e}", foreground="red")
            self.chart_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def open_image_folder(self):
        """打开图片文件夹"""
        if os.path.exists(OUTPUT_DIR):
            os.startfile(OUTPUT_DIR)
        else:
            messagebox.showwarning("提示", f"文件夹不存在: {OUTPUT_DIR}")


# ============== 主程序入口 ==============
def main():
    root = tk.Tk()
    app = LotteryPredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
