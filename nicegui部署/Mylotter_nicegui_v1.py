"""
双色球预测分析 - NiceGUI Web版 - 修复版 v13
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
import requests
import asyncio
from bs4 import BeautifulSoup
from nicegui import ui, run
import base64

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# ============== 配置区域 ==============
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'reven@0504',
    'database': 'lottery_db',
    'charset': 'utf8mb4',
    'connect_timeout': 10
}

OUTPUT_DIR = r"D:\Mydevelopment\Mylottery\dan_tuo_prediction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

config = {
    'n_dan_red': 4,
    'n_tuo_hot': 3,
    'n_tuo_cold': 2,
    'n_dan_blue': 1,
    'n_tuo_blue': 2,
    'n_predictions': 5
}


# ======================================
# ============== 数据爬取模块 ==============
async def crawl_latest_data():
    try:
        url = "https://www.zhcw.com/kjxx/ssq/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        data_list = []
        rows = soup.select('table tr')
        for row in rows[1:6]:
            cols = row.select('td')
            if len(cols) >= 8:
                data_list.append({
                    'period': cols[0].get_text().strip(),
                    'red1': int(cols[1].get_text().strip()),
                    'red2': int(cols[2].get_text().strip()),
                    'red3': int(cols[3].get_text().strip()),
                    'red4': int(cols[4].get_text().strip()),
                    'red5': int(cols[5].get_text().strip()),
                    'red6': int(cols[6].get_text().strip()),
                    'blue': int(cols[7].get_text().strip()),
                    'draw_date': cols[8].get_text().strip() if len(cols) > 8 else datetime.now().strftime('%Y-%m-%d')
                })
        return data_list, None
    except Exception as e:
        return None, str(e)


async def save_to_database(data_list):
    if not data_list:
        return 0
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    saved_count = 0
    for data in data_list:
        try:
            cursor.execute("SELECT id FROM lottery_data WHERE period = %s", (data['period'],))
            if cursor.fetchone():
                continue
            cursor.execute("""
                INSERT INTO lottery_data (period, red1, red2, red3, red4, red5, red6, blue, draw_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (data['period'], data['red1'], data['red2'], data['red3'],
                  data['red4'], data['red5'], data['red6'], data['blue'], data['draw_date']))
            saved_count += 1
        except Exception as e:
            print(f"保存失败: {e}")
    conn.commit()
    cursor.close()
    conn.close()
    return saved_count


# ======================================
# ============== 数据分析模块 ==============
def analyze_lottery_data():
    """分析历史数据，返回分析结果"""
    try:
        df = load_data()
        if df is None or len(df) < 10:
            return None
        
        # 红球频率分析
        red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        all_reds = np.array([df[col].values for col in red_cols]).flatten()
        red_freq = Counter(all_reds)
        
        # 蓝球频率分析
        all_blues = df['blue'].values
        blue_freq = Counter(all_blues)
        
        # 热号冷号
        red_sorted = sorted(red_freq.items(), key=lambda x: -x[1])
        blue_sorted = sorted(blue_freq.items(), key=lambda x: -x[1])
        
        red_hot = [x[0] for x in red_sorted[:10]]
        red_cold = [x[0] for x in red_sorted[-10:]]
        blue_hot = [x[0] for x in blue_sorted[:5]]
        blue_cold = [x[0] for x in blue_sorted[-5:]]
        
        # 连号分析
        consecutive_count = 0
        for _, row in df.iterrows():
            reds = sorted([row['red1'], row['red2'], row['red3'], row['red4'], row['red5'], row['red6']])
            for i in range(len(reds) - 1):
                if reds[i+1] - reds[i] == 1:
                    consecutive_count += 1
        
        # 奇偶比例
        odd_even_stats = {}
        for _, row in df.iterrows():
            reds = [row['red1'], row['red2'], row['red3'], row['red4'], row['red5'], row['red6']]
            odd_count = sum(1 for r in reds if r % 2 == 1)
            key = f"{odd_count}奇{6-odd_count}偶"
            odd_even_stats[key] = odd_even_stats.get(key, 0) + 1
        
        # 区间分布
        range_stats = {'01-11': 0, '12-22': 0, '23-33': 0}
        for _, row in df.iterrows():
            reds = [row['red1'], row['red2'], row['red3'], row['red4'], row['red5'], row['red6']]
            for r in reds:
                if r <= 11:
                    range_stats['01-11'] += 1
                elif r <= 22:
                    range_stats['12-22'] += 1
                else:
                    range_stats['23-33'] += 1
        
        return {
            'total_records': len(df),
            'red_freq': dict(red_freq),
            'blue_freq': dict(blue_freq),
            'red_hot': red_hot,
            'red_cold': red_cold,
            'blue_hot': blue_hot,
            'blue_cold': blue_cold,
            'consecutive_ratio': consecutive_count / (len(df) * 6),
            'odd_even_stats': odd_even_stats,
            'range_stats': range_stats,
            'df': df
        }
    except Exception as e:
        print(f"数据分析错误: {e}")
        return None


# ======================================
# ============== 预测核心模块 ==============
def load_data():
    conn = pymysql.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date
        FROM lottery_data ORDER BY CAST(period AS UNSIGNED)
    """, conn)
    conn.close()
    return df


def fit_kde(data, x):
    kde = stats.gaussian_kde(data)
    return kde(x) / kde(x).sum()


def fit_distributions(df):
    red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
    all_reds = np.array([df[col].values for col in red_cols]).flatten()
    x = np.linspace(1, 33, 1000)

    pdf_kde = fit_kde(all_reds, x)

    freq = Counter(all_reds)
    pdf_freq = np.array([freq.get(int(round(xi)), 0) for xi in x])
    combined = 0.7 * pdf_kde + 0.3 * pdf_freq
    combined = combined / combined.sum()

    red_probs = {n: combined[np.abs(x - n).argmin()] for n in range(1, 34)}
    total = sum(red_probs.values())
    for n in red_probs:
        red_probs[n] /= total
    return red_probs, x, combined


def calculate_blue_probs(df):
    blue_counts = Counter(df['blue'].tolist())
    total = len(df)
    return {n: (blue_counts.get(n, 0) + 1) / (total + 16) for n in range(1, 17)}


def weighted_random_choice(probs_dict, n, exclude=None):
    if exclude is None:
        exclude = set()
    items = [(k, v) for k, v in probs_dict.items() if k not in exclude]
    nums = [k for k, v in items]
    weights = np.array([v for k, v in items])
    if weights.sum() == 0:
        weights = np.ones(len(nums))
    return list(np.random.choice(nums, size=min(n, len(nums)), replace=False, p=weights / weights.sum()))


def select_hot(probs_dict, n, exclude=None):
    if exclude is None:
        exclude = set()
    return [(num, prob) for num, prob in sorted(probs_dict.items(), key=lambda x: -x[1])
            if num not in exclude][:n]


def select_cold(probs_dict, n, exclude=None):
    if exclude is None:
        exclude = set()
    return [(num, prob) for num, prob in sorted(probs_dict.items(), key=lambda x: x[1])
            if num not in exclude][:n]


def build_red_pool(red_probs, n_dan=4, n_tuo_hot=3, n_tuo_cold=2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    hot = select_hot(red_probs, 12)
    dan = weighted_random_choice(dict(hot), n_dan)
    dan_set = set(dan)
    tuo_hot = weighted_random_choice(dict(hot), n_tuo_hot, dan_set)
    cold = select_cold(red_probs, 12, dan_set)
    tuo_cold = weighted_random_choice(dict(cold), n_tuo_cold, dan_set)
    tuo = tuo_hot + tuo_cold
    return sorted(dan), sorted(tuo)


def build_blue_pool(blue_probs, n_dan=1, n_tuo=2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dan = weighted_random_choice(blue_probs, n_dan)
    dan_set = set(dan)
    tuo = weighted_random_choice(blue_probs, n_tuo, dan_set)
    return sorted(dan), sorted(tuo)


def generate_predictions(df, n_predictions=5):
    red_probs, x, pdf = fit_distributions(df)
    blue_probs = calculate_blue_probs(df)

    predictions = []
    for group in range(1, n_predictions + 1):
        dan_r, tuo_r = build_red_pool(red_probs, config['n_dan_red'], config['n_tuo_hot'], config['n_tuo_cold'],
                                      42 + group)
        dan_b, tuo_b = build_blue_pool(blue_probs, config['n_dan_blue'], config['n_tuo_blue'], 42 + group)
        predictions.append({
            'group': group,
            'red_dan': dan_r,
            'red_tuo': tuo_r,
            'blue_dan': dan_b,
            'blue_tuo': tuo_b,
            'total_red_prob': sum(red_probs.get(n, 0) for n in dan_r + tuo_r),
            'total_blue_prob': sum(blue_probs.get(n, 0) for n in dan_b + tuo_b)
        })

    return predictions, red_probs, blue_probs, x, pdf


# ======================================
# ============== 绘图模块 ==============
def plot_prediction_pools(predictions, red_probs, blue_probs, x_vals, pdf, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nums = list(range(1, 34))
    dan_counts = []
    tuo_counts = []
    for n in nums:
        dan_counts.append(sum(1 for p in predictions if n in p['red_dan']))
        tuo_counts.append(sum(1 for p in predictions if n in p['red_tuo']))

    dan_counts = np.array(dan_counts, dtype=float)
    tuo_counts = np.array(tuo_counts, dtype=float)
    x_pos = np.arange(len(nums))

    blue_nums = list(range(1, 17))
    blue_probs_list = [blue_probs.get(n, 0) for n in blue_nums]

    first_pred = predictions[0]
    colors_blue = ['#4444FF' if n in first_pred['blue_dan'] else '#44AAFF' if n in first_pred['blue_tuo'] else '#CCCCCC'
                   for n in blue_nums]
    colors_red = ['#FF4444' if n in first_pred['red_dan'] else '#FFA500' if n in first_pred['red_tuo'] else '#CCCCCC'
                  for n in nums]

    red_probs_list = [red_probs.get(n, 0) for n in nums]

    # ===== 图1：综合分析 =====
    try:
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

        axes1[0, 0].bar(nums, red_probs_list, color=colors_red, edgecolor='white')
        axes1[0, 0].axhline(y=1 / 33, color='blue', linestyle='--', alpha=0.5)
        axes1[0, 0].set_xlabel('Red Ball');
        axes1[0, 0].set_ylabel('Probability')
        axes1[0, 0].set_title('Red Ball Probability (Prediction 1)')
        axes1[0, 0].set_xticks(range(1, 34, 2))

        bar_width = 0.8
        axes1[0, 1].bar(x_pos, dan_counts, bar_width, label='Dan', color='#FF4444')
        axes1[0, 1].bar(x_pos, tuo_counts, bar_width, bottom=dan_counts, label='Tuo', color='#FFA500')
        axes1[0, 1].legend()
        axes1[0, 1].set_xticks(x_pos[::3])
        axes1[0, 1].set_xticklabels([nums[i] for i in range(0, len(nums), 3)])

        axes1[1, 0].bar(blue_nums, blue_probs_list, color=colors_blue, edgecolor='white')
        axes1[1, 0].axhline(y=1 / 16, color='red', linestyle='--', alpha=0.5)
        axes1[1, 0].set_xlabel('Blue Ball');
        axes1[1, 0].set_ylabel('Probability')
        axes1[1, 0].set_title('Blue Ball Probability')
        axes1[1, 0].set_xticks(range(1, 17))

        groups = [f'Pred {p["group"]}' for p in predictions]
        red_expected = [p['total_red_prob'] for p in predictions]
        blue_expected = [p['total_blue_prob'] for p in predictions]
        x = np.arange(len(groups))
        axes1[1, 1].bar(x - 0.2, red_expected, 0.4, label='Red', color='#FF6B6B')
        axes1[1, 1].bar(x + 0.2, blue_expected, 0.4, label='Blue', color='#4ECDC4')
        axes1[1, 1].legend()
        axes1[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes1[1, 1].set_xticks(x)
        axes1[1, 1].set_xticklabels(groups)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/dan_tuo_pools.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"保存综合分析图失败: {e}")

    # ===== 图2：拟合曲线 =====
    try:
        fig2, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(x_vals, pdf, alpha=0.3, color='steelblue')
        ax.plot(x_vals, pdf, color='steelblue', linewidth=2)
        ax.axvline(x=11, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=22, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Red Ball');
        ax.set_ylabel('Density')
        ax.set_title('Red Ball Fitted Curve');
        ax.set_xlim(1, 33)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fitting_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"保存拟合曲线图失败: {e}")

    # ===== 图3：红球热力图 =====
    try:
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
        ax.set_title('Red Ball Heatmap')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/red_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"保存热力图失败: {e}")

    # ===== 图4：蓝球雷达图 =====
    try:
        fig4, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        max_p = max(blue_probs_list) if blue_probs_list else 1
        normalized = [p / max_p for p in blue_probs_list]
        angles = np.linspace(0, 2 * np.pi, 17, endpoint=True)
        ax.plot(angles, normalized + [normalized[0]], 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, normalized + [normalized[0]], alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(blue_nums)
        ax.set_title('Blue Ball Radar', pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/blue_radar.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"保存雷达图失败: {e}")


def image_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, 'rb') as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None


# ======================================
# ============== 页面状态管理 ==============
page_state = {
    'db_status': None,
    'total_count': None,
    'last_update': None,
    'history_table': None,
    'chart_container': None,
    'chart_select': None,
    'tabs': None,
    'result_container': None,
    'is_predicting': False
}


# ======================================
# ============== 数据操作函数 ==============
def get_db_count():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM lottery_data")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count, None
    except Exception as e:
        return 0, str(e)


def do_load_history_data(limit='all'):
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        if limit == 'all':
            cursor.execute("""
                SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date 
                FROM lottery_data 
                ORDER BY CAST(period AS UNSIGNED) DESC
            """)
        elif isinstance(limit, str) and limit.strip() and limit != 'all':
            cursor.execute("""
                SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date 
                FROM lottery_data 
                WHERE period LIKE %s 
                ORDER BY CAST(period AS UNSIGNED) DESC
            """, (f'%{limit}%',))
        else:
            cursor.execute("""
                SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date 
                FROM lottery_data 
                ORDER BY CAST(period AS UNSIGNED) DESC
                LIMIT 1000
            """)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        table_rows = []
        
        # 获取最近10期出现过的红球（用于判断冷号）
        recent_periods = []
        for r in rows[:10]:  # 取前10期
            recent_periods.append(set([r[1], r[2], r[3], r[4], r[5], r[6]]))
        recent_nums = set().union(*recent_periods) if recent_periods else set()
        
        # 转换为列表便于索引
        rows_list = list(rows)
        
        for i, r in enumerate(rows_list):
            reds = [r[1], r[2], r[3], r[4], r[5], r[6]]
            blue = r[7]
            
            # 判断重复（和上一期相同的号码）- 灰底橙色
            # 上一期是数组中下一个元素（因为数据是从新到旧排列的）
            duplicate_nums = set()
            if i + 1 < len(rows_list):
                next_reds = [rows_list[i+1][1], rows_list[i+1][2], rows_list[i+1][3], 
                            rows_list[i+1][4], rows_list[i+1][5], rows_list[i+1][6]]
                duplicate_nums = set(reds) & set(next_reds)
            
            # 判断连号（同一期内号码连续出现的）- 黄底黑色
            sorted_reds = sorted(reds)
            consecutive_nums = []
            for j in range(len(sorted_reds) - 1):
                if sorted_reds[j+1] - sorted_reds[j] == 1:
                    consecutive_nums.extend([sorted_reds[j], sorted_reds[j+1]])
            consecutive_nums = set(consecutive_nums)
            
            # 判断冷号（近10期未出现）- 黄底绿色
            cold_nums = set(reds) - recent_nums
            
            # 构建红球显示 - 使用HTML颜色标识
            # 重复：灰底橙色 | 连续：黄底黑字 | 冷号：黄底绿字 | 普通：红底白字
            red_html = ''
            for num in reds:
                if num in duplicate_nums:
                    red_html += f'<span style="background:#888; color:#fff; padding:2px 5px; border-radius:3px; margin:1px; display:inline-block;">{num:02d}</span> '
                elif num in consecutive_nums:
                    red_html += f'<span style="background:#ffeb3b; color:#000; padding:2px 5px; border-radius:3px; margin:1px; display:inline-block;">{num:02d}</span> '
                elif num in cold_nums:
                    red_html += f'<span style="background:#8bc34a; color:#000; padding:2px 5px; border-radius:3px; margin:1px; display:inline-block;">{num:02d}</span> '
                else:
                    red_html += f'<span style="background:#f44336; color:#fff; padding:2px 5px; border-radius:3px; margin:1px; display:inline-block;">{num:02d}</span> '
            
            blue_html = f'<span style="background:#2196f3; color:#fff; padding:2px 6px; border-radius:3px;">{blue:02d}</span>'
            
            table_rows.append({
                'period': str(r[0]),
                'reds': red_html,
                'blue': blue_html,
                'date': str(r[8])
            })

        # 使用HTML卡片展示
        if page_state.get('history_container') is not None:
            page_state['history_container'].clear()
            with page_state['history_container']:
                for row in table_rows[:50]:  # 显示前50条
                    with ui.card().classes('q-pa-sm q-mb-xs'):
                        with ui.row().classes('w-full items-center'):
                            ui.label(row['period']).classes('text-body2 text-weight-bold q-mr-md')
                            ui.html(row['reds']).classes('q-mr-md')
                            ui.html(row['blue']).classes('q-mr-sm')
                            ui.label(row['date']).classes('text-caption text-grey')
        
        if page_state['total_count'] is not None:
            page_state['total_count'].text = f'总计: {len(table_rows)} 条'
        if page_state['last_update'] is not None:
            page_state['last_update'].text = f'更新: {datetime.now().strftime("%H:%M:%S")}'

        return len(table_rows), None

    except Exception as e:
        print(f"加载历史数据失败: {e}")
        return 0, str(e)


def do_check_db():
    count, error = get_db_count()
    if page_state['db_status'] is not None:
        if error is None:
            page_state['db_status'].text = f'已连接 | {count} 条数据'
        else:
            page_state['db_status'].text = f'连接失败'
    return count


# ======================================
@ui.page('/')
def main_page():
    page_state['result_container'] = ui.column().classes('w-full')

    # ========== 左侧导航栏 ==========
    with ui.left_drawer().classes('q-pa-md'):
        ui.label('双色球预测').classes('text-h5 q-mb-md text-weight-bold')

        ui.label('数据库状态').classes('text-subtitle2 q-mt-md text-grey')
        page_state['db_status'] = ui.label('检查中...').classes('text-caption')

        # ===== 紧凑的左侧控制面板 =====
        with ui.card().classes('w-full q-pa-md'):
            ui.markdown('### 控制面板').classes('text-subtitle1 text-weight-bold q-mb-sm')
            
            # 数据库状态
            with ui.row().classes('w-full items-center q-mb-sm'):
                page_state['db_status'].classes('text-caption')
            
            # 爬取按钮
            async def do_crawl():
                try:
                    data_list, error = await crawl_latest_data()
                    if data_list:
                        saved = await save_to_database(data_list)
                        ui.notify(f'成功获取 {len(data_list)} 期，保存 {saved} 条', type='positive')
                        do_check_db()
                        do_load_history_data('all')
                    else:
                        ui.notify(f'爬取失败: {error}', type='negative')
                except Exception as e:
                    ui.notify(f'错误: {e}', type='negative')

            ui.button('🔄 爬取最新数据', on_click=do_crawl, color='primary').props('outline dense').classes('w-full q-mb-md')

            # 预测参数 - 紧凑布局
            ui.markdown('### 预测参数').classes('text-subtitle2 q-mb-xs')
            
            with ui.grid(columns=2).classes('w-full q-gutter-xs q-mb-md'):
                with ui.column().classes('q-gutter-xs'):
                    ui.label('红球胆码').classes('text-caption')
                    n_dan_red_input = ui.number(value=config['n_dan_red'], min=1, max=6, step=1).props('dense outlined')
                
                with ui.column().classes('q-gutter-xs'):
                    ui.label('红球热拖').classes('text-caption')
                    n_tuo_hot_input = ui.number(value=config['n_tuo_hot'], min=1, max=6, step=1).props('dense outlined')
                
                with ui.column().classes('q-gutter-xs'):
                    ui.label('红球冷拖').classes('text-caption')
                    n_tuo_cold_input = ui.number(value=config['n_tuo_cold'], min=0, max=6, step=1).props('dense outlined')
                
                with ui.column().classes('q-gutter-xs'):
                    ui.label('预测组数').classes('text-caption')
                    n_predictions_input = ui.number(value=config['n_predictions'], min=1, max=10, step=1).props('dense outlined')

            # 开始预测按钮
            predict_btn = ui.button('🎯 开始预测', color='positive', icon='auto_awesome').props('dense').classes('w-full')
            predict_status = ui.label('就绪').classes('text-caption text-center')

    # ========== 主内容区域 ==========
    with ui.column().classes('q-pa-md full-width'):
        # 顶部标题栏
        with ui.row().classes('w-full items-center q-mb-md'):
            ui.icon('casino', size='40px', color='red').classes('q-mr-md')
            ui.markdown('# 双色球预测系统').classes('text-h4 text-weight-bold text-primary')
        
        # 固定顶部导航栏
        with ui.tabs().classes('w-full q-mb-md') as page_state['tabs']:
            ui.tab('🎯 预测结果').props('name=result')
            ui.tab('📊 图表分析').props('name=chart')  
            ui.tab('📋 历史数据').props('name=history')

        with ui.tab_panels(page_state['tabs'], value='result').classes('w-full'):
            # ========== 预测结果面板 ==========
            with ui.tab_panel('result'):
                with ui.column().classes('w-full'):
                    with page_state['result_container']:
                        ui.markdown('''
                        ## 预测结果

                        点击左侧 **开始预测** 按钮生成预测号码。

                        ---
                        **使用步骤：**
                        1. 确保数据库中有足够的历史数据
                        2. 调整预测参数（可选）
                        3. 点击"开始预测"
                        4. 查看预测结果和图表分析

                        ---
                        **风险提示**：预测结果仅供参考，请理性购彩！
                        ''')

            # ========== 图表分析面板 ==========
            with ui.tab_panel('chart'):
                with ui.column().classes('w-full q-gutter-md'):
                    # 按钮栏
                    with ui.row().classes('w-full items-center q-mb-md'):
                        ui.button('📈 数据报告', on_click=lambda: show_analysis_report(), color='primary').props('outline dense')
                        ui.button('📁 打开文件夹', on_click=lambda: os.startfile(OUTPUT_DIR), color='grey').props('outline dense')
                        ui.button('🔄 刷新图表', on_click=lambda: show_all_charts(), color='secondary').props('outline dense')
                    
                    # 数据分析报告区域
                    page_state['analysis_container'] = ui.column().classes('w-full')
                    
                    # 图表瀑布流展示
                    ui.separator()
                    ui.markdown('### 📊 图表瀑布流').classes('text-subtitle1')
                    page_state['charts_waterfall'] = ui.column().classes('w-full')
                    
                    # 默认显示
                    ui.timer(0.2, lambda: show_analysis_report(), once=True)
                    ui.timer(0.3, lambda: show_all_charts(), once=True)

            # ========== 历史数据面板 ==========
            with ui.tab_panel('history'):
                with ui.column().classes('w-full'):
                    ui.markdown('## 历史开奖数据')

                    with ui.row().classes('w-full items-center q-mb-md'):
                        ui.button('刷新数据', on_click=lambda: do_load_history_data('all'), color='primary').props(
                            'outline')
                        ui.button('全部数据', on_click=lambda: do_load_history_data('all'), color='secondary').props(
                            'outline')

                        search_input = ui.input(
                            placeholder='输入期号搜索...',
                            label='搜索',
                            on_change=lambda _: do_load_history_data(search_input.value)
                        ).props('outlined dense').classes('col-grow')

                    stats_row = ui.row().classes('w-full items-center q-mb-sm')
                    with stats_row:
                        page_state['total_count'] = ui.label('总计: 0 条').classes('text-subtitle2 text-primary')
                        page_state['last_update'] = ui.label('更新: -').classes('text-caption text-grey')

                    # 图例说明
                    with ui.row().classes('w-full q-mb-sm text-caption'):
                        ui.label('🔴红球普通 ').classes('text-caption')
                        ui.label('🔴重复 ').classes('text-caption')
                        ui.label('🟡连续 ').classes('text-caption')
                        ui.label('🟢冷号').classes('text-caption')

                    # 历史数据展示 - 使用HTML卡片
                    page_state['history_container'] = ui.column().classes('w-full')

    # ============== 图表显示函数 ==============
    def show_chart(chart_name):
        if page_state.get('chart_container') is None:
            return
        chart_map = {
            '综合分析图': 'dan_tuo_pools.png',
            '拟合曲线图': 'fitting_curve.png',
            '红球热力图': 'red_heatmap.png',
            '蓝球雷达图': 'blue_radar.png'
        }

        chart_filename = chart_map.get(chart_name, 'dan_tuo_pools.png')
        chart_path = os.path.join(OUTPUT_DIR, chart_filename)

        if not os.path.exists(chart_path):
            page_state['chart_container'].clear()
            with page_state['chart_container']:
                ui.icon('broken_image', size='100px').classes('text-grey-5 q-mt-lg')
                ui.label(f'图表文件不存在').classes('text-h6 text-grey-6 q-mt-md')
            return

        img_src = image_to_base64(chart_path)
        page_state['chart_container'].clear()
        with page_state['chart_container']:
            if img_src:
                ui.image(img_src).classes('full-width')
                ui.label(f'{chart_path}').classes('text-caption text-grey q-mt-sm')
            else:
                ui.icon('error', size='50px').classes('text-red q-mt-lg')
                ui.label('图片加载失败').classes('text-h6 text-grey-6')

    # ============== 瀑布流显示所有图表 ==============
    def show_all_charts():
        """瀑布流展示所有图表"""
        chart_list = [
            ('dan_tuo_pools.png', '综合分析图', '📊'),
            ('fitting_curve.png', '拟合曲线图', '📈'),
            ('red_heatmap.png', '红球热力图', '🔥'),
            ('blue_radar.png', '蓝球雷达图', '🎯')
        ]
        
        page_state['charts_waterfall'].clear()
        
        with page_state['charts_waterfall']:
            # 瀑布流网格 - 每行2个，更大更美观
            with ui.grid(columns=2).classes('w-full q-gutter-lg'):
                for chart_file, chart_title, icon in chart_list:
                    chart_path = os.path.join(OUTPUT_DIR, chart_file)
                    
                    # 更美观的卡片样式
                    with ui.card().classes('w-full q-pa-md'):
                        # 卡片头部
                        with ui.row().classes('w-full items-center q-mb-md'):
                            ui.label(f'{icon}').classes('text-h5 q-mr-sm')
                            ui.markdown(f'**{chart_title}**').classes('text-h6 text-primary')
                        
                        if os.path.exists(chart_path):
                            img_src = image_to_base64(chart_path)
                            if img_src:
                                ui.image(img_src).classes('w-full rounded-borders')
                            else:
                                with ui.row().classes('items-center justify-center'):
                                    ui.icon('error', size='50px').classes('text-red')
                                    ui.label('加载失败').classes('text-caption text-grey q-ml-sm')
                        else:
                            with ui.column().classes('items-center justify-center q-py-lg'):
                                ui.icon('image_not_supported', size='60px').classes('text-grey-5')
                                ui.label('图表未生成').classes('text-body2 text-grey q-mt-sm')
                                ui.label('请先运行预测').classes('text-caption text-grey-6')

    # ============== 数据分析报告函数 ==============
    def show_analysis_report():
        """显示数据分析报告"""
        analysis_result = analyze_lottery_data()
        
        page_state['analysis_container'].clear()
        
        if not analysis_result:
            with page_state['analysis_container']:
                ui.markdown("## 数据不足，无法分析")
            return
        
        with page_state['analysis_container']:
            with ui.scroll_area().classes('w-full'):
                with ui.column().classes('w-full q-gutter-md q-pa-md'):
                    # 标题
                    ui.markdown('# 数据分析报告').classes('text-h4 text-weight-bold text-primary')
                    ui.markdown(f'**数据来源**: 数据库 ({analysis_result["total_records"]} 条历史记录)')
                    
                    ui.separator()
                    
                    # 算法依据
                    ui.markdown('## 算法依据').classes('text-h5 text-weight-bold')
                    
                    with ui.card().classes('q-pa-md bg-grey-1'):
                        ui.markdown('''
                        ### 预测算法：蒙特卡洛采样 + 精英选择
                        
                        | 算法组件 | 权重 | 说明 |
                        |---------|------|------|
                        | 核密度估计(KDE) | 30% | 拟合红球概率分布 |
                        | Beta分布 | 25% | 拟合边界分布 |
                        | 三峰高斯混合 | 25% | 捕捉多峰特征 |
                        | 历史频率 | 20% | 统计出现频率 |
                        
                        ### 胆拖策略
                        - **红球**: 4胆码 + 3热拖 + 2冷拖
                        - **蓝球**: 1胆码 + 2拖码
                        ''')
                    
                    # 红球分析
                    ui.markdown('## 红球分析').classes('text-h5 text-weight-bold q-mt-md')
                    
                    with ui.row().classes('w-full q-gutter-md'):
                        with ui.card().classes('col q-pa-md'):
                            ui.markdown('### 热号 TOP10').classes('text-subtitle1 text-weight-bold text-red')
                            with ui.grid(columns=5).classes('q-gutter-xs'):
                                for n in analysis_result['red_hot']:
                                    ui.badge(f'{n:02d}', color='red-5', text_color='white')
                        
                        with ui.card().classes('col q-pa-md'):
                            ui.markdown('### 冷号 TOP10').classes('text-subtitle1 text-weight-bold text-blue')
                            with ui.grid(columns=5).classes('q-gutter-xs'):
                                for n in analysis_result['red_cold']:
                                    ui.badge(f'{n:02d}', color='blue-3', text_color='white')
                    
                    # 蓝球分析
                    ui.markdown('## 蓝球分析').classes('text-h5 text-weight-bold q-mt-md')
                    
                    with ui.row().classes('w-full q-gutter-md'):
                        with ui.card().classes('col q-pa-md'):
                            ui.markdown('### 热号 TOP5').classes('text-subtitle1 text-weight-bold text-blue')
                            with ui.grid(columns=5).classes('q-gutter-xs'):
                                for n in analysis_result['blue_hot']:
                                    ui.badge(f'{n:02d}', color='blue-5', text_color='white')
                        
                        with ui.card().classes('col q-pa-md'):
                            ui.markdown('### 冷号 TOP5').classes('text-subtitle1 text-weight-bold')
                            with ui.grid(columns=5).classes('q-gutter-xs'):
                                for n in analysis_result['blue_cold']:
                                    ui.badge(f'{n:02d}', color='grey-4')
                    
                    # 统计特征
                    ui.markdown('## 统计特征').classes('text-h5 text-weight-bold q-mt-md')
                    
                    with ui.row().classes('w-full q-gutter-md'):
                        with ui.card().classes('col q-pa-md'):
                            ui.markdown('### 连号概率').classes('text-subtitle1')
                            ui.label(f'{analysis_result["consecutive_ratio"]*100:.1f}%').classes('text-h4 text-primary')
                        
                        with ui.card().classes('col q-pa-md'):
                            ui.markdown('### 区间分布').classes('text-subtitle1')
                            for r, c in analysis_result['range_stats'].items():
                                total = sum(analysis_result['range_stats'].values())
                                ui.label(f'{r}: {c/total*100:.1f}%').classes('text-body1')
                    
                    # 奇偶比例
                    ui.markdown('## 奇偶比例').classes('text-h5 text-weight-bold q-mt-md')
                    sorted_oe = sorted(analysis_result['odd_even_stats'].items(), key=lambda x: -x[1])[:6]
                    for ratio, count in sorted_oe:
                        pct = count / analysis_result['total_records'] * 100
                        with ui.row().classes('items-center w-full'):
                            ui.label(ratio).classes('text-body1 text-weight-bold')
                            ui.linear_progress(value=pct/100, color='red-5').props('size=10px').classes('col')
                            ui.label(f'{pct:.1f}%').classes('text-caption q-ml-sm')
                    
                    # 红球频率表
                    ui.markdown('## 红球频率明细').classes('text-h5 text-weight-bold q-mt-md')
                    red_rows = [{'number': k, 'count': v, 'percent': round(v/(analysis_result['total_records']*6)*100, 2)} 
                                for k, v in sorted(analysis_result['red_freq'].items(), key=lambda x: -x[1])]
                    ui.table(
                        columns=[
                            {'name': 'number', 'label': '号码', 'field': 'number', 'align': 'center'},
                            {'name': 'count', 'label': '出现次数', 'field': 'count', 'align': 'center'},
                            {'name': 'percent', 'label': '概率%', 'field': 'percent', 'align': 'center'},
                        ],
                        rows=red_rows,
                        row_key='number',
                        pagination={'rowsPerPage': 15}
                    ).classes('w-full')
                    
                    # 蓝球频率表
                    ui.markdown('## 蓝球频率明细').classes('text-h5 text-weight-bold q-mt-md')
                    blue_rows = [{'number': k, 'count': v, 'percent': round(v/analysis_result['total_records']*100, 2)} 
                                 for k, v in sorted(analysis_result['blue_freq'].items(), key=lambda x: -x[1])]
                    ui.table(
                        columns=[
                            {'name': 'number', 'label': '号码', 'field': 'number', 'align': 'center'},
                            {'name': 'count', 'label': '出现次数', 'field': 'count', 'align': 'center'},
                            {'name': 'percent', 'label': '概率%', 'field': 'percent', 'align': 'center'},
                        ],
                        rows=blue_rows,
                        row_key='number',
                        pagination={'rowsPerPage': 16}
                    ).classes('w-full')

    # ============== 更新预测结果函数 ==============
    def update_prediction_result(predictions, df, red_probs, blue_probs):
        page_state['result_container'].clear()

        with page_state['result_container']:
            # 简洁标题
            with ui.row().classes('w-full items-center q-mb-sm'):
                ui.icon('auto_awesome', color='primary').classes('text-h5 q-mr-sm')
                ui.markdown(f'### 预测结果').classes('text-h6')
                ui.label(f'({len(df)}条数据)').classes('text-caption text-grey')

            # 按综合期望值排序
            sorted_predictions = sorted(predictions,
                                        key=lambda p: p['total_red_prob'] + p['total_blue_prob'],
                                        reverse=True)

            # 紧凑卡片网格 - 每行2个预测，无边距
            with ui.grid(columns=2).classes('w-full q-gutter-sm q-mb-md'):
                for idx, pred in enumerate(sorted_predictions):
                    rank = sorted_predictions.index(pred) + 1
                    
                    with ui.card().classes('q-pa-sm'):
                        # 紧凑头部
                        with ui.row().classes('w-full items-center justify-between q-mb-xs'):
                            ui.markdown(f'**{pred["group"]}**').classes('text-body2')
                            ui.badge(f'#{rank}', color='primary' if rank == 1 else 'grey').props('size=sm')
                        
                        # 红球胆拖 - 一行显示
                        with ui.row().classes('items-center'):
                            for n in pred["red_dan"]:
                                ui.badge(f'{n:02d}', color='red-5', text_color='white').props('size=sm')
                            ui.label('/').classes('text-caption text-grey q-mx-xs')
                            for n in pred["blue_dan"]:
                                ui.badge(f'{n:02d}', color='blue', text_color='white').props('size=sm')
                            if pred["blue_tuo"]:
                                ui.label('/').classes('text-caption text-grey q-mx-xs')
                                for n in pred["blue_tuo"]:
                                    ui.badge(f'{n:02d}', color='light-blue-4', text_color='white').props('size=sm')
                        
                        # 期望值
                        total_prob = pred['total_red_prob'] + pred['total_blue_prob']
                        ui.label(f'期望: {total_prob:.3f}').classes('text-caption text-primary q-mt-xs')

            # 概率排名 - 更紧凑
            with ui.card().classes('q-pa-sm'):
                ui.markdown('**🎯 概率排名**').classes('text-body2 q-mb-sm')
                
                with ui.grid(columns=2).classes('w-full q-gutter-sm'):
                    # 红球TOP10
                    with ui.column().classes('q-gutter-xs'):
                        ui.label('红球TOP10').classes('text-caption')
                        for i, (num, prob) in enumerate(select_hot(red_probs, 10), 1):
                            with ui.row().classes('items-center'):
                                ui.badge(f'{num:02d}', color='red-3' if i <= 3 else 'grey-4').props('size=sm')
                                ui.linear_progress(value=prob * 200, color='red').props('size=4px style="width: 60px"').classes('col')
                    
                    # 蓝球TOP5
                    with ui.column().classes('q-gutter-xs'):
                        ui.label('蓝球TOP5').classes('text-caption')
                        for i, (num, prob) in enumerate(select_hot(blue_probs, 5), 1):
                            with ui.row().classes('items-center'):
                                ui.badge(f'{num:02d}', color='blue' if i <= 2 else 'light-blue-3').props('size=sm')
                                ui.linear_progress(value=prob * 400, color='blue').props('size=4px style="width: 60px"').classes('col')
                                ui.label(f'{prob:.3f}').classes('text-caption')
            ui.markdown('预测结果仅供参考，请理性购彩！')

        page_state['tabs'].value = 'result'

    # ============== 预测函数 ==============
    async def do_prediction():
        global config
        if page_state['is_predicting']:
            ui.notify('预测进行中，请稍候...', type='warning')
            return

        page_state['is_predicting'] = True

        config['n_dan_red'] = int(n_dan_red_input.value or 4)
        config['n_tuo_hot'] = int(n_tuo_hot_input.value or 3)
        config['n_tuo_cold'] = int(n_tuo_cold_input.value or 2)
        config['n_predictions'] = int(n_predictions_input.value or 5)

        predict_btn.set_enabled(False)
        predict_status.text = '加载数据中...'

        try:
            predict_status.text = '加载数据库...'
            df = await run.io_bound(load_data)

            if len(df) < 10:
                predict_status.text = '数据量不足'
                ui.notify('数据库中数据不足，请先点击"爬取最新开奖数据"！', type='warning')
                return

            predict_status.text = '生成预测中...'
            predictions, red_probs, blue_probs, x_vals, pdf = await run.io_bound(
                lambda: generate_predictions(df, config['n_predictions'])
            )

            predict_status.text = '生成图表中...'
            await run.io_bound(
                lambda: plot_prediction_pools(predictions, red_probs, blue_probs, x_vals, pdf, OUTPUT_DIR)
            )

            predict_status.text = '更新界面...'
            update_prediction_result(predictions, df, red_probs, blue_probs)

            # 切换到图表面板
            page_state['tabs'].value = 'chart'
            page_state['chart_select'].set_value('综合分析图')
            show_chart('综合分析图')

            predict_status.text = '预测完成'
            ui.notify(f'成功生成 {len(predictions)} 组预测！', type='positive')

        except Exception as e:
            import traceback
            predict_status.text = f'错误: {str(e)[:20]}'
            ui.notify(f'预测失败: {e}', type='negative')
            print(traceback.format_exc())
        finally:
            predict_btn.set_enabled(True)
            page_state['is_predicting'] = False

    predict_btn.on_click(do_prediction)

    # ============== 初始化 ==============
    def init_page():
        do_check_db()
        do_load_history_data('all')

    ui.timer(0.5, init_page, once=True)
    
    # 延迟3秒后自动更新数据（不阻塞页面加载）
    async def delayed_auto_update():
        await asyncio.sleep(3)
        await do_auto_update()
    
    ui.timer(3.0, delayed_auto_update, once=True)


# ============== 自动更新数据 ==============
async def do_auto_update():
    """自动检查并更新数据"""
    try:
        # 获取数据库最新期号
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(CAST(period AS UNSIGNED)) FROM lottery_data")
        db_result = cursor.fetchone()
        db_latest = db_result[0] if db_result else 0
        cursor.close()
        conn.close()
        
        if not db_latest:
            return
        
        print(f"[自动更新] 数据库最新期号: {db_latest}")
        
        # 在后台线程中爬取数据
        data_list = await run.io_bound(_sync_crawl_latest)
        
        if not data_list:
            print(f"[自动更新] 未能获取在线数据")
            return
        
        # 获取网站最新期号
        website_latest = max(int(row['period'][-5:]) for row in data_list if row['period'])
        print(f"[自动更新] 网站最新期号: {website_latest}")
        
        if website_latest > db_latest:
            print(f"[自动更新] 发现新数据，正在更新...")
            saved = await run.io_bound(_sync_save_to_db, data_list)
            print(f"[自动更新] 更新完成，新增 {saved} 条")
            # 刷新页面数据
            await do_load_history_data('all')
        else:
            print(f"[自动更新] 数据已是最新")
            
    except Exception as e:
        print(f"[自动更新] 错误: {e}")


def _sync_crawl_latest():
    """同步爬取数据"""
    try:
        url = "https://datachart.500star.com/ssq/history/newinc/history.php"
        params = {'start': '26001', 'end': '27000'}
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://datachart.500star.com/',
        })
        
        response = session.get(url, params=params, timeout=30)
        response.encoding = 'gbk'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        tbody = soup.find('tbody', id='tdata')
        if not tbody:
            return None
        
        data_list = []
        rows = tbody.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 16:
                try:
                    period = cells[0].get_text(strip=True)
                    reds = [int(cells[i].get_text(strip=True)) for i in range(1, 7)]
                    blue = int(cells[7].get_text(strip=True))
                    draw_date = cells[15].get_text(strip=True)
                    
                    if all(1 <= r <= 33 for r in reds) and 1 <= blue <= 16:
                        data_list.append({
                            'period': period,
                            'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
                            'red4': reds[3], 'red5': reds[4], 'red6': reds[5],
                            'blue': blue,
                            'draw_date': draw_date
                        })
                except:
                    continue
        
        return data_list if data_list else None
    except Exception as e:
        print(f"爬取错误: {e}")
        return None


def _sync_save_to_db(data_list):
    """同步保存到数据库"""
    if not data_list:
        return 0
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    saved_count = 0
    for data in data_list:
        try:
            cursor.execute("SELECT id FROM lottery_data WHERE period = %s", (data['period'],))
            if cursor.fetchone():
                continue
            cursor.execute("""
                INSERT INTO lottery_data (period, red1, red2, red3, red4, red5, red6, blue, draw_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (data['period'], data['red1'], data['red2'], data['red3'],
                  data['red4'], data['red5'], data['red6'], data['blue'], data['draw_date']))
            saved_count += 1
        except:
            continue
    conn.commit()
    cursor.close()
    conn.close()
    return saved_count


# ============== 启动 ==============
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host='0.0.0.0', port=8082, title='双色球预测系统', dark=False, reload=False)
