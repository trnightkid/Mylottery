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

OUTPUT_DIR = r"D:\Mydevelopment\MultiContentProject\Mylottery\dan_tuo_prediction"
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
        for r in rows:
            reds = f"{r[1]:02d} {r[2]:02d} {r[3]:02d} {r[4]:02d} {r[5]:02d} {r[6]:02d}"
            table_rows.append({
                'period': str(r[0]),
                'reds': reds,
                'blue': f"{r[7]:02d}",
                'date': str(r[8])
            })

        if page_state['history_table'] is not None:
            page_state['history_table'].rows = table_rows
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

        ui.separator()

        ui.label('数据爬取').classes('text-subtitle2 q-mt-md')

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

        ui.button('爬取最新数据', on_click=do_crawl, color='primary').classes('full-width q-mt-sm')
        ui.label('获取最新5期开奖数据').classes('text-caption text-grey')

        ui.separator()

        ui.label('预测参数').classes('text-subtitle2 q-mt-md')

        with ui.column().classes('q-gutter-y-sm q-mt-sm'):
            ui.label('红球胆码数:').classes('text-caption')
            n_dan_red_input = ui.number(value=config['n_dan_red'], min=1, max=6, step=1).props('dense outlined')

            ui.label('红球热拖数:').classes('text-caption')
            n_tuo_hot_input = ui.number(value=config['n_tuo_hot'], min=1, max=6, step=1).props('dense outlined')

            ui.label('红球冷拖数:').classes('text-caption')
            n_tuo_cold_input = ui.number(value=config['n_tuo_cold'], min=0, max=6, step=1).props('dense outlined')

            ui.label('预测组数:').classes('text-caption')
            n_predictions_input = ui.number(value=config['n_predictions'], min=1, max=10, step=1).props(
                'dense outlined')

        ui.separator()

        ui.label('开始预测').classes('text-subtitle2 q-mt-md')
        predict_btn = ui.button('开始预测', color='positive').classes('full-width q-mt-sm')
        predict_status = ui.label('就绪').classes('text-caption text-center q-mt-sm')

    # ========== 主内容区域 ==========
    with ui.column().classes('q-pa-md full-width'):
        ui.label('双色球胆拖投注预测系统').classes('text-h4 text-weight-bold q-mb-md')

        with ui.tabs().classes('w-full') as page_state['tabs']:
            ui.tab('预测结果').props('name=result')
            ui.tab('图表分析').props('name=chart')
            ui.tab('历史数据').props('name=history')

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
                    ui.markdown('## 图表分析')

                    with ui.row().classes('w-full items-center'):
                        ui.label('选择图表:').classes('text-subtitle2')
                        page_state['chart_select'] = ui.select(
                            ['综合分析图', '拟合曲线图', '红球热力图', '蓝球雷达图'],
                            value='综合分析图'
                        ).classes('col-grow')

                        ui.button('打开文件夹', on_click=lambda: os.startfile(OUTPUT_DIR),
                                  color='secondary').props('outline')

                    page_state['chart_container'] = ui.column().classes('w-full items-center q-mt-md')
                    with page_state['chart_container']:
                        ui.icon('insert_chart', size='100px').classes('text-grey-5 q-mt-lg')
                        ui.label('请先运行预测生成图表').classes('text-h6 text-grey-6 q-mt-md')

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

                    page_state['history_table'] = ui.table(
                        columns=[
                            {'name': 'period', 'label': '期号', 'field': 'period', 'align': 'center', 'sortable': True},
                            {'name': 'reds', 'label': '红球', 'field': 'reds', 'align': 'center'},
                            {'name': 'blue', 'label': '蓝球', 'field': 'blue', 'align': 'center'},
                            {'name': 'date', 'label': '开奖日期', 'field': 'date', 'align': 'center', 'sortable': True},
                        ],
                        rows=[],
                        row_key='period',
                        pagination={'rowsPerPage': 20, 'rowsPerPageOptions': [10, 20, 50, 100, 200, 500, 1000]}
                    ).classes('w-full').props('dense')

    # ============== 图表显示函数 ==============
    def show_chart(chart_name):
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

    page_state['chart_select'].on_value_change(lambda value: show_chart(value))

    # ============== 更新预测结果函数 ==============
    def update_prediction_result(predictions, df, red_probs, blue_probs):
        page_state['result_container'].clear()

        with page_state['result_container']:
            ui.markdown(f'### 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            ui.markdown(f'数据来源: 数据库 ({len(df)} 条历史记录)')
            ui.markdown(f'预测组数: {len(predictions)} 组')
            ui.markdown('---')

            ui.markdown('## 预测号码')

            # 按综合期望值排序
            sorted_predictions = sorted(predictions,
                                        key=lambda p: p['total_red_prob'] + p['total_blue_prob'],
                                        reverse=True)

            for pred in sorted_predictions:
                with ui.card().classes('q-mb-md'):
                    with ui.column().classes('q-pa-md'):
                        rank = sorted_predictions.index(pred) + 1
                        ui.markdown(f'#### 预测 {pred["group"]} (排名 #{rank})')

                        # 红球胆码
                        with ui.row().classes('items-center q-mt-sm'):
                            ui.label('红球胆码: ').classes('text-weight-medium')
                            for n in pred["red_dan"]:
                                ui.badge(f'{n:02d}', color='red-5', text_color='white').classes('q-mr-xs')

                        # 红球拖码
                        with ui.row().classes('items-center q-mt-sm'):
                            ui.label('红球拖码: ').classes('text-weight-medium')
                            for n in pred["red_tuo"]:
                                ui.badge(f'{n:02d}', color='orange-3', text_color='white').classes('q-mr-xs')

                        # 蓝球胆码
                        with ui.row().classes('items-center q-mt-sm'):
                            ui.label('蓝球胆码: ').classes('text-weight-medium')
                            for n in pred["blue_dan"]:
                                ui.badge(f'{n:02d}', color='blue', text_color='white').classes('q-mr-xs')

                        # 蓝球拖码
                        with ui.row().classes('items-center q-mt-sm'):
                            ui.label('蓝球拖码: ').classes('text-weight-medium')
                            for n in pred["blue_tuo"]:
                                ui.badge(f'{n:02d}', color='light-blue', text_color='white').classes('q-mr-xs')

                        # 期望命中 - 使用 ui.row() 和 ui.label() 替代 HTML
                        total_prob = pred['total_red_prob'] + pred['total_blue_prob']
                        with ui.row().classes('items-center q-mt-md'):
                            ui.label(f'综合期望: ').classes('text-weight-bold')
                            ui.label(f'{total_prob:.4f} ').classes('text-h6 text-primary')
                            ui.label(
                                f'(红球 {pred["total_red_prob"]:.4f} + 蓝球 {pred["total_blue_prob"]:.4f})').classes(
                                'text-caption')

            ui.markdown('---')

            ui.markdown('## 概率排名')

            with ui.row().classes('w-full q-gutter-lg'):
                with ui.column().classes('col'):
                    ui.markdown('### 红球 TOP10')
                    for i, (num, prob) in enumerate(select_hot(red_probs, 10), 1):
                        with ui.row().classes('items-center'):
                            ui.badge(f'{i}', color='grey').classes('q-mr-sm')
                            ui.badge(f'{num:02d}', color='red-3').classes('q-mr-sm')
                            ui.progress(value=prob * 200, show_value=False, color='red').props(
                                'style="width: 120px"').classes('q-mr-sm')
                            ui.label(f'{prob:.4f}').classes('text-caption')

                with ui.column().classes('col'):
                    ui.markdown('### 蓝球 TOP5')
                    for i, (num, prob) in enumerate(select_hot(blue_probs, 5), 1):
                        with ui.row().classes('items-center'):
                            ui.badge(f'{i}', color='grey').classes('q-mr-sm')
                            ui.badge(f'{num:02d}', color='blue').classes('q-mr-sm')
                            ui.progress(value=prob * 400, show_value=False, color='blue').props(
                                'style="width: 120px"').classes('q-mr-sm')
                            ui.label(f'{prob:.4f}').classes('text-caption')

            ui.markdown('---')
            ui.markdown('### 风险提示')
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


# ============== 启动 ==============
ui.run(host='0.0.0.0', port=8080, title='双色球预测系统', dark=True)
