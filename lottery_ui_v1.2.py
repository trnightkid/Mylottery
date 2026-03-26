"""
双色球预测分析 - 调优修复版
"""
import pymysql
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from datetime import datetime
import warnings
import lottery_dantuo_prediction_v2 as pred_module
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import requests
import threading
import csv
import os
import sys
import traceback
import inspect
import time
import random

warnings.filterwarnings('ignore')

# ============== 配置 ==============
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'reven@0504',
    'database': 'lottery_db',
    'charset': 'utf8mb4'
}

TABLE_NAME = "lottery_data"
OUTPUT_DIR = r"D:\Mydevelopment\MultiContentProject\Mylottery\dan_tuo_prediction"
CSV_FILE = r"D:\Mydevelopment\MultiContentProject\Mylottery\lottery_data_from_web.csv"

DEFAULT_N_DAN_RED = 4
DEFAULT_N_TUO_HOT_RED = 3
DEFAULT_N_TUO_COLD_RED = 2
DEFAULT_N_DAN_BLUE = 1
DEFAULT_N_TUO_BLUE = 2
DEFAULT_N_PREDICTIONS = 5

# 尝试导入预测模块
PREDICTION_MODULE_LOADED = False
PREDICTION_MODULE_ERROR = None
GENERATE_SIGNATURE = None


def try_import_prediction_module():
    global PREDICTION_MODULE_LOADED, PREDICTION_MODULE_ERROR, GENERATE_SIGNATURE

    script_dir = os.path.dirname(os.path.abspath(__file__))
    py_file = os.path.join(script_dir, "lottery_dantuo_prediction_v2.py")

    if not os.path.exists(py_file):
        PREDICTION_MODULE_ERROR = f"文件不存在: {py_file}"
        return False

    module_name = "lottery_dantuo_prediction"
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        import py_compile
        py_compile.compile(py_file, doraise=True)

        spec = __import__(module_name, fromlist=['*'])
        sys.modules[module_name] = spec

        required_funcs = ['generate_predictions', 'load_data', 'plot_prediction_pools', 'fit_distributions',
                          'calculate_blue_probs']
        missing = [f for f in required_funcs if not hasattr(spec, f)]

        if missing:
            PREDICTION_MODULE_ERROR = f"缺少函数: {', '.join(missing)}"
            return False

        try:
            GENERATE_SIGNATURE = inspect.signature(spec.generate_predictions)
            print("=" * 60)
            print("✅ 预测模块加载成功!")
            print(f"   generate_predictions 签名: {GENERATE_SIGNATURE}")
            print("=" * 60)
        except Exception as e:
            print(f"✅ 预测模块加载成功! (签名获取失败: {e})")

        PREDICTION_MODULE_LOADED = True
        return True

    except Exception as e:
        PREDICTION_MODULE_ERROR = f"导入错误: {e}"
        return False


try_import_prediction_module()


# ============== 数据库函数 ==============
def get_db_connection():
    return pymysql.connect(**DB_CONFIG, autocommit=True)


def check_db_status():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = cursor.fetchone()[0]
        cursor.execute(f"SELECT MAX(CAST(period AS UNSIGNED)) FROM {TABLE_NAME}")
        latest = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return True, count, latest
    except Exception as e:
        return False, 0, str(e)


def create_table_if_not_exists():
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        period VARCHAR(10) NOT NULL UNIQUE,
        red1 TINYINT UNSIGNED NOT NULL, red2 TINYINT UNSIGNED NOT NULL,
        red3 TINYINT UNSIGNED NOT NULL, red4 TINYINT UNSIGNED NOT NULL,
        red5 TINYINT UNSIGNED NOT NULL, red6 TINYINT UNSIGNED NOT NULL,
        blue TINYINT UNSIGNED NOT NULL, draw_date DATE,
        INDEX idx_period (period)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    cursor.execute(sql)
    cursor.close()
    conn.close()


# ============== 数据爬取 ==============
def crawl_latest_data():
    """从网络爬取双色球最新数据"""
    import json
    
    print("[爬取] 开始尝试获取在线数据...")
    
    # 尝试500彩票网API
    try:
        print("[爬取] 尝试 500彩票网...")
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://datachart.500star.com/',
        })
        
        url = "https://datachart.500star.com/ssq/history/newinc/history.php"
        params = {'start': '26001', 'end': '26020'}
        
        response = session.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            # 检测返回的是HTML还是JSON
            response_text = response.text.strip()
            
            # 如果返回HTML（div标签），说明API改版
            if response_text.startswith('<div') or response_text.startswith('<!DOCTYPE'):
                print("[爬取] 警告: API返回HTML而非JSON")
                print("[爬取] 原因: 500彩票网已改版，数据需要JavaScript渲染")
                print()
                print("解决方案:")
                print("   1. 手动访问 https://datachart.500star.com/ssq/history/history.shtml")
                print("   2. 复制最新开奖数据到 lottery_data_from_web.csv")
                print("   3. 使用'同步CSV数据'功能导入")
                return None
            
            # 尝试解析JSON
            try:
                data = response.json()
                items = data if isinstance(data, list) else data.get('list', data.get('data', []))
                
                results = []
                for item in items:
                    try:
                        period = str(item.get('period', ''))[-5:].zfill(5)
                        reds = [int(x) for x in item.get('red', '').split(',')]
                        blue = int(item.get('blue', 0))
                        if len(reds) == 6 and 1 <= blue <= 16:
                            results.append({
                                'period': period,
                                'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
                                'red4': reds[3], 'red5': reds[4], 'red6': reds[5],
                                'blue': blue,
                                'draw_date': item.get('date', datetime.now().strftime('%Y-%m-%d'))
                            })
                    except:
                        continue
                
                if results:
                    print(f"[爬取] 成功获取 {len(results)} 条真实数据")
                    return results
                else:
                    print("[爬取] 解析后无有效数据")
                    return None
                    
            except json.JSONDecodeError:
                print("[爬取] JSON解析失败，返回内容不是有效JSON")
                return None
        else:
            print(f"[爬取] HTTP错误: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("[爬取] 请求超时")
    except requests.exceptions.RequestException as e:
        print(f"[爬取] 网络错误: {e}")
    except Exception as e:
        print(f"[爬取] 错误: {e}")
    
    print("[爬取] 无法从在线API获取数据")
    return None


def save_to_db(data_list):
    if not data_list:
        return 0
    create_table_if_not_exists()
    conn = get_db_connection()
    cursor = conn.cursor()
    saved = 0
    for data in data_list:
        try:
            sql = f"INSERT INTO {TABLE_NAME} VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE blue=VALUES(blue)"
            cursor.execute(sql, (data['period'], data['red1'], data['red2'], data['red3'], data['red4'], data['red5'],
                                 data['red6'], data['blue'], data['draw_date']))
            saved += 1
        except:
            continue
    cursor.close()
    conn.close()
    return saved


# ============== CSV同步 ==============
def sync_from_csv():
    if not os.path.exists(CSV_FILE):
        return -1, "文件不存在"

    data_list = []
    with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                period = row.get('period', '').strip()[-5:].zfill(5)
                reds = [int(row.get(f'red{i}', 0)) for i in range(1, 7)]
                blue = int(row.get('blue', 0))
                if len(reds) == 6 and all(1 <= r <= 33 for r in reds) and 1 <= blue <= 16:
                    data_list.append({
                        'period': period,
                        'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
                        'red4': reds[3], 'red5': reds[4], 'red6': reds[5],
                        'blue': blue,
                        'draw_date': row.get('draw_date', datetime.now().strftime('%Y-%m-%d'))
                    })
            except:
                continue

    if not data_list:
        return 0, "无有效数据"

    create_table_if_not_exists()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(CAST(period AS UNSIGNED)) FROM {TABLE_NAME}")
    db_latest = cursor.fetchone()[0] or 0
    new_data = [d for d in data_list if int(d['period']) > db_latest]
    inserted = 0
    for row in new_data:
        try:
            sql = f"INSERT INTO {TABLE_NAME} VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE draw_date=VALUES(draw_date)"
            cursor.execute(sql,
                           (row['period'], row['red1'], row['red2'], row['red3'], row['red4'], row['red5'], row['red6'],
                            row['blue'], row['draw_date']))
            inserted += 1
        except:
            continue
    cursor.close()
    conn.close()
    return inserted, f"新增{inserted}条"


# ============== GUI类 ==============
class LotteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎱 双色球胆拖预测系统 - 调优版")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        # 状态变量
        self.prediction_counter = 0
        self.last_prediction_result = None
        self.is_predicting = False  # 防止重复点击
        self.chart_label = None

        # 样式配置
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Prediction.TButton', font=('Microsoft YaHei', 11, 'bold'), foreground='blue')

        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ===== 左侧面板 =====
        ttk.Label(left, text="🎱 双色球预测", font=('Microsoft YaHei', 16, 'bold')).pack(pady=(0, 15))

        # 模块状态
        module_frame = ttk.LabelFrame(left, text="📦 模块状态", padding=10)
        module_frame.pack(fill=tk.X, pady=(0, 10))

        if PREDICTION_MODULE_LOADED:
            ttk.Label(module_frame, text="✅ 预测模块已加载", foreground="green").pack(anchor=tk.W)
            sig_str = str(GENERATE_SIGNATURE) if GENERATE_SIGNATURE else "无法获取"
            ttk.Label(module_frame, text="函数签名: " + sig_str, foreground="blue", font=('Arial', 8)).pack(anchor=tk.W)
        else:
            error_text = (PREDICTION_MODULE_ERROR or "未知错误")[:50]
            ttk.Label(module_frame, text="❌ 预测模块未加载", foreground="red").pack(anchor=tk.W)
            ttk.Label(module_frame, text="原因: " + error_text + "...", foreground="gray", font=('Arial', 8)).pack(
                anchor=tk.W)

        # 数据库状态
        db_frame = ttk.LabelFrame(left, text="📊 数据库状态", padding=10)
        db_frame.pack(fill=tk.X, pady=(0, 10))

        self.db_status = ttk.Label(db_frame, text="检查中...", foreground="blue")
        self.db_status.pack(anchor=tk.W)

        self.data_count = ttk.Label(db_frame, text="", foreground="gray")
        self.data_count.pack(anchor=tk.W)

        # 数据更新
        crawl_frame = ttk.LabelFrame(left, text="🌐 数据更新", padding=10)
        crawl_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(crawl_frame, text="📂 同步CSV数据", command=self.sync_csv).pack(fill=tk.X, pady=2)
        ttk.Button(crawl_frame, text="🌐 爬取最新数据", command=self.crawl_data).pack(fill=tk.X, pady=2)

        self.crawl_status = ttk.Label(crawl_frame, text="就绪", foreground="gray")
        self.crawl_status.pack(anchor=tk.W, pady=(5, 0))

        # 参数设置
        param_frame = ttk.LabelFrame(left, text="⚙️ 预测参数", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(param_frame, text="红球胆码:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.n_dan_red = ttk.Spinbox(param_frame, from_=1, to=6, width=8)
        self.n_dan_red.set(DEFAULT_N_DAN_RED)
        self.n_dan_red.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="红球拖码(热):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.n_tuo_hot_red = ttk.Spinbox(param_frame, from_=0, to=10, width=8)
        self.n_tuo_hot_red.set(DEFAULT_N_TUO_HOT_RED)
        self.n_tuo_hot_red.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="红球拖码(冷):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.n_tuo_cold_red = ttk.Spinbox(param_frame, from_=0, to=10, width=8)
        self.n_tuo_cold_red.set(DEFAULT_N_TUO_COLD_RED)
        self.n_tuo_cold_red.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="蓝球胆码:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.n_dan_blue = ttk.Spinbox(param_frame, from_=1, to=5, width=8)
        self.n_dan_blue.set(DEFAULT_N_DAN_BLUE)
        self.n_dan_blue.grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="蓝球拖码:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.n_tuo_blue = ttk.Spinbox(param_frame, from_=1, to=10, width=8)
        self.n_tuo_blue.set(DEFAULT_N_TUO_BLUE)
        self.n_tuo_blue.grid(row=4, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="预测组数:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.n_pred = ttk.Spinbox(param_frame, from_=1, to=10, width=8)
        self.n_pred.set(DEFAULT_N_PREDICTIONS)
        self.n_pred.grid(row=5, column=1, padx=5, pady=2)

        # 当前参数显示
        self.param_display = tk.StringVar(value="")
        self.lbl_current_params = ttk.Label(param_frame, textvariable=self.param_display,
                                            foreground="blue", font=('Arial', 9))
        self.lbl_current_params.grid(row=6, column=0, columnspan=3, pady=(10, 0))

        for spin in [self.n_dan_red, self.n_tuo_hot_red, self.n_tuo_cold_red,
                     self.n_dan_blue, self.n_tuo_blue, self.n_pred]:
            spin.bind('<<Increment>>', self.update_param_display)
            spin.bind('<<Decrement>>', self.update_param_display)

        # 预测按钮
        self.btn_predict = ttk.Button(left, text="🎯 开始预测", command=self.run_prediction, style='Prediction.TButton')
        self.btn_predict.pack(fill=tk.X, pady=5)

        self.predict_status = ttk.Label(left, text="就绪", foreground="gray")
        self.predict_status.pack(anchor=tk.W)

        ttk.Button(left, text="🔄 刷新状态", command=self.refresh_db).pack(fill=tk.X, pady=5)

        # ===== 右侧选项卡 =====
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="📋 预测结果")

        # 预测结果显示区域
        result_container = ttk.Frame(self.tab1)
        result_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_text = ScrolledText(result_container, wrap=tk.WORD, font=('Consolas', 11))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.show_welcome()

        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="📈 图表分析")

        chart_sel = ttk.Frame(self.tab2)
        chart_sel.pack(fill=tk.X, padx=10, pady=5)

        self.chart_var = tk.StringVar(value='dan_tuo_pools')
        for text, val in [('综合分析', 'dan_tuo_pools'), ('拟合曲线', 'fitting_curve')]:
            ttk.Radiobutton(chart_sel, text=text, variable=self.chart_var, value=val,
                            command=self.refresh_chart).pack(side=tk.LEFT, padx=5)

        ttk.Button(chart_sel, text="🔄 刷新图表", command=self.refresh_chart).pack(side=tk.LEFT, padx=10)
        ttk.Button(chart_sel, text="📂 打开文件夹", command=self.open_folder).pack(side=tk.RIGHT)

        self.chart_frame = ttk.Frame(self.tab2)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._show_chart_placeholder("请先运行预测生成图表")

        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="📜 历史数据")

        search_f = ttk.Frame(self.tab3)
        search_f.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(search_f, text="搜索:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_f, width=15)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_f, text="搜索", command=self.search_history).pack(side=tk.LEFT)
        ttk.Button(search_f, text="全部", command=self.load_history).pack(side=tk.LEFT, padx=5)

        cols = ('period', 'reds', 'blue', 'date')
        self.tree = ttk.Treeview(self.tab3, columns=cols, show='headings', height=20)
        for col in cols:
            self.tree.heading(col, text={'period': '期号', 'reds': '红球', 'blue': '蓝球', 'date': '日期'}[col])
            self.tree.column(col, width={'period': 90, 'reds': 180, 'blue': 60, 'date': 100}[col], anchor='center')

        scroll = ttk.Scrollbar(self.tab3, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # 初始化
        self.update_param_display()
        self.refresh_db()
        self.load_history()

    def _handle_prediction_complete(self, result, predict_id):
        """在主线程中处理UI更新和绘图"""
        self.is_predicting = False
        self.btn_predict.config(state=tk.NORMAL)

        if result['success']:
            print(f"[GUI] 预测成功，正在绘图...")
            self.predict_status.config(text=f"✅ 完成 (#{predict_id})", foreground="green")

            # --- 在主线程中绘图 (安全) ---
            try:
                # 这里调用绘图函数，传入刚才计算好的数据
                # 注意：plot_prediction_pools 必须在主线程调用
                pred_module.plot_prediction_pools(
                    result['pred'],
                    result['red_p'],
                    result['blue_p'],
                    result['x'],
                    result['pdf'],
                    OUTPUT_DIR
                )
                print(f"[GUI] 绘图完成")
            except Exception as e:
                print(f"[GUI] 绘图出错: {e}")

            # 更新文本显示
            result_str = self._format_prediction_result_debug(result['pred'], result['params'], predict_id)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_str)

            # 刷新界面
            self.notebook.select(self.tab1)
            self.refresh_chart()
        else:
            print(f"[GUI] 预测失败")
            self.predict_status.config(text="❌ 失败", foreground="red")
            error_msg = f"预测任务 (# {predict_id}) 失败\n\n错误: {result['error']}\n\n详情:\n{result['traceback_str']}"
            messagebox.showerror("预测出错", error_msg)

    def _show_chart_placeholder(self, message, foreground="gray"):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        ttk.Label(self.chart_frame, text=message, foreground=foreground).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def get_params(self):
        try:
            dan_red = int(self.n_dan_red.get())
            tuo_hot_red = int(self.n_tuo_hot_red.get())
            tuo_cold_red = int(self.n_tuo_cold_red.get())
            dan_blue = int(self.n_dan_blue.get())
            tuo_blue = int(self.n_tuo_blue.get())
            n_pred = int(self.n_pred.get())

            params = {
                'dan_red': dan_red,
                'tuo_hot_red': tuo_hot_red,
                'tuo_cold_red': tuo_cold_red,
                'dan_blue': dan_blue,
                'tuo_blue': tuo_blue,
                'n_predictions': n_pred
            }
            return params
        except Exception as e:
            print(f"参数获取错误: {e}")
            return {
                'dan_red': DEFAULT_N_DAN_RED,
                'tuo_hot_red': DEFAULT_N_TUO_HOT_RED,
                'tuo_cold_red': DEFAULT_N_TUO_COLD_RED,
                'dan_blue': DEFAULT_N_DAN_BLUE,
                'tuo_blue': DEFAULT_N_TUO_BLUE,
                'n_predictions': DEFAULT_N_PREDICTIONS
            }

    def update_param_display(self, event=None):
        p = self.get_params()
        text = "当前设置:\n"
        text += "红: " + str(p['dan_red']) + "胆 + " + str(p['tuo_hot_red']) + "热 + " + str(p['tuo_cold_red']) + "冷\n"
        text += "蓝: " + str(p['dan_blue']) + "胆 + " + str(p['tuo_blue']) + "拖"
        self.param_display.set(text)

    def show_welcome(self):
        self.result_text.insert(tk.END, "=" * 60 + "\n")
        self.result_text.insert(tk.END, "🎱 双色球胆拖投注预测系统\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n\n")
        self.result_text.insert(tk.END, "调试模式：每次预测都会在控制台输出详细信息\n\n")
        if not PREDICTION_MODULE_LOADED:
            error_msg = PREDICTION_MODULE_ERROR or "未知错误"
            self.result_text.insert(tk.END, f"⚠️ 预测模块未加载!\n错误: {error_msg}\n\n")
        self.result_text.insert(tk.END, "使用说明：\n")
        self.result_text.insert(tk.END, "1. 调整左侧「预测参数」\n")
        self.result_text.insert(tk.END, "2. 点击「开始预测」生成预测结果\n")
        self.result_text.insert(tk.END, "3. 查看控制台输出，了解参数传递情况\n")
        self.result_text.insert(tk.END, "\n⚠️ 预测结果仅供参考，请理性购彩！\n")

    def refresh_db(self):
        ok, count, latest = check_db_status()
        if ok:
            self.db_status.config(text="已连接", foreground="green")
            self.data_count.config(text="数据量: " + str(count) + " 条 | 最新期: " + str(latest))
        else:
            self.db_status.config(text="❌ " + latest, foreground="red")
        
        # 自动检查并更新数据
        self.auto_check_update()

    def auto_check_update(self):
        """自动检查并更新数据"""
        def task():
            try:
                # 获取数据库最新期号
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(f"SELECT MAX(CAST(period AS UNSIGNED)) FROM {TABLE_NAME}")
                db_latest = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                if not db_latest:
                    return
                
                # 爬取网站最新期号
                print(f"[自动更新] 数据库最新期号: {db_latest}")
                data = crawl_latest_data()
                
                if not data:
                    # 尝试使用备用方法
                    print(f"[自动更新] 尝试备用爬取方法...")
                    data = self._backup_crawl()
                
                if data:
                    website_latest = max(int(row['period']) for row in data)
                    print(f"[自动更新] 网站最新期号: {website_latest}")
                    
                    if website_latest > db_latest:
                        print(f"[自动更新] 发现新数据: {website_latest} > {db_latest}，正在更新...")
                        new_count = save_to_db(data)
                        print(f"[自动更新] 更新完成，新增 {new_count} 条")
                        self.root.after(0, self.refresh_db)
                        self.root.after(0, self.load_history)
                    else:
                        print(f"[自动更新] 数据已是最新")
                else:
                    print(f"[自动更新] 未能获取在线数据")
                    
            except Exception as e:
                print(f"[自动更新] 错误: {e}")
        
        threading.Thread(target=task, daemon=True).start()

    def _backup_crawl(self):
        """备用爬取方法 - 使用完整的爬虫逻辑"""
        try:
            import Lottery_spider as spider
            session = spider.get_session()
            spider.visit_homepage(session)
            
            # 获取最新期号
            latest_period = spider.get_latest_period(session)
            if not latest_period:
                return None
            
            # 获取数据
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"SELECT MAX(CAST(period AS UNSIGNED)) FROM {TABLE_NAME}")
            db_latest = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            start_period = str(db_latest + 1) if db_latest else "26001"
            end_period = latest_period
            
            html = spider.fetch_data(session, start_period, end_period)
            if html:
                data = spider.parse_html(html)
                return data
        except Exception as e:
            print(f"[备用爬取] 错误: {e}")
        return None

    def sync_csv(self):
        self.crawl_status.config(text="同步中...", foreground="blue")
        self.root.update()

        def task():
            result, msg = sync_from_csv()
            self.root.after(0, lambda: self.crawl_status.config(
                text=msg if result > 0 else "❌ " + msg,
                foreground="green" if result > 0 else "red"))
            if result > 0:
                self.root.after(0, self.refresh_db)
                self.root.after(0, self.load_history)
                self.root.after(0, lambda: messagebox.showinfo("完成", msg))

        threading.Thread(target=task, daemon=True).start()

    def crawl_data(self):
        self.crawl_status.config(text="爬取中...", foreground="blue")
        self.root.update()

        def task():
            data = crawl_latest_data()
            if data:
                new = save_to_db(data)
                self.root.after(0, lambda: self.crawl_status.config(
                    text="✅ 获取" + str(len(data)) + "期，保存" + str(new) + "条", foreground="green"))
                self.root.after(0, self.refresh_db)
                self.root.after(0, self.load_history)
            else:
                self.root.after(0, lambda: self.crawl_status.config(text="❌ 爬取失败", foreground="red"))

        threading.Thread(target=task, daemon=True).start()

    def run_prediction(self):
        if self.is_predicting:
            messagebox.showwarning("提示", "正在预测中，请稍候...")
            return

        if not PREDICTION_MODULE_LOADED:
            messagebox.showerror("错误", f"预测模块未加载!\n\n原因: {PREDICTION_MODULE_ERROR}")
            return

        params = self.get_params()

        self.prediction_counter += 1
        current_predict_id = self.prediction_counter
        self.is_predicting = True
        self.btn_predict.config(state=tk.DISABLED)
        self.predict_status.config(text="预测中...", foreground="blue")

        print(f"\n[GUI] 开始第 {current_predict_id} 次预测任务...")
        print(f"[GUI] 参数: {params}")

        def task():
            thread_result = {
                'success': False,
                'pred': None,
                'red_p': None,
                'blue_p': None,
                'x': None,
                'pdf': None,
                'error': None,
                'traceback_str': None
            }

            try:
                print(f"[Thread] 正在加载数据...")

                df = pred_module.load_data()
                if df is None or len(df) < 10:
                    raise ValueError(f"数据加载失败或数据不足，当前记录数: {len(df) if df is not None else 0}")

                print(f"[Thread] 数据加载成功，共 {len(df)} 条记录")

                # 设置全局参数
                config_vars = {
                    'N_DAN_RED': params['dan_red'],
                    'N_TUO_HOT_RED': params['tuo_hot_red'],
                    'N_TUO_COLD_RED': params['tuo_cold_red'],
                    'N_DAN_BLUE': params['dan_blue'],
                    'N_TUO_BLUE': params['tuo_blue'],
                    'N_PREDICTIONS': params['n_predictions']
                }

                for var_name, value in config_vars.items():
                    if hasattr(pred_module, var_name):
                        setattr(pred_module, var_name, value)
                        print(f"[Thread] 设置 {var_name} = {value}")

                print(f"[Thread] 正在调用预测函数...")
                print(f"[Thread] 参数: n={params['n_predictions']}")

                result = pred_module.generate_predictions(
                    df, 
                    n=params['n_predictions'],
                    n_dan_blue=params['dan_blue'],
                    n_tuo_blue=params['tuo_blue']
                )

                if result is None or len(result) < 1:
                    raise ValueError("预测函数返回空结果")

                pred, red_p, blue_p, x, pdf = result

                print(f"[Thread] 预测成功，生成 {len(pred)} 组预测")

                thread_result['success'] = True
                thread_result['pred'] = pred
                thread_result['red_p'] = red_p
                thread_result['blue_p'] = blue_p
                thread_result['x'] = x
                thread_result['pdf'] = pdf
                thread_result['params'] = params

            except Exception as e:
                import traceback
                thread_result['success'] = False
                thread_result['error'] = str(e)
                thread_result['traceback_str'] = traceback.format_exc()
                print(f"[Thread] ❌ 错误: {e}")
                print(f"[Thread] 详细错误:\n{traceback.format_exc()}")

            self.root.after(0, lambda r=thread_result, pid=current_predict_id: self._handle_prediction_complete(r, pid))

        threading.Thread(target=task, daemon=True).start()

    def _reset_prediction_state(self):
        self.is_predicting = False
        self.btn_predict.config(state=tk.NORMAL)

    def _format_prediction_result(self, pred, params, predict_id):
        lines = []
        lines.append("=" * 70)
        lines.append(f"🎱 预测结果 [第 {predict_id} 次]")
        lines.append("⏰ " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        lines.append("=" * 70)
        lines.append("")

        sample_red_dan = len(pred[0]['red_dan']) if pred else 0
        sample_red_tuo = len(pred[0]['red_tuo']) if pred else 0
        sample_blue_dan = len(pred[0]['blue_dan']) if pred else 0
        sample_blue_tuo = len(pred[0]['blue_tuo']) if pred else 0

        lines.append("📊 【参数对比 - 输入 vs 实际】")
        lines.append(f"   红球胆码: 输入={params['dan_red']}, 实际={sample_red_dan}")
        lines.append(f"   红球拖码: 输入={params['tuo_hot_red'] + params['tuo_cold_red']}, 实际={sample_red_tuo}")
        lines.append(f"   蓝球胆码: 输入={params['dan_blue']}, 实际={sample_blue_dan}")
        lines.append(f"   蓝球拖码: 输入={params['tuo_blue']}, 实际={sample_blue_tuo}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("【预测号码详情】")
        lines.append("-" * 70)

        for idx, p in enumerate(pred, 1):
            lines.append(f"【预测 {idx}】")
            lines.append(f"   红球胆码: {sorted([f'{n:02d}' for n in p['red_dan']])} ({len(p['red_dan'])}个)")
            lines.append(f"   红球拖码: {sorted([f'{n:02d}' for n in p['red_tuo']])} ({len(p['red_tuo'])}个)")
            lines.append(f"   蓝球胆码: {sorted([f'{n:02d}' for n in p['blue_dan']])} ({len(p['blue_dan'])}个)")
            lines.append(f"   蓝球拖码: {sorted([f'{n:02d}' for n in p['blue_tuo']])} ({len(p['blue_tuo'])}个)")
            lines.append("")

        lines.append("-" * 70)
        lines.append(f"图表已保存至: {OUTPUT_DIR}")

        return "\n".join(lines)

    def _update_prediction_ui(self, result_str, predict_id):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_str)
        self.predict_status.config(text=f"预测完成 (#{predict_id})", foreground="green")

    def _format_prediction_result_debug(self, pred, input_params, predict_id):
        lines = []
        lines.append("=" * 70)
        lines.append(f"🎱 预测结果 [第 {predict_id} 次] - 参数验证模式")
        lines.append("⏰ " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        lines.append("=" * 70)

        if pred:
            first = pred[0]
            # 提取实际数量
            r_dan = len(first.get('red_dan', []))
            r_tuo = len(first.get('red_tuo', []))
            b_dan = len(first.get('blue_dan', []))
            b_tuo = len(first.get('blue_tuo', []))

            lines.append("")
            lines.append("🔍 【参数对比（请求 vs 实际）】")
            lines.append(
                f"   🔴 红胆: {input_params['dan_red']} ➡ {r_dan}  {'✅' if r_dan == input_params['dan_red'] else '❌'}")
            lines.append(
                f"   🟡 红拖: {input_params['tuo_hot_red'] + input_params['tuo_cold_red']} ➡ {r_tuo}  {'✅' if r_tuo == (input_params['tuo_hot_red'] + input_params['tuo_cold_red']) else '❌'}")
            lines.append(
                f"   🔵 蓝胆: {input_params['dan_blue']} ➡ {b_dan}  {'✅' if b_dan == input_params['dan_blue'] else '❌'}")
            lines.append(
                f"   🟦 蓝拖: {input_params['tuo_blue']} ➡ {b_tuo}  {'✅' if b_tuo == input_params['tuo_blue'] else '❌'}")
            lines.append("")

            if r_dan != input_params['dan_blue']:
                lines.append("⚠️ 注意：如果出现 ❌，说明预测模块没有响应您的参数修改。")
                lines.append("⚠️ 请检查控制台输出，查看是否成功修改了模块内部变量。")
                lines.append("")

        lines.append("-" * 70)
        lines.append("🎯 【详细号码】")
        lines.append("-" * 70)

        for idx, p in enumerate(pred, 1):
            lines.append(f"【预测 {idx}】")
            lines.append(
                f"   红: {sorted([f'{n:02d}' for n in p['red_dan']])} + {sorted([f'{n:02d}' for n in p['red_tuo']])}")
            lines.append(
                f"   蓝: {sorted([f'{n:02d}' for n in p['blue_dan']])} + {sorted([f'{n:02d}' for n in p['blue_tuo']])}")
            lines.append("")

        return "\n".join(lines)

    def refresh_chart(self):
        chart = self.chart_var.get()
        path = OUTPUT_DIR + "/" + chart + ".png"

        self._show_chart_placeholder("加载中...", "blue")
        self.root.update()

        def load_image():
            try:
                img = Image.open(path)
                w, h = img.size

                frame_w = self.chart_frame.winfo_width()
                frame_h = self.chart_frame.winfo_height()
                max_w = frame_w - 20 if frame_w > 20 else 800
                max_h = frame_h - 20 if frame_h > 20 else 500

                ratio = min(max_w / w, max_h / h, 1)
                new_w = int(w * ratio)
                new_h = int(h * ratio)

                img = img.resize((new_w, new_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                self.root.after(0, self._display_image, photo, new_w, new_h)

            except Exception as e:
                error_msg = str(e)
                self.root.after(0, self._show_chart_error, error_msg)

        threading.Thread(target=load_image, daemon=True).start()

    def _display_image(self, photo, width, height):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        self.chart_label = ttk.Label(self.chart_frame, image=photo)
        self.chart_label.image = photo  # 保持引用防止被垃圾回收
        self.chart_label.pack(fill=tk.BOTH, expand=True)

    def _show_chart_error(self, error_msg):
        self._show_chart_placeholder("加载图表失败:\n" + error_msg, "red")

    def open_folder(self):
        if os.path.exists(OUTPUT_DIR):
            os.startfile(OUTPUT_DIR)
        else:
            messagebox.showerror("错误", "文件夹不存在")

    def load_history(self, keyword=''):
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            if keyword:
                cursor.execute(f"""
                    SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date
                    FROM {TABLE_NAME} WHERE period LIKE %s
                    ORDER BY CAST(period AS UNSIGNED) DESC LIMIT 100
                """, (f'%{keyword}%',))
            else:
                cursor.execute(f"""
                    SELECT period, red1, red2, red3, red4, red5, red6, blue, draw_date
                    FROM {TABLE_NAME} ORDER BY CAST(period AS UNSIGNED) DESC LIMIT 100
                """)

            for row in cursor.fetchall():
                reds = f"{row[1]:02d} {row[2]:02d} {row[3]:02d} {row[4]:02d} {row[5]:02d} {row[6]:02d}"
                self.tree.insert('', tk.END, values=(row[0], reds, f"{row[7]:02d}", row[8]))

            cursor.close()
            conn.close()
        except Exception as e:
            print("加载历史数据失败: " + str(e))

    def search_history(self):
        self.load_history(self.search_entry.get().strip())


# ============== 主程序 ==============
def main():
    print("=" * 60)
    print("🚀 启动双色球预测系统 - 调优版")
    print("=" * 60)
    print("预测模块状态: " + ("✅ 已加载" if PREDICTION_MODULE_LOADED else "❌ 未加载"))
    if GENERATE_SIGNATURE:
        print("generate_predictions 签名: " + str(GENERATE_SIGNATURE))
    print("=" * 60)

    root = tk.Tk()
    app = LotteryApp(root)
    print("✅ GUI已创建成功")
    root.mainloop()
    print("程序已退出")


if __name__ == "__main__":
    main()
