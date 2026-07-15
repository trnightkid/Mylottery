"""
双色球预测分析 - GUI界面版 (兼容版)
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
import sys
import traceback
import inspect

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
GENERATE_SIGNATURE = None  # 存储函数签名


def try_import_prediction_module():
    global PREDICTION_MODULE_LOADED, PREDICTION_MODULE_ERROR, GENERATE_SIGNATURE

    script_dir = os.path.dirname(os.path.abspath(__file__))
    py_file = os.path.join(script_dir, "lottery_dantuo_prediction.py")

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

        # 检查必需的函数
        required_funcs = ['generate_predictions', 'load_data', 'plot_prediction_pools', 'fit_distributions',
                          'calculate_blue_probs']
        missing = [f for f in required_funcs if not hasattr(spec, f)]

        if missing:
            PREDICTION_MODULE_ERROR = f"缺少函数: {', '.join(missing)}"
            return False

        # 获取函数签名
        try:
            GENERATE_SIGNATURE = inspect.signature(spec.generate_predictions)
            print(f"✅ 预测模块加载成功!")
            print(f"   generate_predictions 签名: {GENERATE_SIGNATURE}")
        except:
            print(f"✅ 预测模块加载成功!")
            print(f"   (无法获取函数签名)")

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
    try:
        url = "https://datachart.500star.com/ssq/history/newinc/history.php"
        params = {'start': '26001', 'end': '26050'}
        response = requests.get(url, params=params, timeout=30)
        response.encoding = 'gbk'
        import json
        data = json.loads(response.text)
        results = []
        items = data if isinstance(data, list) else data.get('list', [])
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
        return results
    except Exception as e:
        print(f"爬取失败: {e}")
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
        self.root.title("🎱 双色球胆拖预测系统")
        self.root.geometry("1200x750")
        self.root.minsize(900, 600)

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
            if GENERATE_SIGNATURE:
                ttk.Label(module_frame, text=f"函数签名: {GENERATE_SIGNATURE}", foreground="blue",
                          font=('Arial', 8)).pack(anchor=tk.W)
        else:
            error_text = (PREDICTION_MODULE_ERROR or "未知错误")[:50]
            ttk.Label(module_frame, text="❌ 预测模块未加载", foreground="red").pack(anchor=tk.W)
            ttk.Label(module_frame, text=f"原因: {error_text}...", foreground="gray", font=('Arial', 8)).pack(
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

        # 参数汇总
        self.param_summary = ttk.Label(param_frame, text="", foreground="blue", font=('Arial', 9))
        self.param_summary.grid(row=6, column=0, columnspan=3, pady=(10, 0))
        self.update_param_summary()

        for spin in [self.n_dan_red, self.n_tuo_hot_red, self.n_tuo_cold_red,
                     self.n_dan_blue, self.n_tuo_blue]:
            spin.bind('<<Increment>>', self.update_param_summary)
            spin.bind('<<Decrement>>', self.update_param_summary)

        # 预测按钮
        ttk.Button(left, text="🎯 开始预测", command=self.run_prediction).pack(fill=tk.X, pady=5)

        self.predict_status = ttk.Label(left, text="就绪", foreground="gray")
        self.predict_status.pack(anchor=tk.W)

        ttk.Button(left, text="🔄 刷新状态", command=self.refresh_db).pack(fill=tk.X, pady=5)

        # ===== 右侧选项卡 =====
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="📋 预测结果")
        self.result_text = ScrolledText(self.tab1, wrap=tk.WORD, font=('Consolas', 11))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.show_welcome()

        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="📈 图表分析")

        chart_sel = ttk.Frame(self.tab2)
        chart_sel.pack(fill=tk.X, padx=10, pady=5)

        self.chart_var = tk.StringVar(value='dan_tuo_pools')
        for text, val in [('综合分析', 'dan_tuo_pools'), ('拟合曲线', 'fitting_curve')]:
            ttk.Radiobutton(chart_sel, text=text, variable=self.chart_var, value=val, command=self.show_chart).pack(
                side=tk.LEFT, padx=5)

        ttk.Button(chart_sel, text="📂 打开文件夹", command=self.open_folder).pack(side=tk.RIGHT)

        self.chart_frame = ttk.Frame(self.tab2)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        ttk.Label(self.chart_frame, text="请先运行预测", foreground="gray").place(relx=0.5, rely=0.5, anchor=tk.CENTER)

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

        self.refresh_db()
        self.load_history()

    def get_params(self):
        try:
            return {
                'dan_red': int(self.n_dan_red.get()),
                'tuo_hot_red': int(self.n_tuo_hot_red.get()),
                'tuo_cold_red': int(self.n_tuo_cold_red.get()),
                'dan_blue': int(self.n_dan_blue.get()),
                'tuo_blue': int(self.n_tuo_blue.get()),
                'n_predictions': int(self.n_pred.get())
            }
        except:
            return {
                'dan_red': DEFAULT_N_DAN_RED,
                'tuo_hot_red': DEFAULT_N_TUO_HOT_RED,
                'tuo_cold_red': DEFAULT_N_TUO_COLD_RED,
                'dan_blue': DEFAULT_N_DAN_BLUE,
                'tuo_blue': DEFAULT_N_TUO_BLUE,
                'n_predictions': DEFAULT_N_PREDICTIONS
            }

    def update_param_summary(self, event=None):
        p = self.get_params()
        total_red = p['dan_red'] + p['tuo_hot_red'] + p['tuo_cold_red']
        text = f"红球: {p['dan_red']}胆+{p['tuo_hot_red']}热+{p['tuo_cold_red']}冷={total_red}个\n"
        text += f"蓝球: {p['dan_blue']}胆+{p['tuo_blue']}拖={p['dan_blue'] + p['tuo_blue']}个"
        self.param_summary.config(text=text)

    def show_welcome(self):
        self.result_text.insert(tk.END, "=" * 60 + "\n")
        self.result_text.insert(tk.END, "🎱 双色球胆拖投注预测系统\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n\n")
        if not PREDICTION_MODULE_LOADED:
            self.result_text.insert(tk.END, f"⚠️ 预测模块未加载!\n错误: {PREDICTION_MODULE_ERROR}\n\n")
        self.result_text.insert(tk.END, "使用说明：\n")
        self.result_text.insert(tk.END, "1. 点击「同步CSV数据」或「爬取最新数据」\n")
        self.result_text.insert(tk.END, "2. 调整右侧预测参数\n")
        self.result_text.insert(tk.END, "3. 点击「开始预测」生成预测结果\n")
        self.result_text.insert(tk.END, "\n⚠️ 预测结果仅供参考，请理性购彩！\n")

    def refresh_db(self):
        ok, count, latest = check_db_status()
        if ok:
            self.db_status.config(text="✅ 已连接", foreground="green")
            self.data_count.config(text=f"数据量: {count} 条 | 最新期: {latest}")
        else:
            self.db_status.config(text=f"❌ {latest}", foreground="red")

    def sync_csv(self):
        self.crawl_status.config(text="同步中...", foreground="blue")
        self.root.update()

        def task():
            result, msg = sync_from_csv()
            self.root.after(0, lambda: self.crawl_status.config(
                text=msg if result > 0 else f"❌ {msg}",
                foreground="green" if result > 0 else "red"))
            if result > 0:
                self.root.after(0, self.refresh_db)
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
                    text=f"✅ 获取{len(data)}期，保存{new}条", foreground="green"))
                self.root.after(0, self.refresh_db)
            else:
                self.root.after(0, lambda: self.crawl_status.config(text="❌ 爬取失败", foreground="red"))

        threading.Thread(target=task, daemon=True).start()

    def run_prediction(self):
        if not PREDICTION_MODULE_LOADED:
            messagebox.showerror("模块错误", f"无法加载预测模块!\n\n{PREDICTION_MODULE_ERROR}")
            return

        try:
            params = self.get_params()
            if params['dan_red'] + params['tuo_hot_red'] + params['tuo_cold_red'] > 33:
                messagebox.showerror("错误", "红球总数超过33个")
                return
            if params['dan_blue'] + params['tuo_blue'] > 16:
                messagebox.showerror("错误", "蓝球总数超过16个")
                return
        except Exception as err:
            messagebox.showerror("错误", f"参数无效: {err}")
            return

        self.predict_status.config(text="预测中...", foreground="blue")

        def task():
            try:
                import lottery_dantuo_prediction as pred_module

                load_data = pred_module.load_data
                generate_predictions = pred_module.generate_predictions
                plot_prediction_pools = pred_module.plot_prediction_pools

                df = load_data()
                if df is None or len(df) < 10:
                    self.root.after(0, lambda: self.predict_status.config(
                        text="❌ 数据不足", foreground="red"))
                    return

                # 根据函数签名决定如何调用
                try:
                    # 尝试使用蓝球参数调用
                    pred, red_p, blue_p, x, pdf = generate_predictions(
                        df,
                        n=params['n_predictions'],
                        n_dan_blue=params['dan_blue'],
                        n_tuo_blue=params['tuo_blue']
                    )
                except TypeError:
                    # 如果不支持蓝球参数，使用简化调用
                    print("⚠️ generate_predictions 不支持蓝球参数，使用默认参数")
                    pred, red_p, blue_p, x, pdf = generate_predictions(
                        df,
                        n=params['n_predictions']
                    )

                plot_prediction_pools(pred, red_p, blue_p, x, pdf, OUTPUT_DIR)

                # 显示结果
                text = self.result_text
                text.delete(1.0, tk.END)

                text.insert(tk.END, "=" * 70 + "\n")
                text.insert(tk.END, f"🎱 预测结果 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                text.insert(tk.END, "=" * 70 + "\n\n")
                text.insert(tk.END,
                            f"参数: 红胆{params['dan_red']} + 红拖{params['tuo_hot_red']}热{params['tuo_cold_red']}冷 + 蓝胆{params['dan_blue']} + 蓝拖{params['tuo_blue']}\n\n")

                for p in pred:
                    text.insert(tk.END, f"【预测{p['group']}】\n")
                    text.insert(tk.END, f"  🔴 红球胆码: {[f'{n:02d}' for n in p['red_dan']]}\n")
                    text.insert(tk.END, f"  🟡 红球拖码: {[f'{n:02d}' for n in p['red_tuo']]}\n")
                    text.insert(tk.END, f"  🔵 蓝球胆码: {[f'{n:02d}' for n in p['blue_dan']]}\n")
                    text.insert(tk.END, f"  🟦 蓝球拖码: {[f'{n:02d}' for n in p['blue_tuo']]}\n\n")

                text.insert(tk.END, f"📁 图表已保存至: {OUTPUT_DIR}\n")

                self.root.after(0, lambda: self.predict_status.config(
                    text="✅ 预测完成", foreground="green"))
                self.root.after(0, lambda: self.notebook.select(self.tab1))
                self.root.after(0, self.show_chart)

            except Exception as err:
                error_msg = f"{err}\n\n{traceback.format_exc()}"
                print(error_msg)
                self.root.after(0, lambda: self.predict_status.config(
                    text=f"❌ 预测失败", foreground="red"))
                self.root.after(0, lambda e=error_msg: messagebox.showerror("预测错误", e))

        threading.Thread(target=task, daemon=True).start()

    def show_chart(self):
        chart = self.chart_var.get()
        path = f"{OUTPUT_DIR}/{chart}.png"

        for w in self.chart_frame.winfo_children():
            w.destroy()

        if not os.path.exists(path):
            ttk.Label(self.chart_frame, text="请先运行预测", foreground="gray").place(relx=0.5, rely=0.5,
                                                                                      anchor=tk.CENTER)
            return

        try:
            img = Image.open(path)
            w, h = img.size
            max_w = self.chart_frame.winfo_width() - 20 or 800
            max_h = self.chart_frame.winfo_height() - 20 or 500

            ratio = min(max_w / w, max_h / h, 1)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            lbl = ttk.Label(self.chart_frame, image=photo)
            lbl.image = photo
            lbl.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            print(f"显示图表失败: {e}")

    def open_folder(self):
        if os.path.exists(OUTPUT_DIR):
            os.startfile(OUTPUT_DIR)

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
        except:
            pass

    def search_history(self):
        self.load_history(self.search_entry.get().strip())


# ============== 主程序 ==============
def main():
    print("🚀 启动程序...")
    print(f"预测模块状态: {'✅ 已加载' if PREDICTION_MODULE_LOADED else '❌ 未加载'}")
    if GENERATE_SIGNATURE:
        print(f"generate_predictions 签名: {GENERATE_SIGNATURE}")

    root = tk.Tk()
    app = LotteryApp(root)

    print("✅ GUI已创建")
    root.mainloop()
    print("程序已退出")


if __name__ == "__main__":
    main()
