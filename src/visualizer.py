import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from config import Config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


class LotteryVisualizer:
    """数据可视化"""

    def __init__(self, output_dir=Config.OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_red_ball_frequency(self, frequency: Dict[int, int]):
        """
        绘制红球频率图

        Args:
            frequency: 频率字典 {号码: 次数}
        """
        plt.figure(figsize=(15, 6))
        nums = list(frequency.keys())
        counts = list(frequency.values())

        bars = plt.bar(nums, counts, color='coral', edgecolor='darkred', alpha=0.7)

        # 标注数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=8)

        plt.xlabel('红球号码', fontsize=12)
        plt.ylabel('出现次数', fontsize=12)
        plt.title('红球历史出现频率统计', fontsize=14, fontweight='bold')
        plt.xticks(np.arange(1, 34))
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/red_ball_frequency.png', dpi=300)
        plt.close()
        print(f"图表已保存: {self.output_dir}/red_ball_frequency.png")

    def plot_blue_ball_frequency(self, frequency: Dict[int, int]):
        """绘制蓝球频率图"""
        plt.figure(figsize=(12, 6))
        nums = list(frequency.keys())
        counts = list(frequency.values())

        colors = ['skyblue' if num % 2 == 0 else 'salmon' for num in nums]
        bars = plt.bar(nums, counts, color=colors, edgecolor='black', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.xlabel('蓝球号码', fontsize=12)
        plt.ylabel('出现次数', fontsize=12)
        plt.title('蓝球历史出现频率统计', fontsize=14, fontweight='bold')
        plt.xticks(np.arange(1, 17))
        plt.grid(axis='y', alpha=0.3)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='偶数'),
            Patch(facecolor='salmon', label='奇数')
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/blue_ball_frequency.png', dpi=300)
        plt.close()
        print(f"图表已保存: {self.output_dir}/blue_ball_frequency.png")

    def plot_odd_even_ratio(self, ratio_stats: Dict[str, int]):
        """绘制奇偶比例统计图"""
        sorted_data = sorted(ratio_stats.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(labels, values, color='steelblue', alpha=0.7)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2.,
                     f'{int(width)}', ha='left', va='center', fontsize=10)

        plt.xlabel('出现次数', fontsize=12)
        plt.ylabel('奇偶比例', fontsize=12)
        plt.title('红球奇偶比例分布', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/odd_even_ratio.png', dpi=300)
        plt.close()
        print(f"图表已保存: {self.output_dir}/odd_even_ratio.png")

    def plot_consecutive_ratio(self, consecutive_stats: Dict[str, int]):
        """绘制连号比例图"""
        labels = list(consecutive_stats.keys())
        values = list(consecutive_stats.values())

        plt.figure(figsize=(10, 6))
        wedges, texts, autotexts = plt.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=['lightblue', 'lightgreen', 'orange', 'red'],
            startangle=90
        )

        plt.setp(autotexts, size=10, weight="bold")
        plt.title('连号出现比例分布', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/consecutive_ratio.png', dpi=300)
        plt.close()
        print(f"图表已保存: {self.output_dir}/consecutive_ratio.png")

    def plot_blue_ball_patterns(self, pattern_stats: Dict[str, int]):
        """绘制蓝球规律图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 单双
        single_double = {'单': pattern_stats['单'], '双': pattern_stats['双']}
        axes[0].pie(single_double.values(), labels=single_double.keys(),
                    autopct='%1.1f%%', colors=['salmon', 'skyblue'])
        axes[0].set_title('单双分布')

        # 大小
        big_small = {'大(9-16)': pattern_stats['大(9-16)'], '小(1-8)': pattern_stats['小(1-8)']}
        axes[1].pie(big_small.values(), labels=big_small.keys(),
                    autopct='%1.1f%%', colors=['orange', 'lightgreen'])
        axes[1].set_title('大小分布')

        # 质数合数
        prime_comp = {'质数': pattern_stats['质数'], '合数': pattern_stats['合数']}
        axes[2].pie(prime_comp.values(), labels=prime_comp.keys(),
                    autopct='%1.1f%%', colors=['purple', 'gold'])
        axes[2].set_title('质数合数分布')

        plt.suptitle('蓝球规律分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/blue_ball_patterns.png', dpi=300)
        plt.close()
        print(f"图表已保存: {self.output_dir}/blue_ball_patterns.png")
