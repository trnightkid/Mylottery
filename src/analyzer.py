import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class LotteryAnalyzer:
    """双色球数据分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.red_range = (1, 33)
        self.blue_range = (1, 16)

    def analyze_red_ball_frequency(self) -> Dict[int, int]:
        """
        分析红球频率

        Returns:
            字典格式，key为号码，value为出现次数
        """
        red_columns = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        all_red_balls = self.df[red_columns].values.flatten()
        frequency = Counter(all_red_balls)

        # 确保所有可能的号码都在统计中（出现0次的也要显示）
        all_possible_reds = range(self.red_range[0], self.red_range[1] + 1)
        for num in all_possible_reds:
            if num not in frequency:
                frequency[num] = 0

        return dict(sorted(frequency.items()))

    def analyze_blue_ball_frequency(self) -> Dict[int, int]:
        """
        分析蓝球频率

        Returns:
            字典格式，key为号码，value为出现次数
        """
        all_blue_balls = self.df['blue'].values
        frequency = Counter(all_blue_balls)

        # 确保所有可能的号码都在统计中
        all_possible_blues = range(self.blue_range[0], self.blue_range[1] + 1)
        for num in all_possible_blues:
            if num not in frequency:
                frequency[num] = 0

        return dict(sorted(frequency.items()))

    def get_hot_and_cold_numbers(self, frequency_dict: Dict[int, int],
                                 top_n: int = 10) -> Tuple[List[int], List[int]]:
        """
        获取热号和冷号

        Args:
            frequency_dict: 频率字典
            top_n: 返回前N个号码

        Returns:
            (热号列表, 冷号列表)
        """
        sorted_nums = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
        hot_numbers = [num for num, _ in sorted_nums[:top_n]]
        cold_numbers = [num for num, _ in sorted_nums[-top_n:]]
        return hot_numbers, cold_numbers

    def calculate_missing_value(self, frequency_dict: Dict[int, int],
                                total_draws: int) -> Dict[int, int]:
        """
        计算遗漏值（号码多久没出现）

        Args:
            frequency_dict: 频率字典
            total_draws: 总开奖期数

        Returns:
            遗漏值字典，key为号码，value为连续未出现的期数
        """
        missing = {num: 0 for num in frequency_dict.keys()}

        # 从最新的期号开始倒序遍历
        for _, row in self.df.iterrows():
            red_balls = [row['red1'], row['red2'], row['red3'],
                         row['red4'], row['red5'], row['red6']]
            blue_ball = row['blue']

            # 对于每个号码，如果在当期出现，则重置遗漏值
            for num in missing.keys():
                if num in red_balls or num == blue_ball:
                    missing[num] = 0
                else:
                    missing[num] += 1

        return missing

    def analyze_consecutive_numbers(self) -> Dict[str, int]:
        """
        分析连号出现情况

        Returns:
            连号统计字典
        """
        consecutive_stats = {
            '0连号': 0,
            '1组连号': 0,
            '2组连号': 0,
            '3组连号及以上': 0
        }

        for _, row in self.df.iterrows():
            red_balls = sorted([row['red1'], row['red2'], row['red3'],
                                row['red4'], row['red5'], row['red6']])

            consecutive_groups = 0
            for i in range(len(red_balls) - 1):
                if red_balls[i + 1] - red_balls[i] == 1:
                    consecutive_groups += 1
                # 避免连号重复计数（如1,2,3算作1组而不是2组）
                if i > 0 and red_balls[i + 1] - red_balls[i] == 1 \
                        and red_balls[i] - red_balls[i - 1] == 1:
                    consecutive_groups -= 1

            if consecutive_groups == 0:
                consecutive_stats['0连号'] += 1
            elif consecutive_groups == 1:
                consecutive_stats['1组连号'] += 1
            elif consecutive_groups == 2:
                consecutive_stats['2组连号'] += 1
            else:
                consecutive_stats['3组连号及以上'] += 1

        return consecutive_stats

    def analyze_blue_ball_patterns(self) -> Dict[str, int]:
        """
        分析蓝球规律：单双、大小、奇偶
        """
        patterns = {
            '单': 0, '双': 0,
            '大(9-16)': 0, '小(1-8)': 0,
            '质数': 0, '合数': 0
        }

        primes = {2, 3, 5, 7, 11, 13}

        for blue in self.df['blue']:
            # 单双
            if blue % 2 == 1:
                patterns['单'] += 1
            else:
                patterns['双'] += 1

            # 大小
            if blue >= 9:
                patterns['大(9-16)'] += 1
            else:
                patterns['小(1-8)'] += 1

            # 质数合数
            if blue in primes:
                patterns['质数'] += 1
            else:
                patterns['合数'] += 1

        return patterns

    def calculate_odd_even_ratio(self) -> Dict[str, int]:
        """
        计算红球中奇偶比例的出现频率
        """
        ratio_stats = Counter()

        for _, row in self.df.iterrows():
            red_balls = [row['red1'], row['red2'], row['red3'],
                         row['red4'], row['red5'], row['red6']]
            odd_count = sum(1 for num in red_balls if num % 2 == 1)
            ratio = f"{odd_count}奇{6 - odd_count}偶"
            ratio_stats[ratio] += 1

        return dict(ratio_stats)

    def generate_report(self) -> str:
        """生成分析报告"""
        red_freq = self.analyze_red_ball_frequency()
        blue_freq = self.analyze_blue_ball_frequency()
        red_hot, red_cold = self.get_hot_and_cold_numbers(red_freq, top_n=10)
        blue_hot, blue_cold = self.get_hot_and_cold_numbers(blue_freq, top_n=5)
        missing = self.calculate_missing_value(red_freq, len(self.df))
        consecutive = self.analyze_consecutive_numbers()
        blue_patterns = self.analyze_blue_ball_patterns()
        odd_even = self.calculate_odd_even_ratio()

        report = []
        report.append("=" * 60)
        report.append("双色球历史数据分析报告")
        report.append("=" * 60)
        report.append(f"\n【基础信息】")
        report.append(f"分析期数: {len(self.df)}")
        report.append(f"期号范围: {self.df['period'].min()} - {self.df['period'].max()}")

        # 热号冷号
        report.append(f"\n【红球热号TOP10】")
        report.append(f"热号: {', '.join(map(str, red_hot))}")
        report.append(f"冷号: {', '.join(map(str, red_cold))}")

        report.append(f"\n【蓝球热号TOP5】")
        report.append(f"热号: {', '.join(map(str, blue_hot))}")
        report.append(f"冷号: {', '.join(map(str, blue_cold))}")

        # 遗漏值TOP10
        sorted_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)
        report.append(f"\n【红球遗漏值TOP10】")
        for num, miss in sorted_missing[:10]:
            report.append(f"  号码 {num:2d}: 已遗漏 {miss} 期")

        # 连号统计
        report.append(f"\n【连号统计】")
        for key, val in consecutive.items():
            percentage = val / len(self.df) * 100
            report.append(f"  {key}: {val} 次 ({percentage:.2f}%)")

        # 蓝球规律
        report.append(f"\n【蓝球规律统计】")
        for key, val in blue_patterns.items():
            percentage = val / len(self.df) * 100
            report.append(f"  {key}: {val} 次 ({percentage:.2f}%)")

        # 奇偶比例
        report.append(f"\n【红球奇偶比例统计】")
        sorted_ratio = sorted(odd_even.items(), key=lambda x: x[1], reverse=True)
        for ratio, count in sorted_ratio:
            percentage = count / len(self.df) * 100
            report.append(f"  {ratio}: {count} 次 ({percentage:.2f}%)")

        report.append("\n" + "=" * 60)
        report.append("报告生成完成")
        report.append("=" * 60)

        return "\n".join(report)