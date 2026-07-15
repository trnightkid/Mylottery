import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)


class LotteryPredictor:
    """双色球概率预测器"""

    def __init__(self, df, recent_weight=Config.RECENT_WEIGHT):
        """
        初始化预测器

        Args:
            df: 数据DataFrame
            recent_weight: 近期数据权重(0-1)
        """
        self.df = df
        self.recent_weight = recent_weight
        self.history_weight = 1 - recent_weight
        self.red_probabilities = {}
        self.blue_probabilities = {}

    def _weighted_frequency(self, numbers: np.ndarray,
                            split_ratio: float = 0.3) -> Tuple[Counter, Counter]:
        """
        计算加权频率

        Args:
            numbers: 号码数组
            split_ratio: 分界比例，用于区分近期和历史数据

        Returns:
            (近期频率, 历史频率)
        """
        split_index = int(len(numbers) * (1 - split_ratio))
        recent_numbers = numbers[split_index:]
        history_numbers = numbers[:split_index]

        recent_freq = Counter(recent_numbers)
        history_freq = Counter(history_numbers)

        return recent_freq, history_freq

    def calculate_red_probabilities(self) -> Dict[int, float]:
        """
        计算红球出现概率

        Returns:
            字典，key为号码，value为概率
        """
        red_columns = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        all_red_balls = self.df[red_columns].values.flatten()

        recent_freq, history_freq = self._weighted_frequency(all_red_balls)

        # 计算加权概率
        total_recent = sum(recent_freq.values())
        total_history = sum(history_freq.values())

        probabilities = {}
        for num in range(1, 34):
            recent_count = recent_freq.get(num, 0)
            history_count = history_freq.get(num, 0)

            # 避免除以0
            recent_prob = recent_count / total_recent if total_recent > 0 else 0
            history_prob = history_count / total_history if total_history > 0 else 0

            # 加权计算
            probabilities[num] = (
                    recent_prob * self.recent_weight +
                    history_prob * self.history_weight
            )

        self.red_probabilities = probabilities
        return probabilities

    def calculate_blue_probabilities(self) -> Dict[int, float]:
        """
        计算蓝球出现概率

        Returns:
            字典，key为号码，value为概率
        """
        all_blue_balls = self.df['blue'].values

        recent_freq, history_freq = self._weighted_frequency(all_blue_balls)

        total_recent = sum(recent_freq.values())
        total_history = sum(history_freq.values())

        probabilities = {}
        for num in range(1, 17):
            recent_count = recent_freq.get(num, 0)
            history_count = history_freq.get(num, 0)

            recent_prob = recent_count / total_recent if total_recent > 0 else 0
            history_prob = history_count / total_history if total_history > 0 else 0

            probabilities[num] = (
                    recent_prob * self.recent_weight +
                    history_prob * self.history_weight
            )

        self.blue_probabilities = probabilities
        return probabilities

    def predict_top_numbers(self, probabilities: Dict[int, float],
                            top_n: int = 10) -> List[Tuple[int, float]]:
        """
        预测TOP N号码

        Args:
            probabilities: 概率字典
            top_n: 返回前N个

        Returns:
            [(号码, 概率), ...] 列表
        """
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:top_n]

    def predict_probability_of_number(self, number: int, is_red: bool = True) -> float:
        """
        预测某个特定号码的出现概率

        Args:
            number: 号码
            is_red: 是否为红球

        Returns:
            概率值(0-1)
        """
        if is_red:
            if not self.red_probabilities:
                self.calculate_red_probabilities()
            return self.red_probabilities.get(number, 0)
        else:
            if not self.blue_probabilities:
                self.calculate_blue_probabilities()
            return self.blue_probabilities.get(number, 0)

    def generate_recommendation(self) -> Dict:
        """
        生成推荐号码组合

        Returns:
            推荐字典，包含多种推荐方式
        """
        self.calculate_red_probabilities()
        self.calculate_blue_probabilities()

        recommendations = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'based_on_draws': len(self.df),
            'methods': {}
        }

        # 方法1: 基于高概率推荐
        top_red_prob = self.predict_top_numbers(self.red_probabilities, top_n=10)
        top_blue_prob = self.predict_top_numbers(self.blue_probabilities, top_n=5)

        recommendations['methods']['高概率推荐'] = {
            'red_candidates': [num for num, _ in top_red_prob],
            'red_with_prob': top_red_prob,
            'blue_candidates': [num for num, _ in top_blue_prob],
            'blue_with_prob': top_blue_prob
        }

        # 方法2: 基于遗漏值推荐（反向思维）
        # 从analyzer中获取遗漏值数据
        missing_values = self._calculate_missing_for_prediction()
        sorted_missing = sorted(missing_values.items(), key=lambda x: x[1], reverse=True)

        recommendations['methods']['遗漏值推荐(冷号反弹)'] = {
            'red_candidates': [num for num, _ in sorted_missing[:10]],
            'missing_values': sorted_missing[:10]
        }

        # 方法3: 混合推荐(平衡法) - 综合概率和遗漏值
        mixed_red = self._calculate_mixed_scores()
        sorted_mixed = sorted(mixed_red.items(), key=lambda x: x[1], reverse=True)

        recommendations['methods']['混合推荐(平衡法)'] = {
            'red_candidates': [num for num, _ in sorted_mixed[:10]],
            'scores': sorted_mixed[:10]
        }

        # 生成一组推荐号码(从混合推荐中选择)
        mixed_red_sorted = [num for num, _ in sorted_mixed]
        selected_red = sorted(mixed_red_sorted[:6])
        selected_blue = top_blue_prob[0][0] if top_blue_prob else 8

        recommendations['final_recommendation'] = {
            'red_balls': selected_red,
            'blue_ball': selected_blue,
            'description': '基于混合推荐算法的最优组合'
        }

        return recommendations

    def _calculate_missing_for_prediction(self) -> Dict[int, int]:
        """计算遗漏值"""
        missing = {num: 0 for num in range(1, 34)}

        for _, row in self.df.iterrows():
            red_balls = [row['red1'], row['red2'], row['red3'],
                         row['red4'], row['red5'], row['red6']]

            for num in missing.keys():
                if num in red_balls:
                    missing[num] = 0
                else:
                    missing[num] += 1

        return missing

    def _calculate_mixed_scores(self) -> Dict[int, float]:
        """
        计算混合评分 = 概率 * 0.7 + (1/(遗漏值+1)) * 0.3
        """
        missing = self._calculate_missing_for_prediction()
        mixed_scores = {}

        for num in range(1, 34):
            prob = self.red_probabilities.get(num, 0)
            miss_value = missing.get(num, 0)

            # 遗漏值倒数作为冷号反弹因子
            miss_factor = 1 / (miss_value + 1)

            # 混合评分
            mixed_score = prob * 0.7 + miss_factor * 0.3 * 0.01
            mixed_scores[num] = mixed_score

        return mixed_scores