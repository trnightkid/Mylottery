#!/usr/bin/env python3
"""
双色球避号系统预测器 v3
=========================
核心逻辑：
1. 分析近几期彩民购买热度的变化趋势
2. 预测下一期哪些号码会成为"热门"
3. "避号系统"会刻意避开这些热门号
4. 所以真正的开奖号码 = 预测的"冷门号"

简言之：热号的补集 = 预测的开奖号

注意：此分析仅供娱乐研究
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import json
import os

DATA_FILE = 'lottery_data.csv'
OUTPUT_DIR = 'output'


class AvoidPredictor:
    """
    避号系统预测器
    
    核心假设：
    - 彩民购买有惯性：近期热号下期可能继续热
    - 机构知道这个趋势，故意选"冷的补集"
    - 所以：预测热号 → 热号的补集就是开奖号
    """
    
    def __init__(self, df):
        self.df = df.sort_values('period', ascending=False).reset_index(drop=True)
        self.red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
    
    def analyze_recent_trend(self, window=10):
        """
        分析近window期的热度趋势
        返回每个号码的"热度评分"
        """
        print("\n" + "="*60)
        print(f"📊 热度趋势分析 (近{window}期)")
        print("="*60)
        
        # 加权频率：近期权重更高
        heat_scores = defaultdict(float)
        
        for i in range(window):
            if i >= len(self.df):
                break
            
            row = self.df.iloc[i]
            # 时间衰减权重：越近期权重越高
            weight = 1.0 / (1 + i * 0.15)
            
            for c in self.red_cols:
                num = int(row[c])
                heat_scores[num] += weight
        
        # 归一化
        max_score = max(heat_scores.values()) if heat_scores else 1
        for num in heat_scores:
            heat_scores[num] /= max_score
        
        # 排序
        sorted_heat = sorted(heat_scores.items(), key=lambda x: x[1], reverse=True)
        
        hot_numbers = set()   # 热门号
        cold_numbers = set()  # 冷门号
        
        # 热度最高的6-8个号定义为"热号"
        for num, score in sorted_heat[:7]:
            hot_numbers.add(num)
        
        # 热度最低的号码定义为"冷门"
        for num, score in sorted_heat[-15:]:
            cold_numbers.add(num)
        
        print(f"\n  🔥 预测热号 (机构会回避): {sorted(hot_numbers)}")
        print(f"  ❄️ 预测冷号 (可能开奖): {sorted(cold_numbers)}")
        
        return {
            'heat_scores': dict(heat_scores),
            'hot_numbers': sorted(hot_numbers),
            'cold_numbers': sorted(cold_numbers),
            'sorted_heat': [(n, round(s, 3)) for n, s in sorted_heat]
        }
    
    def predict_by_heat_complement(self, window=10):
        """
        热度补集预测法
        
        逻辑：
        - 预测热号 → 机构会回避
        - 真正的开奖号 ≈ 热号的补集（冷门号）
        """
        print("\n" + "="*60)
        print("🎯 热度补集预测法")
        print("="*60)
        
        heat = self.analyze_recent_trend(window)
        
        hot = set(heat['hot_numbers'])
        cold = set(heat['cold_numbers'])
        all_nums = set(range(1, 34))
        
        # 热号的补集 = 可能被选为开奖号的候选
        complement = all_nums - hot
        
        print(f"\n  📌 热号: {sorted(hot)}")
        print(f"  📌 热号补集 (候选开奖号): {sorted(complement)}")
        
        # 生成预测组合
        predictions = []
        
        # 策略1: 纯冷门组合 - 主要从补集中选
        for _ in range(3):
            # 从补集中选4个
            from_complement = list(complement)
            np.random.shuffle(from_complement)
            selected = from_complement[:4]
            
            # 从冷号中补2个
            from_cold = list(cold - set(selected))
            np.random.shuffle(from_cold)
            selected.extend(from_cold[:2])
            
            predictions.append(sorted(selected[:6]))
        
        # 策略2: 纯补集组合 - 全部从补集中选
        for _ in range(2):
            from_complement = list(complement)
            np.random.shuffle(from_complement)
            predictions.append(sorted(from_complement[:6]))
        
        return {
            'hot_numbers': sorted(hot),
            'cold_numbers': sorted(cold),
            'complement': sorted(complement),
            'predictions': predictions
        }
    
    def predict_by_missing_reverse(self, window=20):
        """
        遗漏值反向预测法
        
        逻辑：
        - 高遗漏(很久没出) = 可能被"压制"的冷号
        - 机构倾向选这些被压制的冷号
        - 所以：高遗漏 = 可能开奖号
        """
        print("\n" + "="*60)
        print("🎯 遗漏值反向预测法")
        print("="*60)
        
        # 计算每个号码的遗漏值
        last_appear = {}
        periods = []
        
        for i in range(min(window * 2, len(self.df))):
            row = self.df.iloc[i]
            period = int(row['period'])
            periods.append(period)
            for c in self.red_cols:
                num = int(row[c])
                last_appear[num] = period
        
        max_period = max(periods)
        
        missing = {}
        for num in range(1, 34):
            if num in last_appear:
                missing[num] = max_period - last_appear[num]
            else:
                missing[num] = 999
        
        # 按遗漏值排序
        sorted_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  ⏱️ 当前最高遗漏:")
        for num, m in sorted_missing[:8]:
            print(f"     号码{num:02d}: {m}期")
        
        # 遗漏最高的号码 = 可能被选为开奖号
        high_missing = set([n for n, m in sorted_missing[:15]])
        
        predictions = []
        for _ in range(3):
            from_missing = [n for n, m in sorted_missing[:12]]
            np.random.shuffle(from_missing)
            predictions.append(sorted(from_missing[:6]))
        
        return {
            'missing_values': {str(k): v for k, v in sorted_missing},
            'high_missing': sorted(high_missing),
            'predictions': predictions
        }
    
    def predict_by_frequency_inverse(self, window=10):
        """
        频率反向预测法
        
        逻辑：
        - 近{window}期出现频率最低的号码
        - 机构为了"平衡"，可能会选这些低频号
        """
        print("\n" + "="*60)
        print("🎯 频率反向预测法")
        print("="*60)
        
        freq = Counter()
        for i in range(window):
            if i >= len(self.df):
                break
            row = self.df.iloc[i]
            for c in self.red_cols:
                freq[int(row[c])] += 1
        
        # 按频率排序（升序 = 最不常出现的）
        sorted_freq = sorted(freq.items(), key=lambda x: x[1])
        
        print(f"\n  📉 出现次数最少的号码:")
        for num, count in sorted_freq[:8]:
            print(f"     号码{num:02d}: {count}次")
        
        # 最低频的号码 = 预测开奖号
        low_freq = set([n for n, c in sorted_freq[:12]])
        
        predictions = []
        for _ in range(3):
            from_low = [n for n, c in sorted_freq[:12]]
            np.random.shuffle(from_low)
            predictions.append(sorted(from_low[:6]))
        
        return {
            'frequency': {str(k): v for k, v in sorted_freq},
            'low_frequency': sorted(low_freq),
            'predictions': predictions
        }
    
    def backtest_heat_complement(self, test_periods=50):
        """
        回测：热度补集预测法的准确率
        """
        print("\n" + "="*60)
        print(f"📈 回测验证 (热度补集法, 最近{test_periods}期)")
        print("="*60)
        
        total_hit = 0
        hit_details = []
        
        for i in range(test_periods):
            if i + 10 >= len(self.df):
                break
            
            # 用前10期预测
            train_df = self.df.iloc[i+10:i+20]
            actual_row = self.df.iloc[i]
            
            # 计算热度
            heat = defaultdict(float)
            for idx, row in train_df.iterrows():
                weight = 1.0 / (1 + idx * 0.15)
                for c in self.red_cols:
                    num = int(row[c])
                    heat[num] += weight
            
            # 热号
            hot = set([n for n, s in sorted(heat.items(), key=lambda x: x[1], reverse=True)[:7]])
            
            # 热号补集
            complement = set(range(1, 34)) - hot
            
            # 实际开奖
            actual = set([int(actual_row[c]) for c in self.red_cols])
            
            # 计算命中
            hit = len(complement & actual)
            total_hit += hit
            
            hit_details.append({
                'period': actual_row['period'],
                'hit': hit,
                'actual': sorted(actual)
            })
        
        avg_hit = total_hit / len(hit_details) if hit_details else 0
        
        print(f"\n  📊 平均每期从补集中命中: {avg_hit:.2f}个")
        print(f"  📊 理论期望值: 约5.45个 (27/33 × 6)")
        
        if avg_hit > 5.5:
            print(f"  ⚠️ 命中偏高，可能存在'避热号'效应")
        elif avg_hit < 4.5:
            print(f"  ⚠️ 命中偏低，可能存在其他机制")
        else:
            print(f"  ✅ 接近理论值")
        
        return {
            'avg_hit': avg_hit,
            'theoretical': 27/33*6,
            'details': hit_details[:10]
        }
    
    def generate_final_prediction(self):
        """
        综合多种策略，生成最终预测
        """
        print("\n" + "="*60)
        print("🏆 综合预测结果")
        print("="*60)
        
        # 运行各种预测
        heat_result = self.predict_by_heat_complement(window=10)
        missing_result = self.predict_by_missing_reverse(window=20)
        freq_result = self.predict_by_frequency_inverse(window=10)
        
        # 综合分析
        all_candidates = set(range(1, 34))
        
        # 各策略认为的"冷门号"
        heat_cold = set(heat_result['cold_numbers'])
        missing_high = set(missing_result['high_missing'][:12])
        freq_low = set(freq_result['low_frequency'][:10])
        
        # 综合冷门分数
        cold_scores = {}
        for num in range(1, 34):
            score = 0
            if num in heat_cold:
                score += 1
            if num in missing_high:
                score += 1
            if num in freq_low:
                score += 1
            cold_scores[num] = score
        
        sorted_cold = sorted(cold_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  📌 综合冷门评分 (3=三策略都认为冷):")
        for num, score in sorted_cold[:12]:
            stars = "⭐" * score
            print(f"     号码{num:02d}: {score}分 {stars}")
        
        # 最终推荐
        final_cold = [n for n, s in sorted_cold if s >= 2]
        
        print(f"\n  🎱 最终推荐号码 (2+策略认可的冷门号): {sorted(final_cold)}")
        
        # 生成5组推荐
        predictions = []
        for _ in range(5):
            if len(final_cold) >= 6:
                np.random.shuffle(final_cold)
                predictions.append(sorted(final_cold[:6]))
            else:
                # 补一些单策略冷门号
                pool = final_cold.copy()
                for n, s in sorted_cold:
                    if n not in pool and len(pool) < 6:
                        pool.append(n)
                np.random.shuffle(pool)
                predictions.append(sorted(pool[:6]))
        
        print(f"\n  🏆 推荐号码组合 (5组):")
        for i, reds in enumerate(predictions, 1):
            print(f"     组合{i}: {reds}")
        
        return {
            'heat_complement': heat_result,
            'missing_reverse': missing_result,
            'frequency_inverse': freq_result,
            'final_cold_scores': {str(k): v for k, v in sorted_cold},
            'final_recommendation': sorted(final_cold),
            'predictions': predictions
        }


def main():
    print("="*60)
    print("🎰 双色球避号系统预测器 v3")
    print("="*60)
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ 数据文件不存在: {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values('period', ascending=False)
    print(f"\n📥 加载 {len(df)} 条数据")
    print(f"   最新期号: {df['period'].iloc[0]}")
    
    predictor = AvoidPredictor(df)
    
    # 综合预测
    result = predictor.generate_final_prediction()
    
    # 回测
    backtest = predictor.backtest_heat_complement(test_periods=50)
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f'{OUTPUT_DIR}/avoid_predictor_v3.json', 'w', encoding='utf-8') as f:
        json.dump({
            'result': result,
            'backtest': backtest
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存到 {OUTPUT_DIR}/avoid_predictor_v3.json")


if __name__ == "__main__":
    main()
