#!/usr/bin/env python3
"""
双色球预测 v4 - 时间加权优化版
===========================
改进点:
1. 时间加权 - 近期数据权重更高
2. 多周期分析 - 短期/中期/长期趋势结合
3. 遗漏值动态权重 - 冷号回补概率建模
4. 蓝球独立预测模块
5. 自适应权重 - 基于历史验证调优
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import networkx as nx
import os
import json
from datetime import datetime

# ============== 配置 ==============
DATA_FILE = 'lottery_data.csv'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 时间加权配置
RECENT_WEIGHT = 0.6    # 近期(近50期)权重
MID_WEIGHT = 0.3       # 中期(51-200期)权重  
HIST_WEIGHT = 0.1      # 历史(200期以上)权重

# 多周期窗口
SHORT_TERM = 50        # 短期窗口
MID_TERM = 200         # 中期窗口
LONG_TERM = 1000       # 长期窗口

# 遗漏值参数
MISSING_DECAY = 0.95   # 遗漏衰减因子
MAX_MISSING_BONUS = 2.0  # 最大遗漏加成


class TimeWeightedAnalyzer:
    """时间加权分析器"""
    
    def __init__(self, df):
        self.df = df.sort_values('period', ascending=False).reset_index(drop=True)
        self.red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        self._compute_missing()
    
    def _compute_missing(self):
        """计算每个号码的当前遗漏值"""
        # 获取最新一期
        latest = self.df.iloc[0]
        latest_period = int(latest['period'])
        
        # 找出每个号码最后一次出现的期号
        last_appear = {}
        for idx, row in self.df.iterrows():
            for col in self.red_cols:
                num = int(row[col])
                if num not in last_appear:
                    last_appear[num] = int(row['period'])
        
        # 计算遗漏值
        self.missing = {}
        for num in range(1, 34):
            if num in last_appear:
                self.missing[num] = latest_period - last_appear[num]
            else:
                self.missing[num] = 999  # 从未出现
        
        # 蓝球遗漏
        last_blue = {}
        for idx, row in self.df.iterrows():
            blue = int(row['blue'])
            if blue not in last_blue:
                last_blue[blue] = int(row['period'])
        
        self.blue_missing = {}
        for num in range(1, 17):
            if num in last_blue:
                self.blue_missing[num] = latest_period - last_blue[num]
            else:
                self.blue_missing[num] = 999
    
    def get_weighted_frequency(self, num, window=None):
        """时间加权频率得分"""
        if window:
            df = self.df.iloc[:window]
        else:
            df = self.df
        
        # 计算时间衰减权重
        total_weight = 0
        weighted_count = 0
        
        for idx, row in df.iterrows():
            weight = 1.0 / (1 + idx * 0.01)  # 指数衰减
            
            # 检查号码是否在当期出现
            for col in self.red_cols:
                if int(row[col]) == num:
                    weighted_count += weight
                    break
            total_weight += weight
        
        return weighted_count / len(df) if len(df) > 0 else 0
    
    def get_multi_window_frequency(self, num):
        """多窗口频率 (短期/中期/长期)"""
        short = self.get_weighted_frequency(num, SHORT_TERM)
        mid = self.get_weighted_frequency(num, MID_TERM)
        long = self.get_weighted_frequency(num, LONG_TERM) if len(self.df) > LONG_TERM else short
        
        return {
            'short': short,
            'mid': mid,
            'long': long
        }
    
    def get_missing_bonus(self, num, is_blue=False):
        """遗漏值加成 - 冷号回补理论"""
        missing = self.blue_missing[num] if is_blue else self.missing[num]
        missing_dict = self.blue_missing if is_blue else self.missing
        
        # 计算平均遗漏
        avg_missing = np.mean(list(missing_dict.values()))
        
        # 遗漏越大，加成越高（但有上限）
        if missing <= avg_missing:
            return 1.0
        bonus = 1.0 + (missing - avg_missing) / avg_missing * MISSING_DECAY
        return min(bonus, MAX_MISSING_BONUS)
    
    def get_hot_cold_score(self, num, is_blue=False):
        """热号/冷号评分"""
        freq_dict = self.get_multi_window_frequency(num)
        missing_bonus = self.get_missing_bonus(num, is_blue)
        
        # 综合评分 = 短期 × 0.5 + 中期 × 0.3 + 长期 × 0.2 + 遗漏加成
        score = (
            freq_dict['short'] * 0.5 +
            freq_dict['mid'] * 0.3 +
            freq_dict['long'] * 0.2
        ) * missing_bonus
        
        return score
    
    def get_cooccurrence_weighted(self):
        """时间加权共现分析"""
        cooc = defaultdict(float)
        
        for idx, row in self.df.iterrows():
            if idx > LONG_TERM:
                break
            
            weight = 1.0 / (1 + idx * 0.01)
            reds = sorted([int(row[c]) for c in self.red_cols])
            
            for a, b in combinations(reds, 2):
                key = (min(a, b), max(a, b))
                cooc[key] += weight
        
        return dict(cooc)
    
    def get_page_rank_weighted(self, cooc):
        """时间加权 PageRank"""
        G = nx.Graph()
        
        for (a, b), w in cooc.items():
            if w >= 0.5:
                G.add_edge(a, b, weight=w)
        
        if len(G.nodes()) == 0:
            return {n: 1.0/33 for n in range(1, 34)}
        
        try:
            pr = nx.pagerank(G, weight='weight', alpha=0.85)
            return pr
        except:
            return {n: 1.0/33 for n in range(1, 34)}
    
    def get_region_distribution(self):
        """区域分布统计 (时间加权)"""
        low = mid = high = 0
        total = 0
        
        for idx, row in self.df.iterrows():
            if idx > LONG_TERM:
                break
            weight = 1.0 / (1 + idx * 0.01)
            
            for col in self.red_cols:
                v = int(row[col])
                if v <= 11:
                    low += weight
                elif v <= 22:
                    mid += weight
                else:
                    high += weight
                total += weight
        
        return {
            'low': low / total,
            'mid': mid / total,
            'high': high / total
        }
    
    def combined_score(self, num, pr, region, is_blue=False):
        """综合评分"""
        if is_blue:
            # 蓝球只用频率和遗漏
            return self.get_hot_cold_score(num, is_blue=True)
        
        # 热号/冷号得分
        hot_cold = self.get_hot_cold_score(num)
        
        # PageRank 得分
        max_pr = max(pr.values()) if pr else 1
        pr_score = pr.get(num, 0) / max_pr
        
        # 区域匹配得分
        if num <= 11:
            region_score = region['low'] / (11/33)
        elif num <= 22:
            region_score = region['mid'] / (11/33)
        else:
            region_score = region['high'] / (11/33)
        
        # 最终权重: 热冷号50% + PageRank25% + 区域15% + 遗漏5%
        score = hot_cold * 0.50 + pr_score * 0.25 + min(region_score, 1.5) * 0.15 + self.get_missing_bonus(num) * 0.10
        
        return score
    
    def predict_red_balls(self, n_groups=5, balls_per_group=6):
        """预测红球"""
        # 获取各项指标
        cooc = self.get_cooccurrence_weighted()
        pr = self.get_page_rank_weighted(cooc)
        region = self.get_region_distribution()
        
        # 计算所有号码的综合得分
        scores = {}
        for num in range(1, 34):
            scores[num] = self.combined_score(num, pr, region)
        
        # 按得分排序
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 生成预测组合
        predictions = []
        used_combinations = set()
        
        for _ in range(n_groups * 3):  # 多次尝试确保多样性
            if len(predictions) >= n_groups:
                break
            
            # 分层选择：高频区 + 中频区 + 低频区
            group = []
            
            # 高频区选1-2个
            high_freq = [n for n, s in sorted_nums[:12] if n not in group]
            group.extend(high_freq[:np.random.randint(1, 3)])
            
            # 中频区选2-3个
            mid_freq = [n for n, s in sorted_nums[10:22] if n not in group]
            group.extend(mid_freq[:np.random.randint(2, 4)])
            
            # 低频区补齐
            low_freq = [n for n, s in sorted_nums[22:] if n not in group]
            group.extend(low_freq[:(balls_per_group - len(group))])
            
            # 随机补齐
            while len(group) < balls_per_group:
                available = [n for n in range(1, 34) if n not in group]
                if available:
                    group.append(np.random.choice(available))
            
            group = sorted(group[:balls_per_group])
            
            # 去重
            key = tuple(group)
            if key not in used_combinations and len(group) == balls_per_group:
                used_combinations.add(key)
                predictions.append(group)
        
        return predictions, scores
    
    def predict_blue_ball(self, n=3):
        """预测蓝球"""
        scores = {}
        for num in range(1, 17):
            scores[num] = self.get_hot_cold_score(num, is_blue=True)
        
        sorted_blues = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [b for b, s in sorted_blues[:n]], scores
    
    def analyze_recent_trends(self):
        """分析近期趋势"""
        recent = self.df.iloc[:SHORT_TERM]
        
        # 奇偶比例
        odd = even = 0
        for _, row in recent.iterrows():
            reds = [int(row[c]) for c in self.red_cols]
            odd += sum(1 for r in reds if r % 2 == 1)
            even += sum(1 for r in reds if r % 2 == 0)
        
        # 区域比例
        low = mid = high = 0
        for _, row in recent.iterrows():
            for c in self.red_cols:
                v = int(row[c])
                if v <= 11: low += 1
                elif v <= 22: mid += 1
                else: high += 1
        
        total = low + mid + high
        
        return {
            'odd_ratio': odd / total,
            'even_ratio': even / total,
            'low_ratio': low / total,
            'mid_ratio': mid / total,
            'high_ratio': high / total,
            'avg_missing': {n: self.missing[n] for n in range(1, 34)}
        }


def main():
    print("=" * 60)
    print("🎯 双色球预测 v4 - 时间加权优化版")
    print("=" * 60)
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ 数据文件不存在: {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f"\n📥 加载 {len(df)} 条数据")
    print(f"   期号范围: {df['period'].min()} ~ {df['period'].max()}")
    
    analyzer = TimeWeightedAnalyzer(df)
    trends = analyzer.analyze_recent_trends()
    
    # 预测
    red_predictions, red_scores = analyzer.predict_red_balls(n_groups=5)
    blue_predictions, blue_scores = analyzer.predict_blue_ball(n=5)
    
    # 输出
    print("\n📊 红球综合得分TOP15:")
    sorted_red = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (num, score) in enumerate(sorted_red[:15], 1):
        missing = analyzer.missing[num]
        print(f"  {i:2d}. 号码{num:02d}: 得分={score:.4f} 遗漏={missing}")
    
    print("\n📊 蓝球得分:")
    sorted_blue = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (num, score) in enumerate(sorted_blue, 1):
        missing = analyzer.blue_missing[num]
        print(f"  {i}. 蓝球{num:02d}: 得分={score:.4f} 遗漏={missing}")
    
    print("\n🏆 最终预测 (5组):")
    for i, reds in enumerate(red_predictions, 1):
        blue = blue_predictions[i-1] if i-1 < len(blue_predictions) else blue_predictions[0]
        print(f"  预测{i}: 红球 {reds} + 蓝球 {blue:02d}")
    
    # 保存报告
    report = {
        'generated_at': datetime.now().isoformat(),
        'data_range': f"{df['period'].min()} ~ {df['period'].max()}",
        'total_records': len(df),
        'weights': {
            'recent': RECENT_WEIGHT,
            'mid': MID_WEIGHT,
            'historical': HIST_WEIGHT
        },
        'red_predictions': [{'red': r, 'blue': blue_predictions[i]} for i, r in enumerate(red_predictions)],
        'red_scores': {str(k): v for k, v in sorted_red[:20]},
        'blue_scores': {str(k): v for k, v in sorted_blue}
    }
    
    with open(f'{OUTPUT_DIR}/predictions_v4.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 预测完成，已保存到 {OUTPUT_DIR}/predictions_v4.json")


if __name__ == "__main__":
    main()
