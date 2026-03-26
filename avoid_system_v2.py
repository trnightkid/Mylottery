#!/usr/bin/env python3
"""
双色球避号系统模拟器 v2
=========================
核心思路：
彩票机构在开奖前收集彩民购买数据，刻意避开购买热度高的号码，
选择冷门号码作为开奖结果，从而最小化赔付。

本模拟器通过历史数据反推这个机制，并预测下一期号码。

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


class AvoidSystemSimulator:
    """
    避号系统模拟器
    
    核心假设：
    1. 彩票机构知道所有已售号码的热度分布
    2. 他们选择开奖号码时会刻意避开最热的号码
    3. 开奖号码 ≈ 购买热度最低的号码集合
    """
    
    def __init__(self, df):
        self.df = df.sort_values('period', ascending=False).reset_index(drop=True)
        self.red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
    
    def infer_buyer_heatmap(self, period_data):
        """
        从开奖号码反推"彩民购买热度分布"
        
        假设：开奖号码是"被回避"的结果
        - 如果号码X出现了，说明X在购买时是"冷门"
        - 如果号码Y没出现，说明Y可能是"热门但被回避"
        
        但这个反推是不确定的，我们需要用统计方法估算最可能的购买分布
        """
        reds = set([int(period_data[c]) for c in self.red_cols])
        all_nums = set(range(1, 34))
        not_selected = all_nums - reds  # 33个号中没被选中的27个
        
        # 理论分析：
        # 假设购买热度分布大致是正态的（大部分号码热度中等，少数极热/极冷）
        # 如果机构回避热号，那么开奖号码更可能来自"中等偏冷"的区域
        
        # 基于历史共现关系推断热度
        # 如果号码A和B经常一起出现，说明它们可能被同一类彩民购买（热区联动）
        cooc = self._build_recent_cooccurrence(50)  # 近50期的共现矩阵
        
        # 对于开奖号码集合，推断每个位置的"候选热度"
        heat_scores = {}
        for num in all_nums:
            # 热度分：与开奖号码共现越多的号码，可能越热（因为热号往往聚集）
            related_heat = sum(cooc.get((min(num, other), max(num, other)), 0) 
                              for other in reds if other != num)
            # 如果这个号码本身是开奖号码，说明它是"冷门中的幸运"
            if num in reds:
                heat_scores[num] = related_heat * 0.8  # 降低，因为它是"被选中"的冷号
            else:
                heat_scores[num] = related_heat * 1.2  # 升高，可能是因为太热被回避
        
        return heat_scores
    
    def _build_recent_cooccurrence(self, window):
        """构建近N期的共现矩阵"""
        cooc = defaultdict(int)
        for i in range(min(window, len(self.df) - 1)):
            row = self.df.iloc[i]
            reds = sorted([int(row[c]) for c in self.red_cols])
            for a, b in combinations(reds, 2):
                cooc[(min(a, b), max(a, b))] += 1
        return dict(cooc)
    
    def predict_next_with_avoid_logic(self, lookback=5):
        """
        基于避号逻辑预测下一期
        
        思路：
        1. 分析近N期开奖号码
        2. 识别哪些号码是"被回避"的（出现频率相对低的）
        3. 预测下一期会继续回避什么号码
        4. 从"被回避"的号码中选择开奖候选
        
        参数:
        lookback: 分析近几期的模式
        """
        print("\n" + "="*60)
        print("🔮 避号系统预测")
        print("="*60)
        
        # 获取近lookback期的数据
        recent = self.df.iloc[:lookback]
        
        # 分析每个号码的出现频率（在这几期中）
        all_reds = []
        for _, row in recent.iterrows():
            for c in self.red_cols:
                all_reds.append(int(row[c]))
        
        freq = Counter(all_reds)
        total_periods = len(recent)
        
        print(f"\n📊 近{total_periods}期号码频率分析:")
        
        # 按频率分组
        freq_groups = defaultdict(list)
        for num in range(1, 34):
            count = freq[num]
            freq_groups[count].append(num)
        
        # 找出出现次数最多和最少的号码
        min_freq = min(freq.values())
        max_freq = max(freq.values())
        
        cold_numbers = set()   # 最冷门号码（出现次数最少）
        hot_numbers = set()    # 最热门号码（出现次数最多）
        normal_numbers = set() # 正常号码
        
        for num, count in freq.items():
            if count <= min_freq + 0.5:
                cold_numbers.add(num)
            elif count >= max_freq - 0.5:
                hot_numbers.add(num)
            else:
                normal_numbers.add(num)
        
        print(f"\n  🔴 冷门号码(出现{min_freq}次): {sorted(cold_numbers)}")
        print(f"  🟡 正常号码: {sorted(normal_numbers)}")
        print(f"  🟠 热门号码(出现{max_freq}次): {sorted(hot_numbers)}")
        
        # 预测逻辑
        # 如果避号系统存在，下一期应该：
        # 1. 继续回避热门号码
        # 2. 倾向于选择冷门或正常号码
        # 3. 但要注意"冷号回补"——太冷的号码偶尔也会出现
        
        print(f"\n🧮 避号系统预测逻辑:")
        print(f"  • 假设系统会回避: {sorted(hot_numbers)}")
        print(f"  • 候选开奖区: {sorted(cold_numbers | normal_numbers)}")
        
        # 生成预测组合
        predictions = []
        
        # 策略1: 纯冷门策略 - 主要选最冷的号码
        for _ in range(3):
            group = []
            # 从最冷区选2-3个
            cold_list = sorted(cold_numbers)
            np.random.shuffle(cold_list)
            group.extend(cold_list[:3])
            
            # 从正常区选2-3个
            normal_list = sorted(normal_numbers)
            np.random.shuffle(normal_list)
            group.extend(normal_list[:3])
            
            while len(group) < 6:
                group.append(np.random.choice(sorted(cold_numbers | normal_numbers)))
            
            predictions.append(sorted(set(group[:6])))
        
        # 策略2: 冷热均衡策略
        for _ in range(2):
            group = []
            # 1个最冷号
            group.append(np.random.choice(sorted(cold_numbers)))
            # 3个正常号
            group.extend(np.random.choice(sorted(normal_numbers), size=3, replace=False))
            # 2个"回补"号（稍微热门但不是最热）
            hot_minus_top = hot_numbers - set([max(freq, key=freq.get)])
            if hot_minus_top:
                group.extend(np.random.choice(sorted(hot_minus_top), size=2, replace=False))
            else:
                group.extend(np.random.choice(sorted(normal_numbers), size=2, replace=False))
            
            predictions.append(sorted(set(group[:6])))
        
        return {
            'cold_numbers': sorted(cold_numbers),
            'normal_numbers': sorted(normal_numbers),
            'hot_numbers': sorted(hot_numbers),
            'predictions': predictions
        }
    
    def backtest_avoid_logic(self, test_periods=100):
        """
        回测验证：避号逻辑在过去N期中的准确率
        
        思路：
        1. 用前N期的数据预测第N+1期的结果
        2. 验证预测的准确率
        """
        print("\n" + "="*60)
        print(f"📈 回测验证 (最近{test_periods}期)")
        print("="*60)
        
        # 对每期进行预测，然后和实际开奖对比
        hit_scores = []
        
        for i in range(test_periods):
            if i + 1 >= len(self.df):
                break
            
            # 用前i期预测第i期
            train_df = self.df.iloc[i+1:min(i+1+10, len(self.df))]  # 近10期
            actual_df = self.df.iloc[i]  # 当前期
            
            # 统计训练集中的频率
            all_reds = []
            for _, row in train_df.iterrows():
                for c in self.red_cols:
                    all_reds.append(int(row[c]))
            
            freq = Counter(all_reds)
            if not freq:
                continue
            
            min_freq = min(freq.values())
            max_freq = max(freq.values())
            
            # 找出预测的冷门号
            predicted_cold = set()
            predicted_hot = set()
            for num, count in freq.items():
                if count <= min_freq + 0.5:
                    predicted_cold.add(num)
                elif count >= max_freq - 0.5:
                    predicted_hot.add(num)
            
            # 实际开奖号码
            actual_reds = set([int(actual_df[c]) for c in self.red_cols])
            
            # 计算命中率
            cold_hit = len(predicted_cold & actual_reds)
            hot_avoid = len([n for n in actual_reds if n not in predicted_hot])
            
            hit_scores.append({
                'period': actual_df['period'],
                'cold_hit': cold_hit,
                'hot_avoid': hot_avoid,
                'actual': sorted(actual_reds)
            })
        
        # 统计结果
        cold_hits = [h['cold_hit'] for h in hit_scores]
        hot_avoid_rate = [h['hot_avoid'] / 6 * 100 for h in hit_scores]
        
        print(f"\n📊 回测结果:")
        print(f"  平均每期冷门号码命中: {np.mean(cold_hits):.2f}个")
        print(f"  热门号码回避率: {np.mean(hot_avoid_rate):.1f}%")
        
        # 理想情况：如果完全按"避热号"策略
        # 冷门号码应该命中更多（因为开奖号来自冷门）
        # 热门号码应该被回避
        
        cold_3plus = sum(1 for h in cold_hits if h >= 3)
        print(f"  冷门号命中3+次占比: {cold_3plus/len(cold_hits)*100:.1f}%")
        
        return {
            'avg_cold_hit': np.mean(cold_hits),
            'avg_hot_avoid_rate': np.mean(hot_avoid_rate),
            'cold_3plus_rate': cold_3plus / len(cold_hits) * 100 if cold_hits else 0
        }
    
    def predict_specific_period(self, target_period):
        """
        针对特定期号进行预测
        
        例如: 已知第N期开奖号，预测第N+1期
        """
        print("\n" + "="*60)
        print(f"🎯 针对第{target_period}期之后的预测")
        print("="*60)
        
        # 找到目标期号的数据
        target_rows = self.df[self.df['period'] == str(target_period)]
        
        if len(target_rows) == 0:
            # 尝试数值匹配
            target_rows = self.df[self.df['period'] == target_period]
        
        if len(target_rows) == 0:
            print(f"❌ 未找到期号 {target_period} 的数据")
            return None
        
        target_row = target_rows.iloc[0]
        target_reds = [int(target_row[c]) for c in self.red_cols]
        
        print(f"\n📌 第{target_period}期开奖号码: {target_reds}")
        
        # 分析这期号码的特征
        # 如果这些是"被避号系统选中"的冷门号，
        # 那么下一期可能会"回补"一些热号，或继续选冷号
        
        # 获取这期之后的历史模式（如果有）
        # 但我们是倒序的，所以要找这期之前的数据
        
        idx = self.df[self.df['period'] == target_period].index[0]
        if idx + 1 < len(self.df):
            next_row = self.df.iloc[idx + 1]
            next_reds = [int(next_row[c]) for c in self.red_cols]
            print(f"📌 第{target_row['period']}期(下一期)实际号码: {next_reds}")
            
            # 计算关联
            common = set(target_reds) & set(next_reds)
            print(f"   与预测期重复号码: {len(common)}个 → {sorted(common)}")
        
        # 基于历史模式预测
        result = self.predict_next_with_avoid_logic(lookback=10)
        
        return result


def main():
    print("="*60)
    print("🎰 双色球避号系统模拟器 v2")
    print("="*60)
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ 数据文件不存在: {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values('period', ascending=False)
    print(f"\n📥 加载 {len(df)} 条数据")
    print(f"   最新期号: {df['period'].iloc[0]}")
    
    simulator = AvoidSystemSimulator(df)
    
    # 1. 直接预测下一期
    prediction = simulator.predict_next_with_avoid_logic(lookback=5)
    
    print("\n" + "="*60)
    print("🏆 最终预测结果")
    print("="*60)
    print(f"\n  基于近5期分析:")
    print(f"  • 冷门号码池: {prediction['cold_numbers']}")
    print(f"  • 热门回避: {prediction['hot_numbers']}")
    
    print(f"\n  🎱 推荐号码组合:")
    for i, reds in enumerate(prediction['predictions'], 1):
        print(f"    组合{i}: {reds}")
    
    # 2. 回测验证
    backtest = simulator.backtest_avoid_logic(test_periods=100)
    
    # 3. 针对最新一期预测
    latest_period = int(df['period'].iloc[0])
    simulator.predict_specific_period(latest_period)
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        'prediction': prediction,
        'backtest': backtest,
        'latest_period': latest_period
    }
    
    with open(f'{OUTPUT_DIR}/avoid_system_v2_prediction.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存到 {OUTPUT_DIR}/avoid_system_v2_prediction.json")


if __name__ == "__main__":
    main()
