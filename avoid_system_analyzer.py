#!/usr/bin/env python3
"""
双色球"避号系统"反推分析
==========================
通过分析历史数据，检验以下假设：
1. 上期开过的号码，下期是否刻意回避？
2. 高频热号是否在近期被刻意压低出现频率？
3. 销售热度（基于遗漏值/频率）是否与实际开奖负相关？

注意：此分析仅供娱乐研究，不构成任何投资建议
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json
from datetime import datetime

DATA_FILE = 'lottery_data.csv'
OUTPUT_DIR = 'output'


class AvoidSystemAnalyzer:
    """避号系统分析器"""
    
    def __init__(self, df):
        self.df = df.sort_values('period', ascending=False).reset_index(drop=True)
        self.red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        
    def analyze_repeat_rate(self):
        """
        分析【号码重复率】
        假设：如果存在"避号系统"，上期开过的号码下期出现率会偏低
        """
        print("\n" + "="*60)
        print("📊 分析1: 号码重复率检验")
        print("="*60)
        
        # 统计上期号码在下期出现的次数
        repeat_counts = []
        
        for i in range(len(self.df) - 1):
            current_row = self.df.iloc[i]
            prev_row = self.df.iloc[i + 1]
            
            current_reds = set([int(current_row[c]) for c in self.red_cols])
            prev_reds = set([int(prev_row[c]) for c in self.red_cols])
            
            # 上期6个号码中有几个在下期出现了
            repeat = len(current_reds & prev_reds)
            repeat_counts.append(repeat)
        
        # 理论期望值：如果完全随机，6个号码在33个中，下期每个号码出现概率是6/33
        # 期望重复个数 = 6 * (6/33) ≈ 1.09
        theoretical_expectation = 6 * (6/33)
        
        repeat_counter = Counter(repeat_counts)
        avg_repeat = np.mean(repeat_counts)
        
        print(f"\n  理论期望重复个数: {theoretical_expectation:.2f}")
        print(f"  实际平均重复个数: {avg_repeat:.2f}")
        print(f"\n  重复个数分布:")
        for k in sorted(repeat_counter.keys()):
            pct = repeat_counter[k] / len(repeat_counts) * 100
            bar = "█" * int(pct/2)
            print(f"    {k}个重复: {repeat_counter[k]:4d}次 ({pct:5.1f}%) {bar}")
        
        deviation = (avg_repeat - theoretical_expectation) / theoretical_expectation * 100
        
        if deviation < -5:
            print(f"\n  ⚠️ 发现异常: 重复率比理论值低 {abs(deviation):.1f}%")
            print(f"     这可能暗示存在某种'避号'机制")
        elif deviation > 5:
            print(f"\n  ℹ️ 重复率比理论值高 {deviation:.1f}%")
            print(f"     未发现明显避号迹象")
        else:
            print(f"\n  ✅ 重复率接近理论值，未发现明显异常")
        
        return {
            'theoretical': theoretical_expectation,
            'actual': avg_repeat,
            'deviation_pct': deviation,
            'distribution': dict(repeat_counter)
        }
    
    def analyze_hot_number_avoidance(self):
        """
        分析【热号回避】
        假设：高热度号码（近期频繁出现）是否在下一期被刻意压低出现率
        """
        print("\n" + "="*60)
        print("📊 分析2: 热号回避检验")
        print("="*60)
        
        # 定义热号：近10期出现3次以上的号码
        HOT_THRESHOLD = 3
        RECENT_WINDOW = 10
        
        hot_next_appearance = []  # 热号在下期出现的次数
        cold_next_appearance = []  # 冷号在下期出现的次数
        
        for i in range(len(self.df) - RECENT_WINDOW):
            # 计算近10期各号码频率
            recent_window = self.df.iloc[i+1:i+1+RECENT_WINDOW]
            freq = Counter()
            for _, row in recent_window.iterrows():
                for c in self.red_cols:
                    freq[int(row[c])] += 1
            
            hot_numbers = set(n for n, c in freq.items() if c >= HOT_THRESHOLD)
            cold_numbers = set(range(1, 34)) - hot_numbers
            
            # 下一期（当前期）的开奖号码
            current_row = self.df.iloc[i]
            current_reds = set([int(current_row[c]) for c in self.red_cols])
            
            # 热号出现次数
            hot_appeared = len(current_reds & hot_numbers)
            hot_next_appearance.append(hot_appeared)
            
            # 冷号出现次数
            cold_appeared = len(current_reds & cold_numbers)
            cold_next_appearance.append(cold_appeared)
        
        avg_hot = np.mean(hot_next_appearance)
        avg_cold = np.mean(cold_next_appearance)
        
        # 期望值
        # 如果号码出现概率完全随机，热号占比 = 热号数量/33
        # 但热号定义是"近10期出现3+次"，本身出现概率就偏高
        expected_hot = 6 * (len(hot_numbers) / 33) if 'hot_numbers' in dir() else 2.0
        
        print(f"\n  热号定义: 近{RECENT_WINDOW}期出现{HOT_THRESHOLD}次以上")
        print(f"  平均每期热号出现: {avg_hot:.2f}个")
        print(f"  平均每期冷号出现: {avg_cold:.2f}个")
        
        # 冷热号比例分析
        total_hot = sum(hot_next_appearance)
        total_cold = sum(cold_next_appearance)
        if total_hot + total_cold > 0:
            hot_ratio = total_hot / (total_hot + total_cold)
            print(f"  热号占比: {hot_ratio:.1%}")
            
            if hot_ratio < 0.4:
                print(f"\n  ⚠️ 热号出现比例偏低，可能存在'避热号'机制")
            elif hot_ratio > 0.6:
                print(f"\n  ℹ️ 热号出现比例正常，未发现明显避号迹象")
        
        return {
            'avg_hot_appearance': avg_hot,
            'avg_cold_appearance': avg_cold,
            'hot_ratio': hot_ratio if total_hot + total_cold > 0 else 0
        }
    
    def analyze_interval_pattern(self):
        """
        分析【间隔规律】
        假设：如果存在避号系统，号码的间隔分布会被人为操控
        """
        print("\n" + "="*60)
        print("📊 分析3: 号码间隔规律检验")
        print("="*60)
        
        # 计算每个号码的"出现间隔"
        intervals_by_number = defaultdict(list)
        
        last_appear = {}
        for _, row in self.df.iterrows():
            reds = [int(row[c]) for c in self.red_cols]
            period = int(row['period'])
            
            for r in reds:
                if r in last_appear:
                    interval = period - last_appear[r]
                    intervals_by_number[r].append(interval)
                last_appear[r] = period
        
        # 分析间隔分布
        all_intervals = []
        for num, intervals in intervals_by_number.items():
            all_intervals.extend(intervals)
        
        avg_interval = np.mean(all_intervals)
        std_interval = np.std(all_intervals)
        
        print(f"\n  全局平均间隔: {avg_interval:.2f}期")
        print(f"  间隔标准差: {std_interval:.2f}")
        
        # 检验间隔分布是否符合指数分布（随机期望）
        # 如果有人为操控，间隔分布会呈现异常
        expected_mean = 33 / 6  # 每个号码平均每5.5期出现一次
        
        print(f"\n  理论期望间隔: {expected_mean:.2f}期")
        
        deviation = (avg_interval - expected_mean) / expected_mean * 100
        
        if abs(deviation) < 10:
            print(f"  ✅ 间隔分布接近理论值")
        elif deviation > 10:
            print(f"  ⚠️ 平均间隔偏大，号码出现偏少")
        else:
            print(f"  ⚠️ 平均间隔偏小，号码出现偏多")
        
        # 分析最大间隔（遗漏值）
        max_intervals = {num: max(ints) for num, ints in intervals_by_number.items()}
        suspicious_high_missing = {n: v for n, v in max_intervals.items() if v > 40}
        
        if suspicious_high_missing:
            print(f"\n  发现异常高遗漏号码: {suspicious_high_missing}")
        
        return {
            'avg_interval': avg_interval,
            'std_interval': std_interval,
            'expected_interval': expected_mean,
            'deviation_pct': deviation,
            'max_intervals': max_intervals
        }
    
    def analyze_consecutive_avoidance(self):
        """
        分析【连号回避】
        检验开奖号码中连号的比例是否异常
        """
        print("\n" + "="*60)
        print("📊 分析4: 连号回避检验")
        print("="*60)
        
        consecutive_counts = []
        
        for _, row in self.df.iterrows():
            reds = sorted([int(row[c]) for c in self.red_cols])
            
            # 统计连号数量（差值为1的对数）
            cons = 0
            for i in range(len(reds) - 1):
                if reds[i+1] - reds[i] == 1:
                    cons += 1
            consecutive_counts.append(cons)
        
        avg_consecutive = np.mean(consecutive_counts)
        
        # 理论计算：6个号码从33个中选，连号概率约为15%
        # 期望连号对数约为 6 * (5/32) * (6/33) ≈ 0.17
        theoretical_consecutive = 6 * 5 / 32 * 6 / 33
        
        print(f"\n  平均连号对数: {avg_consecutive:.3f}")
        print(f"  理论期望连号: {theoretical_consecutive:.3f}")
        
        # 统计分布
        cons_counter = Counter(consecutive_counts)
        print(f"\n  连号分布:")
        for k in sorted(cons_counter.keys()):
            pct = cons_counter[k] / len(consecutive_counts) * 100
            bar = "█" * int(pct*2)
            print(f"    {k}对连号: {cons_counter[k]:4d}次 ({pct:5.1f}%) {bar}")
        
        deviation = (avg_consecutive - theoretical_consecutive) / theoretical_consecutive * 100
        
        if deviation < -20:
            print(f"\n  ⚠️ 连号比例偏低 {abs(deviation):.1f}%，可能存在'避连号'机制")
        else:
            print(f"\n  ✅ 连号分布正常")
        
        return {
            'actual_avg': avg_consecutive,
            'theoretical': theoretical_consecutive,
            'deviation_pct': deviation,
            'distribution': dict(cons_counter)
        }
    
    def analyze_region_balance(self):
        """
        分析【区域平衡】
        假设：为了控制销售额，开奖号码可能在区域间保持平衡
        """
        print("\n" + "="*60)
        print("📊 分析5: 区域平衡检验")
        print("="*60)
        
        # 将33个红球分成3个区：1-11, 12-22, 23-33
        region_counts = {'low': [], 'mid': [], 'high': []}
        
        for _, row in self.df.iterrows():
            low = mid = high = 0
            for c in self.red_cols:
                v = int(row[c])
                if v <= 11:
                    low += 1
                elif v <= 22:
                    mid += 1
                else:
                    high += 1
            region_counts['low'].append(low)
            region_counts['mid'].append(mid)
            region_counts['high'].append(high)
        
        print(f"\n  各区平均号码数:")
        for region, counts in region_counts.items():
            avg = np.mean(counts)
            std = np.std(counts)
            print(f"    {region}: {avg:.2f} ± {std:.2f}")
        
        # 理论期望: 每区 6/3 = 2个
        theoretical = 2.0
        
        imbalance_detected = False
        for region, counts in region_counts.items():
            avg = np.mean(counts)
            if abs(avg - theoretical) > 0.5:
                imbalance_detected = True
                print(f"\n  ⚠️ {region}区分布异常，期望2个，实际{avg:.2f}个")
        
        if not imbalance_detected:
            print(f"\n  ✅ 各区分布均衡，符合随机期望")
        
        return {
            'region_avg': {r: np.mean(c) for r, c in region_counts.items()},
            'region_std': {r: np.std(c) for r, c in region_counts.items()}
        }
    
    def generate_anti_avoidance_prediction(self):
        """
        基于"避号系统"假设的反向预测
        如果系统倾向于避开热号，那么我们应该：
        1. 优先选择上期出现过的号码（被"刻意回避"的）
        2. 避开近期过于热门的号码
        """
        print("\n" + "="*60)
        print("🎯 反向预测: 基于避号假设的号码推荐")
        print("="*60)
        
        # 上一期号码
        last_row = self.df.iloc[0]
        last_reds = set([int(last_row[c]) for c in self.red_cols])
        
        # 近10期热号
        recent = self.df.iloc[:10]
        freq = Counter()
        for _, row in recent.iterrows():
            for c in self.red_cols:
                freq[int(row[c])] += 1
        
        hot_numbers = set(n for n, c in freq.items() if c >= 4)
        warm_numbers = set(n for n, c in freq.items() if c >= 2)
        cold_numbers = set(range(1, 34)) - warm_numbers
        
        print(f"\n  上一期号码: {sorted(last_reds)}")
        print(f"  近10期热号(4+次): {sorted(hot_numbers)}")
        print(f"  近10期温号(2-3次): {sorted(warm_numbers - hot_numbers)}")
        print(f"  近10期冷号(0-1次): {sorted(cold_numbers)}")
        
        # 预测策略
        predictions = []
        
        # 策略1: 上一期号码的"反弹"
        # 假设系统会避免连续出现，但长期来看有回补需求
        rebound = last_reds & cold_numbers  # 上一期出现且目前是冷号的
        
        print(f"\n  📌 策略1 - 冷号反弹: {sorted(rebound)}")
        
        # 策略2: 避开过热号
        safe_numbers = set(range(1, 34)) - hot_numbers
        print(f"  📌 策略2 - 避开热号: {sorted(safe_numbers)[:12]}")
        
        # 策略3: 间隔理论 - 选择遗漏值较大的号码
        last_appear = {}
        for _, row in self.df.iterrows():
            for c in self.red_cols:
                n = int(row[c])
                if n not in last_appear:
                    last_appear[n] = int(row['period'])
        
        max_period = max(int(row['period']) for _, row in self.df.iterrows())
        missing_values = {n: max_period - last_appear.get(n, 0) for n in range(1, 34)}
        
        high_missing = sorted(missing_values.items(), key=lambda x: x[1], reverse=True)[:12]
        print(f"  📌 策略3 - 高遗漏值: {[(n,m) for n,m in high_missing[:6]]}")
        
        # 综合推荐
        print("\n  🏆 综合推荐号码组合 (3组):")
        
        for i in range(3):
            group = []
            
            # 从反弹号码选1-2个
            rebound_list = sorted(rebound)
            if rebound_list:
                group.extend(rebound_list[:2])
            
            # 从高遗漏选2-3个
            for n, m in high_missing:
                if n not in group and len(group) < 6:
                    group.append(n)
            
            # 补齐到6个（避开热号）
            for n in range(1, 34):
                if n not in group and len(group) < 6 and n not in hot_numbers:
                    group.append(n)
                    if len(group) >= 6:
                        break
            
            print(f"    组合{i+1}: {sorted(group[:6])}")
            predictions.append(sorted(group[:6]))
        
        return predictions
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("\n" + "#"*60)
        print("#  双色球'避号系统'反推分析")
        print("#"*60)
        print(f"\n📥 分析数据: {len(self.df)} 期")
        print(f"   范围: {self.df['period'].min()} ~ {self.df['period'].max()}")
        
        results = {
            'data_range': f"{self.df['period'].min()} ~ {self.df['period'].max()}",
            'total_records': len(self.df),
            'repeat_rate': self.analyze_repeat_rate(),
            'hot_avoidance': self.analyze_hot_number_avoidance(),
            'interval_pattern': self.analyze_interval_pattern(),
            'consecutive_avoidance': self.analyze_consecutive_avoidance(),
            'region_balance': self.analyze_region_balance(),
        }
        
        predictions = self.generate_anti_avoidance_prediction()
        results['predictions'] = predictions
        
        return results


def main():
    print("="*60)
    print("🎯 双色球'避号系统'反推分析")
    print("="*60)
    
    import os
    if not os.path.exists(DATA_FILE):
        print(f"❌ 数据文件不存在: {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f"\n📥 加载 {len(df)} 条数据")
    
    analyzer = AvoidSystemAnalyzer(df)
    results = analyzer.run_full_analysis()
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f'{OUTPUT_DIR}/avoid_system_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 分析完成，结果已保存到 {OUTPUT_DIR}/avoid_system_analysis.json")


if __name__ == "__main__":
    main()
