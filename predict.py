#!/usr/bin/env python3
"""
双色球预测 v3 - 图算法增强版
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import networkx as nx
import os

DATA_FILE = '/home/clawd/Mylottery/lottery_data.csv'
OUTPUT_DIR = '/home/clawd/Mylottery/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class LotteryAnalyzer:
    def __init__(self, df):
        self.df = df
        self.red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        
    def get_frequency(self):
        all_reds = []
        for col in self.red_cols:
            all_reds.extend(self.df[col].tolist())
        return Counter(all_reds)
    
    def get_cooccurrence(self):
        cooc = defaultdict(int)
        for _, row in self.df.iterrows():
            reds = sorted([row[c] for c in self.red_cols])
            for a, b in combinations(reds, 2):
                cooc[(min(a,b), max(a,b))] += 1
        return cooc
    
    def get_page_rank(self, cooc):
        G = nx.Graph()
        for (a, b), w in cooc.items():
            if w >= 3:
                G.add_edge(a, b, weight=w)
        if len(G.nodes()) == 0:
            return {n: 1/33 for n in range(1, 34)}
        try:
            pr = nx.pagerank(G, weight='weight', alpha=0.85)
            return pr
        except:
            return {n: 1/33 for n in range(1, 34)}
    
    def get_regions(self):
        low = mid = high = 0
        for col in self.red_cols:
            for v in self.df[col]:
                if v <= 11: low += 1
                elif v <= 22: mid += 1
                else: high += 1
        total = low + mid + high
        return {'low': low/total, 'mid': mid/total, 'high': high/total}
    
    def combined_score(self, num, freq, pr, region):
        max_freq = max(freq.values()) if freq else 1
        freq_score = freq.get(num, 0) / max_freq
        
        max_pr = max(pr.values()) if pr else 1
        pr_score = pr.get(num, 0) / max_pr
        
        if num <= 11:
            region_score = region['low'] / (11/33)
        elif num <= 22:
            region_score = region['mid'] / (11/33)
        else:
            region_score = region['high'] / (11/33)
        
        return freq_score * 0.5 + pr_score * 0.3 + min(region_score, 1.5) * 0.2
    
    def predict(self, n=5):
        freq = self.get_frequency()
        cooc = self.get_cooccurrence()
        pr = self.get_page_rank(cooc)
        region = self.get_regions()
        blue_freq = Counter(self.df['blue'].tolist())
        
        scores = {n: self.combined_score(n, freq, pr, region) for n in range(1, 34)}
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        predictions = []
        for _ in range(n):
            selected = []
            selected.extend([n for n, _ in sorted_nums[:12] if len(selected) < 2])
            selected.extend([n for n, _ in sorted_nums[10:22] if n not in selected and len(selected) < 4])
            selected.extend([n for n, _ in sorted_nums[20:] if n not in selected and len(selected) < 6])
            remaining = [n for n, _ in sorted_nums if n not in selected]
            np.random.shuffle(remaining)
            while len(selected) < 6:
                for r in remaining:
                    if r not in selected:
                        selected.append(r)
                        break
            predictions.append(sorted(selected[:6]))
        
        blue_sorted = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)
        return predictions, blue_sorted[:5], scores, freq, pr, blue_freq


def main():
    print("=" * 60)
    print("🎯 双色球预测 v3 - 图算法增强版")
    print("=" * 60)
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ 数据文件不存在: {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values('period', ascending=False)
    print(f"\n📥 加载 {len(df)} 条数据")
    print(f"   期号范围: {df['period'].min()} ~ {df['period'].max()}")
    
    analyzer = LotteryAnalyzer(df)
    predictions, blue_preds, scores, freq, pr, blue_freq = analyzer.predict(n=5)
    
    print("\n📊 红球综合得分TOP15:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (num, score) in enumerate(sorted_scores[:15], 1):
        print(f"  {i:2d}. 号码{num:02d}: 综合={score:.4f} 频次={freq.get(num,0):3d} PR={pr.get(num,0):.4f}")
    
    print("\n📊 蓝球频率TOP5:")
    for i, (num, count) in enumerate(blue_freq.most_common(5), 1):
        print(f"  {i}. 号码{num:02d}: {count}次")
    
    print("\n🏆 最终预测 (5组):")
    for i, reds in enumerate(predictions, 1):
        blue = blue_preds[i-1][0] if i <= len(blue_preds) else blue_freq.most_common(1)[0][0]
        print(f"  预测{i}: 红球 {reds} + 蓝球 {blue:02d}")
    
    with open(f'{OUTPUT_DIR}/latest_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n双色球预测分析报告\n" + "=" * 60 + "\n\n红球综合得分TOP15:\n")
        for i, (num, score) in enumerate(sorted_scores[:15], 1):
            f.write(f"  {i:2d}. 号码{num:02d}: 综合={score:.4f}\n")
        f.write("\n蓝球频率TOP5:\n")
        for i, (num, count) in enumerate(blue_freq.most_common(5), 1):
            f.write(f"  {i}. 号码{num:02d}: {count}次\n")
        f.write("\n最终预测:\n")
        for i, reds in enumerate(predictions, 1):
            blue = blue_preds[i-1][0] if i <= len(blue_preds) else blue_freq.most_common(1)[0][0]
            f.write(f"  预测{i}: 红球 {reds} + 蓝球 {blue:02d}\n")
    
    print(f"\n✅ 报告已保存到 {OUTPUT_DIR}/latest_report.txt")


if __name__ == "__main__":
    main()
