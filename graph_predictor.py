"""
双色球预测算法 v3 - 图算法增强版
===================================
新增图论分析模块:
1. 号码共现图 - 分析哪些号码经常一起出现
2. 间隔网络 - 号码之间的间隔规律  
3. 聚类分析 - 将号码聚类后发现热区冷区
4. PageRank风格评分 - 哪些号码是"核心"号码
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations
import networkx as nx
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LotteryGraphAnalyzer:
    """基于图论的双色球分析器"""
    
    def __init__(self, df):
        self.df = df
        self.red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        self.RED_RANGE = range(1, 34)
        self.BLUE_RANGE = range(1, 17)
        
        # 构建号码共现矩阵
        self.cooccurrence_matrix = self._build_cooccurrence_matrix()
        
        # 构建号码间隔网络
        self.interval_network = self._build_interval_network()
        
    def _build_cooccurrence_matrix(self):
        """构建红球共现矩阵"""
        matrix = np.zeros((33, 33), dtype=int)
        
        for _, row in self.df.iterrows():
            reds = sorted([row[c] for c in self.red_cols])
            # 统计每对号码的共现次数
            for i, j in combinations(range(6), 2):
                n1, n2 = reds[i], reds[j]
                matrix[n1-1][n2-1] += 1
                matrix[n2-1][n1-1] += 1
        
        return matrix
    
    def _build_interval_network(self):
        """构建号码间隔网络（基于相邻两期）"""
        intervals = defaultdict(list)
        
        prev_reds = None
        for _, row in self.df.iterrows():
            curr_reds = sorted([row[c] for c in self.red_cols])
            
            if prev_reds:
                for p in prev_reds:
                    for c in curr_reds:
                        diff = abs(c - p)
                        intervals[diff].append(1)
            
            prev_reds = curr_reds
        
        # 统计各间隔出现次数
        interval_counts = {k: sum(v) for k, v in intervals.items()}
        return interval_counts
    
    def get_page_rank_scores(self):
        """使用 PageRank 算法找出"核心"号码"""
        G = nx.Graph()
        
        # 添加节点
        for n in self.RED_RANGE:
            G.add_node(n)
        
        # 添加边（基于共现次数）
        for i in range(33):
            for j in range(i+1, 33):
                weight = self.cooccurrence_matrix[i][j]
                if weight > 0:
                    G.add_edge(i+1, j+1, weight=weight)
        
        # 计算 PageRank
        try:
            pr = nx.pagerank(G, weight='weight', alpha=0.85)
        except:
            pr = {n: 1/33 for n in self.RED_RANGE}
        
        return pr
    
    def get_hot_regions(self):
        """基于聚类分析找出号码热区"""
        # 将33个号码分成3个区域
        regions = {
            'low': list(range(1, 12)),      # 1-11 小号区
            'mid': list(range(12, 23)),     # 12-22 中号区  
            'high': list(range(23, 34))     # 23-33 大号区
        }
        
        region_counts = {'low': 0, 'mid': 0, 'high': 0}
        total_balls = 0
        
        for _, row in self.df.iterrows():
            for col in self.red_cols:
                ball = row[col]
                total_balls += 1
                
                if ball <= 11:
                    region_counts['low'] += 1
                elif ball <= 22:
                    region_counts['mid'] += 1
                else:
                    region_counts['high'] += 1
        
        # 计算各区域出现频率
        region_probs = {
            'low': region_counts['low'] / total_balls,
            'mid': region_counts['mid'] / total_balls,
            'high': region_counts['high'] / total_balls,
        }
        
        # 理论概率
        theory = {
            'low': 11/33,
            'mid': 11/33,
            'high': 11/33,
        }
        
        return region_probs, theory
    
    def get_interval_pattern(self):
        """分析号码间隔规律"""
        if not self.interval_network:
            return {}
        
        total = sum(self.interval_network.values())
        interval_probs = {k: v/total for k, v in self.interval_network.items()}
        
        return interval_probs
    
    def get_cooccurrence_network_stats(self):
        """共现网络统计"""
        G = nx.Graph()
        
        for i in range(33):
            for j in range(i+1, 33):
                weight = self.cooccurrence_matrix[i][j]
                if weight > 10:  # 只添加高频共现
                    G.add_edge(i+1, j+1, weight=weight)
        
        stats = {
            'density': nx.density(G),
            'clustering': nx.average_clustering(G, weight='weight'),
        }
        
        return stats
    
    def predict_with_graph(self, n_predictions=5):
        """基于图算法生成预测"""
        pr_scores = self.get_page_rank_scores()
        region_probs, theory = self.get_hot_regions()
        
        # 综合评分
        scores = {}
        
        # 频率统计
        all_reds = []
        for col in self.red_cols:
            all_reds.extend(self.df[col].tolist())
        
        freq = Counter(all_reds)
        max_freq = max(freq.values())
        
        for num in self.RED_RANGE:
            # 频率得分 (0-1)
            freq_score = freq[num] / max_freq if max_freq > 0 else 0
            
            # PageRank得分 (0-1)
            pr_score = pr_scores.get(num, 0)
            max_pr = max(pr_scores.values()) if pr_scores else 1
            pr_normalized = pr_score / max_pr if max_pr > 0 else 0
            
            # 区域得分
            if num <= 11:
                region_score = region_probs['low'] / theory['low'] if theory['low'] > 0 else 1
            elif num <= 22:
                region_score = region_probs['mid'] / theory['mid'] if theory['mid'] > 0 else 1
            else:
                region_score = region_probs['high'] / theory['high'] if theory['high'] > 0 else 1
            
            # 综合得分: 频率40% + PageRank35% + 区域25%
            combined = freq_score * 0.4 + pr_normalized * 0.35 + region_score * 0.25
            scores[num] = combined
        
        # 按得分排序，选出预测
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        predictions = []
        for i in range(n_predictions):
            # 每次选6个，分层选取
            selected = []
            
            # 高分区选3个
            for num, _ in sorted_nums[:15]:
                if len(selected) >= 3:
                    break
                selected.append(num)
            
            # 中分区选2个
            for num, _ in sorted_nums[10:25]:
                if num not in selected and len(selected) < 5:
                    selected.append(num)
            
            # 低分区选1个
            for num, _ in sorted_nums[20:]:
                if num not in selected and len(selected) < 6:
                    selected.append(num)
            
            selected = sorted(selected)
            predictions.append(selected)
        
        # 蓝球预测（简化版）
        blue_freq = Counter(self.df['blue'].tolist())
        blue_sorted = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)
        
        return predictions, blue_sorted[:5]
    
    def plot_graph_analysis(self, output_dir):
        """绘制图论分析可视化"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 图1: 共现网络
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 子图1: 共现热力图
        ax1 = axes[0, 0]
        im = ax1.imshow(self.cooccurrence_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_title('Red Ball Co-occurrence Heatmap', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Ball Number')
        ax1.set_ylabel('Ball Number')
        ax1.set_xticks(range(0, 33, 3))
        ax1.set_yticks(range(0, 33, 3))
        ax1.set_xticklabels(range(1, 34, 3))
        ax1.set_yticklabels(range(1, 34, 3))
        plt.colorbar(im, ax=ax1, label='Co-occurrence Count')
        
        # 子图2: PageRank分布
        ax2 = axes[0, 1]
        pr_scores = self.get_page_rank_scores()
        nums = list(self.RED_RANGE)
        pr_vals = [pr_scores.get(n, 0) for n in nums]
        
        colors = ['#FF4444' if n <= 11 else '#FFA500' if n <= 22 else '#4ECDC4' for n in nums]
        bars = ax2.bar(nums, pr_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax2.axhline(y=np.mean(pr_vals), color='blue', linestyle='--', label=f'Mean: {np.mean(pr_vals):.4f}')
        ax2.set_title('PageRank Scores (Graph Centrality)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Red Ball Number')
        ax2.set_ylabel('PageRank Score')
        ax2.legend()
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF4444', label='Low (1-11)'),
            Patch(facecolor='#FFA500', label='Mid (12-22)'),
            Patch(facecolor='#4ECDC4', label='High (23-33)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # 子图3: 区域概率对比
        ax3 = axes[1, 0]
        region_probs, theory = self.get_hot_regions()
        regions = ['Low\n(1-11)', 'Mid\n(12-22)', 'High\n(23-33)']
        actual = [region_probs['low'], region_probs['mid'], region_probs['high']]
        expected = [theory['low'], theory['mid'], theory['high']]
        
        x = np.arange(len(regions))
        width = 0.35
        
        ax3.bar(x - width/2, actual, width, label='Actual', color='#4ECDC4')
        ax3.bar(x + width/2, expected, width, label='Theoretical', color='#FF6B6B')
        ax3.set_title('Region Distribution: Actual vs Theoretical', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Probability')
        ax3.set_xticks(x)
        ax3.set_xticklabels(regions)
        ax3.legend()
        
        for i, (a, e) in enumerate(zip(actual, expected)):
            ax3.annotate(f'{a:.3f}', xy=(i - width/2, a), ha='center', va='bottom', fontsize=9)
            ax3.annotate(f'{e:.3f}', xy=(i + width/2, e), ha='center', va='bottom', fontsize=9)
        
        # 子图4: 共现网络图
        ax4 = axes[1, 1]
        G = nx.Graph()
        
        for i in range(33):
            for j in range(i+1, 33):
                weight = self.cooccurrence_matrix[i][j]
                if weight > 50:  # 只显示高频共现
                    G.add_edge(i+1, j+1, weight=weight)
        
        if len(G.nodes()) > 0:
            pr_scores = self.get_page_rank_scores()
            node_sizes = [pr_scores.get(n, 0) * 5000 + 100 for n in G.nodes()]
            
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            nx.draw_networkx_nodes(G, pos, ax=ax4, 
                                   node_size=node_sizes,
                                   node_color='#FF6B6B',
                                   alpha=0.8)
            nx.draw_networkx_edges(G, pos, ax=ax4,
                                   edge_color='#CCCCCC',
                                   alpha=0.5)
            nx.draw_networkx_labels(G, pos, ax=ax4,
                                   font_size=7)
            ax4.set_title('Co-occurrence Network (freq>50)', fontsize=12, fontweight='bold')
        
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/graph_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ graph_analysis.png")
        
        return pr_scores


class EnhancedPredictor:
    """增强版预测器 - 结合图算法 + 蒙特卡洛"""
    
    def __init__(self, df):
        self.df = df
        self.graph_analyzer = LotteryGraphAnalyzer(df)
        self.red_cols = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        
        # 基础统计
        self.red_freq = self._calc_red_frequency()
        self.blue_freq = self._calc_blue_frequency()
        
    def _calc_red_frequency(self):
        all_reds = []
        for col in self.red_cols:
            all_reds.extend(self.df[col].tolist())
        return Counter(all_reds)
    
    def _calc_blue_frequency(self):
        return Counter(self.df['blue'].tolist())
    
    def _weighted_freq_score(self, num, is_red=True):
        """计算加权频率得分"""
        if is_red:
            freq = self.red_freq
            max_freq = max(freq.values())
        else:
            freq = self.blue_freq
            max_freq = max(freq.values())
        
        return freq.get(num, 0) / max_freq if max_freq > 0 else 0
    
    def _page_rank_score(self, num):
        """PageRank得分"""
        pr = self.graph_analyzer.get_page_rank_scores()
        max_pr = max(pr.values()) if pr else 1
        return pr.get(num, 0) / max_pr if max_pr > 0 else 0
    
    def _interval_score(self, num):
        """间隔规律得分"""
        intervals = self.graph_analyzer.get_interval_pattern()
        if not intervals:
            return 0.5
        
        # 号码num与其他号码的常见间隔
        score = 0
        for other_num in range(1, 34):
            if other_num == num:
                continue
            diff = abs(num - other_num)
            score += intervals.get(diff, 0)
        
        return min(score / 100, 1.0)
    
    def _region_score(self, num):
        """区域偏离得分"""
        region_probs, theory = self.graph_analyzer.get_hot_regions()
        
        if num <= 11:
            expected = theory['low']
            actual = region_probs['low']
        elif num <= 22:
            expected = theory['mid']
            actual = region_probs['mid']
        else:
            expected = theory['high']
            actual = region_probs['high']
        
        ratio = actual / expected if expected > 0 else 1
        # 偏离越大，score越低（回归理论均值）
        return 1.0 / (1 + abs(ratio - 1))
    
    def combined_score(self, num, is_red=True):
        """
        综合评分体系：
        - 频率得分: 35%
        - PageRank得分: 30%
        - 间隔规律: 15%
        - 区域回归: 20%
        """
        freq_s = self._weighted_freq_score(num, is_red)
        pr_s = self._page_rank_score(num) if is_red else 0.5
        interval_s = self._interval_score(num) if is_red else 0.5
        region_s = self._region_score(num) if is_red else 0.5
        
        if is_red:
            combined = freq_s * 0.35 + pr_s * 0.30 + interval_s * 0.15 + region_s * 0.20
        else:
            combined = freq_s * 0.70 + interval_s * 0.30
        
        return combined
    
    def monte_carlo_with_graph(self, n_samples=1000, top_k=10):
        """基于图算法增强的蒙特卡洛采样"""
        predictions = []
        
        for _ in range(n_samples):
            # 计算所有号码的综合得分
            red_scores = {n: self.combined_score(n, True) for n in range(1, 34)}
            blue_scores = {n: self.combined_score(n, False) for n in range(1, 17)}
            
            # 按得分分层采样
            reds = self._stratified_sample(red_scores, 6)
            blues = self._stratified_sample(blue_scores, 1)
            
            predictions.append({
                'reds': sorted(reds),
                'blue': blues[0],
                'score': sum(red_scores[r] for r in reds) + blue_scores[blues[0]]
            })
        
        # 按综合得分排序
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        return predictions[:top_k]
    
    def _stratified_sample(self, scores, n):
        """分层采样 - 按得分分段选取"""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        # 高分区取2个
        for num, score in sorted_items[:12]:
            if len(selected) < 2:
                selected.append(num)
        
        # 中分区取2个
        for num, score in sorted_items[10:22]:
            if num not in selected and len(selected) < 4:
                selected.append(num)
        
        # 低分区取2个
        for num, score in sorted_items[20:]:
            if num not in selected and len(selected) < 6:
                selected.append(num)
        
        # 随机替换几个
        remaining = [n for n, _ in sorted_items if n not in selected]
        np.random.shuffle(remaining)
        
        while len(selected) < n:
            if remaining:
                cand = remaining.pop()
                if cand not in selected:
                    selected.append(cand)
            else:
                break
        
        return selected[:n]
    
    def generate_predictions(self, n=5):
        """生成最终预测"""
        print("\n" + "=" * 60)
        print("🎯 图算法增强预测结果")
        print("=" * 60)
        
        # 图分析
        pr_scores = self.graph_analyzer.plot_graph_analysis('/home/clawd/Mylottery/output')
        
        # 蒙特卡洛
        top_preds = self.monte_carlo_with_graph(n_samples=1000, top_k=n)
        
        print(f"\n📊 综合得分TOP15号码:")
        all_red_scores = {n: self.combined_score(n, True) for n in range(1, 34)}
        sorted_scores = sorted(all_red_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (num, score) in enumerate(sorted_scores[:15], 1):
            freq = self.red_freq.get(num, 0)
            pr = pr_scores.get(num, 0)
            print(f"   {i:2d}. 号码{num:02d}: 综合={score:.4f} 频率={freq:3d}次 PR={pr:.4f}")
        
        print(f"\n🏆 最终预测 (TOP {n}):")
        for i, pred in enumerate(top_preds, 1):
            print(f"\n   预测{i}: 红球{pred['reds']} 蓝球{pred['blue']:02d} (得分:{pred['score']:.4f})")
        
        return top_preds


def main():
    """测试运行"""
    import pymysql
    
    print("📥 加载数据...")
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='Lottery2026!',
            database='lottery_db',
            charset='utf8mb4'
        )
        df = pd.read_sql("SELECT * FROM lottery_data ORDER BY CAST(period AS UNSIGNED) DESC LIMIT 500", conn)
        conn.close()
    except:
        # 备用：从CSV加载
        df = pd.read_csv('/home/clawd/Mylottery/lottery_data.csv')
        df = df.sort_values('period', ascending=False).head(500)
    
    print(f"   加载{len(df)}条数据")
    
    # 运行图算法分析
    predictor = EnhancedPredictor(df)
    predictions = predictor.generate_predictions(n=5)
    
    print("\n✅ 分析完成!")


if __name__ == "__main__":
    main()
