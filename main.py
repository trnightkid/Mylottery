import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
from src.database import DatabaseConnector
from src.analyzer import LotteryAnalyzer
from src.predictor import LotteryPredictor
from src.visualizer import LotteryVisualizer
from config import Config
import os


def main():
    """主程序"""
    print("=" * 60)
    print("双色球数据分析与概率预测系统")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    try:
        # 1. 加载数据
        print("\n[步骤 1/5] 正在连接数据库并加载数据...")
        db = DatabaseConnector()
        df = db.load_lottery_data()
        print(f"✓ 成功加载 {len(df)} 期数据")

        # 2. 数据分析
        print("\n[步骤 2/5] 进行历史数据分析...")
        analyzer = LotteryAnalyzer(df)
        report = analyzer.generate_report()

        # 保存报告
        report_path = f"{Config.OUTPUT_DIR}/report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ 分析报告已保存: {report_path}")
        print("\n" + report)

        # 3. 概率预测
        print("\n[步骤 3/5] 计算号码出现概率...")
        predictor = LotteryPredictor(df)

        # 计算红球概率
        red_probs = predictor.calculate_red_probabilities()
        blue_probs = predictor.calculate_blue_probabilities()

        print("\n红球概率TOP10:")
        for num, prob in predictor.predict_top_numbers(red_probs, 10):
            print(f"  号码 {num:2d}: {prob * 100:.2f}%")

        print("\n蓝球概率TOP5:")
        for num, prob in predictor.predict_top_numbers(blue_probs, 5):
            print(f"  号码 {num:2d}: {prob * 100:.2f}%")

        # 生成推荐
        print("\n[步骤 4/5] 生成推荐号码...")
        recommendations = predictor.generate_recommendation()

        # 保存推荐结果
        preds_path = f"{Config.OUTPUT_DIR}/predictions.json"
        with open(preds_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)
        print(f"✓ 预测结果已保存: {preds_path}")

        # 显示最终推荐
        print("\n" + "=" * 50)
        print("【最终推荐号码】")
        print(f"红球: {recommendations['final_recommendation']['red_balls']}")
        print(f"蓝球: {recommendations['final_recommendation']['blue_ball']}")
        print("=" * 50)

        # 4. 数据可视化
        print("\n[步骤 5/5] 生成可视化图表...")
        visualizer = LotteryVisualizer()

        red_freq = analyzer.analyze_red_ball_frequency()
        blue_freq = analyzer.analyze_blue_ball_frequency()
        consecutive = analyzer.analyze_consecutive_numbers()
        odd_even = analyzer.calculate_odd_even_ratio()
        blue_patterns = analyzer.analyze_blue_ball_patterns()

        visualizer.plot_red_ball_frequency(red_freq)
        visualizer.plot_blue_ball_frequency(blue_freq)
        visualizer.plot_consecutive_ratio(consecutive)
        visualizer.plot_odd_even_ratio(odd_even)
        visualizer.plot_blue_ball_patterns(blue_patterns)

        print("\n" + "=" * 60)
        print("所有任务完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()