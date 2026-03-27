"""预测脚本 v3 - 配合 train_v3.py"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v3 import LotteryClassifierV3
from dataset import LotteryDataset

def predict(csv_path='../lottery_data.csv', model_prefix='output/lottery_v3',
           n_samples=8, output_path='output/prediction_v3.json'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LotteryDataset(csv_path)
    latest = ds.records[-1]
    cond = torch.from_numpy(ds.get_condition_vector(len(ds)-1)).float().unsqueeze(0).to(device)

    model = LotteryClassifierV3(device=device)
    if os.path.exists(f"{model_prefix}_v3.pt"):
        model.load(model_prefix)
        print(f"✓ 模型已加载: {model_prefix}")
    else:
        print(f"⚠ 模型不存在: {model_prefix}_v3.pt")
        return

    # 多样本投票
    from collections import Counter
    all_red_votes = []
    all_blue_votes = []

    for i in range(n_samples):
        with torch.no_grad():
            pred_red, pred_blue = model.predict_topk(cond, k=6)
            all_red_votes.extend((pred_red[0].cpu().numpy() + 1).tolist())
            all_blue_votes.append(pred_blue[0].cpu().item() + 1)

    # 投票
    red_counts = Counter(all_red_votes)
    pred_red = [r for r, _ in sorted(red_counts.items(), key=lambda x: -x[1])[:6]]
    blue_counts = Counter(all_blue_votes)
    pred_blue = blue_counts.most_common(1)[0][0]

    # 热号
    hot = Counter([r for rec in ds.records[-20:] for r in rec['red']])
    hot_reds = [n for n, _ in hot.most_common(10)]

    print(f"\n🌟 第 {latest['period']+1} 期预测（基于 v3 分类器）:")
    print(f"  分类器: 红球 {sorted(pred_red)} + 蓝球 {pred_blue}")
    print(f"  近期热号: 红球 {hot_reds[:6]}")

    # 多计划
    recommendations = {
        "plan_A": {"name": "v3分类器投票", "red": sorted(pred_red), "blue": int(pred_blue)},
        "plan_B": {"name": "近期热号", "red": sorted(hot_reds[:6]), "blue": int(pred_blue)},
    }

    result = {
        "target_period": int(latest['period']) + 1,
        "recommendations": recommendations,
        "red_vote_stats": {str(k): v for k, v in red_counts.most_common(10)},
        "blue_vote_stats": {str(k): v for k, v in blue_counts.most_common()},
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='../lottery_data.csv')
    parser.add_argument('--model', default='output/lottery_v3')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--output', default='output/prediction_v3.json')
    args = parser.parse_args()
    predict(args.csv, args.model, args.samples, args.output)
