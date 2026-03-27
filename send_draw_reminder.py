#!/usr/bin/env python3
"""
双色球开奖提醒 - 轻量版
从 OpenClaw 内存读取 Telegram chat ID，用 cron 触发 agent 来发送预测
"""
import os, sys, json, torch
from collections import Counter
from datetime import datetime

CSV_PATH = "/home/openclaw/.openclaw/workspace/lottery_data.csv"
MODEL_PREFIX = "/home/openclaw/.openclaw/workspace/gan_lottery/output/lottery_v3"
GAN_PREFIX = "/home/openclaw/.openclaw/workspace/gan_lottery/output/lottery_gan_v2"
DEVICE = "cpu"


def get_next_draw():
    today = datetime.now()
    wd = today.weekday()  # 0=周一
    table = {0: (1, "周二"), 1: (2, "周四"), 2: (1, "周四"),
             3: (3, "周日"), 4: (2, "周日"), 5: (1, "周日"), 6: (1, "周二")}
    days, name = table[wd]
    from datetime import timedelta
    nd = today + timedelta(days=days)
    return nd, name


def load_v3_prediction():
    sys.path.insert(0, '/home/openclaw/.openclaw/workspace')
    from gan_lottery.train_v3 import LotteryClassifierV3
    from gan_lottery.dataset import LotteryDataset

    ds = LotteryDataset(CSV_PATH)
    latest = ds.records[-1]
    cond = torch.from_numpy(ds.get_condition_vector(len(ds)-1)).float().unsqueeze(0).to(DEVICE)

    model = LotteryClassifierV3(device=DEVICE)
    model.load(MODEL_PREFIX)
    model.eval()

    all_r, all_b = [], []
    for _ in range(16):
        with torch.no_grad():
            pr, pb = model.predict_topk(cond, k=6)
            all_r.extend((pr[0].cpu().numpy() + 1).tolist())
            all_b.append(pb[0].cpu().item() + 1)

    rc = Counter(all_r)
    pred_red = sorted([r for r, _ in sorted(rc.items(), key=lambda x: -x[1])[:6]])
    bc = Counter(all_b)
    pred_blue = bc.most_common(1)[0][0]
    return latest['period'], pred_red, pred_blue


def load_gan_prediction():
    sys.path.insert(0, '/home/openclaw/.openclaw/workspace')
    from gan_lottery.gan_model_v2 import LotteryGANV2
    from gan_lottery.dataset import LotteryDataset

    ds = LotteryDataset(CSV_PATH)
    cond = torch.from_numpy(ds.get_condition_vector(len(ds)-1)).float().unsqueeze(0).to(DEVICE)

    gan = LotteryGANV2(noise_dim=128, cond_dim=99, device=DEVICE)
    gan.load(GAN_PREFIX)
    gan.eval()

    all_r, all_b = [], []
    for _ in range(16):
        with torch.no_grad():
            fr, fb = gan.generate(2, cond, temperature=1.0)
            all_r.extend((fr[0].cpu().numpy() + 1).tolist())
            all_b.append(fb[0].cpu().item() + 1)

    rc = Counter(all_r)
    pred_red = sorted([r for r, _ in rc.most_common(6)])
    bc = Counter(all_b)
    pred_blue = bc.most_common(1)[0][0]
    return pred_red, pred_blue


def load_hot_prediction():
    sys.path.insert(0, '/home/openclaw/.openclaw/workspace')
    from gan_lottery.dataset import LotteryDataset
    ds = LotteryDataset(CSV_PATH)
    hot = Counter([r for rec in ds.records[-20:] for r in rec['red']])
    return sorted([n for n, _ in hot.most_common(6)])


def main():
    from datetime import timedelta

    next_draw, draw_name = get_next_draw()
    latest, v3_red, v3_blue = load_v3_prediction()
    gan_red, gan_blue = load_gan_prediction()
    hot_red = load_hot_prediction()

    lines = [
        f"🎰 <b>双色球开奖提醒</b>",
        f"📅 开奖日: {next_draw.strftime('%Y-%m-%d')} ({draw_name})",
        f"📊 最新一期: 第 {latest} 期",
        "",
        "🔮 <b>预测推荐</b>",
        f"  v3分类器: 红球 {' '.join(f'{n:02d}' for n in v3_red)} + 蓝球 {v3_blue:02d}",
        f"  v2 GAN:   红球 {' '.join(f'{n:02d}' for n in gan_red)} + 蓝球 {gan_blue:02d}",
        f"  近期热号: 红球 {' '.join(f'{n:02d}' for n in hot_red)} + 蓝球 {v3_blue:02d}",
        "",
        "⚠ 彩票为随机事件，预测仅供参考",
        "请理性购彩，切勿沉迷",
    ]
    print("\n".join(lines))

    # 保存到文件，供 agent 读取发送
    output = {
        "next_draw": next_draw.strftime("%Y-%m-%d"),
        "draw_name": draw_name,
        "latest": latest,
        "predictions": {
            "v3分类器": {"red": v3_red, "blue": v3_blue},
            "v2_GAN": {"red": gan_red, "blue": gan_blue},
            "近期热号": {"red": hot_red, "blue": v3_blue},
        },
        "message_text": "\n".join(lines),
    }
    out_path = "/home/openclaw/.openclaw/workspace/output/draw_reminder.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 预测已保存: {out_path}")


if __name__ == "__main__":
    main()
