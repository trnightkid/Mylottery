"""
GAN v3: 分类头替代 GAN 生成器
思路：训练一个直接预测"下一期红球分布"的多标签分类器，
用交叉熵 + 标签平滑，而非对抗训练。
蓝球用独立分类器。
"""
import os, sys, torch, json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import LotteryDataset, prepare_backtest_data


class RedClassifier(nn.Module):
    """红球分类器：33类，预测6个红球（多标签）"""
    def __init__(self, cond_dim=99, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 33 * 6),  # 每球一个33类
        )
        self.hidden = hidden

    def forward(self, cond):
        out = self.net(cond)  # (batch, 33*6)
        out = out.view(-1, 6, 33)  # (batch, 6, 33) - 每球一个分布
        return out  # 返回每球的 logits


class BlueClassifier(nn.Module):
    """蓝球分类器：16类单标签"""
    def __init__(self, cond_dim=99):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16),
        )

    def forward(self, cond):
        return self.net(cond)  # (batch, 16)


class LotteryClassifierV3(nn.Module):
    """双色球分类器 v3"""
    def __init__(self, cond_dim=99, device='cpu'):
        super().__init__()
        self.red_cls = RedClassifier(cond_dim).to(device)
        self.blue_cls = BlueClassifier(cond_dim).to(device)
        
        # 6个红球共享分类器但分开预测
        self.red_opt = torch.optim.Adam(self.red_cls.parameters(), lr=3e-4, betas=(0.5, 0.9))
        self.blue_opt = torch.optim.Adam(self.blue_cls.parameters(), lr=3e-4, betas=(0.5, 0.9))
        self.device = device

    def predict_topk(self, cond, k=6):
        """预测红球：取每球概率最高的 k 个作为候选，再合并去重选6个"""
        self.eval()
        with torch.no_grad():
            cond = cond.to(self.device)
            red_logits = self.red_cls(cond)  # (batch, 6, 33)
            red_probs = F.softmax(red_logits, dim=2)  # (batch, 6, 33)
            
            # 每球取 top 10 候选
            topk = 10
            vals, idxs = torch.topk(red_probs, topk, dim=2)  # (batch, 6, topk)
            
            # 汇总所有候选的得分
            batch_size = cond.size(0)
            all_scores = torch.zeros(batch_size, 33, device=self.device)
            for ball_i in range(6):
                for t in range(topk):
                    all_scores += red_probs[:, ball_i, idxs[:, ball_i, t]]
            
            # 取得分最高的6个
            _, top6_idx = torch.topk(all_scores, k=6, dim=1)  # (batch, 6)
            
            blue_logits = self.blue_cls(cond)
            blue_idx = torch.argmax(blue_logits, dim=1)  # (batch,)
            
            return top6_idx, blue_idx

    def train_step(self, real_red, real_blue, cond, label_smoothing=0.05):
        """
        训练：每球独立交叉熵 + 蓝球交叉熵
        """
        batch_size = real_red.size(0)
        
        # ---- 红球训练 ----
        self.red_opt.zero_grad()
        red_logits = self.red_cls(cond)  # (batch, 6, 33)
        
        # 每球独立交叉熵
        loss_red = 0.0
        for ball_i in range(6):
            # 标签平滑
            target = real_red[:, ball_i]  # (batch,)
            loss = F.cross_entropy(red_logits[:, ball_i], target, label_smoothing=label_smoothing)
            loss_red += loss
        loss_red = loss_red / 6
        
        loss_red.backward()
        self.red_opt.step()
        
        # ---- 蓝球训练 ----
        self.blue_opt.zero_grad()
        blue_logits = self.blue_cls(cond)
        loss_blue = F.cross_entropy(blue_logits, real_blue, label_smoothing=label_smoothing)
        loss_blue.backward()
        self.blue_opt.step()
        
        return loss_red.item(), loss_blue.item()

    def save(self, prefix):
        torch.save({'red': self.red_cls.state_dict(), 'blue': self.blue_cls.state_dict()},
                   f"{prefix}_v3.pt")
        torch.save({'red_opt': self.red_opt.state_dict(), 'blue_opt': self.blue_opt.state_dict()},
                   f"{prefix}_v3_opt.pt")

    def load(self, prefix):
        ckpt = torch.load(f"{prefix}_v3.pt", map_location=self.device, weights_only=False)
        self.red_cls.load_state_dict(ckpt['red'])
        self.blue_cls.load_state_dict(ckpt['blue'])
        opt = torch.load(f"{prefix}_v3_opt.pt", map_location=self.device, weights_only=False)
        self.red_opt.load_state_dict(opt['red_opt'])
        self.blue_opt.load_state_dict(opt['blue_opt'])


def train(csv_path, epochs=500, batch_size=64, test_periods=200, save_dir='output', resume=True):
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = os.path.join(save_dir, "lottery_v3")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    (train_red, train_blue, train_cond), (test_red, test_blue, test_cond), dataset = \
        prepare_backtest_data(csv_path, test_periods=test_periods)
    
    n = train_red.size(0)
    print(f"Train: {n} samples, Epochs: {epochs}, Batch: {batch_size}")

    train_red = train_red.to(device)
    train_blue = train_blue.to(device)
    train_cond = train_cond.to(device)
    test_red = test_red.to(device)
    test_blue = test_blue.to(device)
    test_cond = test_cond.to(device)

    model = LotteryClassifierV3(device=device)

    if resume and os.path.exists(f"{save_prefix}_v3.pt"):
        model.load(save_prefix)
        print("✓ Resume from checkpoint")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        train_red = train_red[perm]
        train_blue = train_blue[perm]
        train_cond = train_cond[perm]

        epoch_loss_r, epoch_loss_b = 0.0, 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            rb = train_red[i:i+batch_size]
            bb = train_blue[i:i+batch_size]
            c = train_cond[i:i+batch_size]
            lr, lb = model.train_step(rb, bb, c)
            epoch_loss_r += lr
            epoch_loss_b += lb
            n_batches += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Red loss: {epoch_loss_r/n_batches:.4f} | Blue loss: {epoch_loss_b/n_batches:.4f}")

        if epoch % 100 == 0:
            model.save(save_prefix)
            print(f"  ✓ Saved (Epoch {epoch})")

    model.save(save_prefix)
    print(f"\n✓ Training done: {save_prefix}")

    # Quick backtest
    model.eval()
    total = test_red.size(0)
    red_hits = []
    blue_hits = 0

    for i in range(total):
        cond = test_cond[i:i+1]
        pred_red, pred_blue = model.predict_topk(cond, k=6)
        pred_red = pred_red[0].cpu().numpy() + 1
        pred_blue = pred_blue[0].cpu().item() + 1
        real_red = test_red[i].cpu().numpy() + 1
        real_blue = test_blue[i].cpu().item() + 1
        
        rh = len(set(real_red) & set(pred_red))
        bh = (real_blue == pred_blue)
        red_hits.append(rh)
        if bh:
            blue_hits += 1

    avg_hit = sum(red_hits) / total
    hit3plus = sum(1 for h in red_hits if h >= 3) / total * 100
    blue_rate = blue_hits / total * 100

    print(f"\n📊 Backtest (v3):")
    print(f"   Red avg hit: {avg_hit:.2f}/6 | 3+ hit: {hit3plus:.1f}% | Blue hit: {blue_rate:.1f}%")

    result = {
        'red_avg_hit': round(avg_hit, 2),
        'hit_3plus_rate': round(hit3plus, 1),
        'blue_hit_rate': round(blue_rate, 1),
        'epochs': epochs,
    }
    with open(os.path.join(save_dir, 'backtest_v3.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Results saved")

    return model, result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='../lottery_data.csv')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_periods', type=int, default=200)
    parser.add_argument('--save_dir', default='output')
    args = parser.parse_args()

    train(args.csv, args.epochs, args.batch_size, args.test_periods, args.save_dir)
