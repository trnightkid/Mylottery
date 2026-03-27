"""
GAN 训练脚本 v2 - 配合 gan_model_v2.py
关键改进：
1. D:G 训练比例 = 1:2（生成器训练更频繁）
2. G 学习率更高（2e-4 vs D 的 5e-5）
3. 断点续训
4. 每个 epoch 输出生成样本检查多样性
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gan_model_v2 import LotteryGANV2
from dataset import LotteryDataset, prepare_backtest_data


def validate_red_diversity(red_indices: torch.Tensor) -> dict:
    """检查生成红球的多样性"""
    batch_size = red_indices.size(0)
    reds = red_indices.cpu().numpy()
    
    # 覆盖了多少个不同号码
    unique_nums = len(set(reds.flatten()))
    # 每行有多少重复
    duplicates = []
    for row in reds:
        duplicates.append(6 - len(set(row)))
    avg_dup = np.mean(duplicates)
    
    return {
        'unique_nums': unique_nums,
        'avg_duplicates': avg_dup,
    }


def train(
    csv_path: str,
    epochs: int = 300,
    batch_size: int = 64,
    test_periods: int = 200,
    save_dir: str = "output",
    device: str = None,
    resume: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = os.path.join(save_dir, "lottery_gan_v2")

    # ---- 设备选择 ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 加载数据 ----
    (train_red, train_blue, train_cond), (test_red, test_blue, test_cond), dataset = \
        prepare_backtest_data(csv_path, test_periods=test_periods)

    n_samples = train_red.size(0)
    print(f"训练样本数: {n_samples}, 批大小: {batch_size}, 轮次: {epochs}")

    train_red = train_red.to(device)
    train_blue = train_blue.to(device)
    train_cond = train_cond.to(device)
    test_red = test_red.to(device)
    test_blue = test_blue.to(device)
    test_cond = test_cond.to(device)

    # ---- 初始化模型 v2 ----
    gan = LotteryGANV2(
        noise_dim=128,
        cond_dim=99,
        red_lr=2e-4,
        blue_lr=2e-4,
        d_lr=5e-5,
        device=device,
    )

    if resume:
        ckpt_path = f"{save_prefix}_gan.pt"
        if os.path.exists(ckpt_path):
            gan.load(save_prefix)
            print(f"✓ 断点续训: {ckpt_path}")

    # ---- 训练循环 ----
    d_loss_history = []
    g_loss_history = []

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_samples)
        train_red = train_red[perm]
        train_blue = train_blue[perm]
        train_cond = train_cond[perm]

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            real_red = train_red[i:i+batch_size]
            real_blue = train_blue[i:i+batch_size]
            cond = train_cond[i:i+batch_size]

            # 真实数据 -> 概率向量
            real_red_probs, real_blue_probs = gan.real_to_probs(real_red, real_blue)

            # 红球训练（D 1步, G 2步）
            d_r, g_r = gan.train_step_red(real_red, cond, real_red_probs, d_steps=1, g_steps=2)
            # 蓝球训练
            d_b, g_b = gan.train_step_blue(real_blue, cond, real_blue_probs, d_steps=1, g_steps=2)

            epoch_d_loss += (d_r + d_b) / 2
            epoch_g_loss += (g_r + g_b) / 2
            n_batches += 1

        avg_d = epoch_d_loss / n_batches
        avg_g = epoch_g_loss / n_batches
        d_loss_history.append(avg_d)
        g_loss_history.append(avg_g)

        # ---- 每10个epoch检查一次多样性 ----
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                fake_red, fake_blue = gan.generate(min(16, batch_size), test_cond[:16])
                div = validate_red_diversity(fake_red)
                blue_unique = len(set(fake_blue.tolist()))
                
                # 生成样本
                fake_r = fake_red.cpu().numpy() + 1
                fake_b = fake_blue.cpu().numpy() + 1
                samples = [f"红:{sorted(r)} 蓝:{b}" for r, b in zip(fake_r[:3], fake_b[:3])]

            print(f"Epoch {epoch:4d} | D={avg_d:.4f} G={avg_g:.4f} | "
                  f"红球覆盖:{div['unique_nums']}/33 重复:{div['avg_duplicates']:.2f} | "
                  f"蓝球种类:{blue_unique}/16")
            for s in samples:
                print(f"    {s}")

        # ---- 保存 ----
        if epoch % 50 == 0:
            gan.save(save_prefix)
            print(f"  ✓ 模型已保存 (Epoch {epoch})")

    # ---- 最终保存 ----
    gan.save(save_prefix)
    print(f"\n✓ 训练完成！模型: {save_prefix}")

    # ---- 绘图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(d_loss_history, label='D loss', alpha=0.7)
        axes[0].plot(g_loss_history, label='G loss', alpha=0.7)
        axes[0].set_title('Training Loss Curves (v2)')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()

        with torch.no_grad():
            fake_red, fake_blue = gan.generate(5, test_cond[-5:])
            fake_red = fake_red.cpu().numpy() + 1
            fake_blue = fake_blue.cpu().numpy() + 1

        sample_text = []
        for i in range(5):
            reds = sorted(fake_red[i].tolist())
            blues = fake_blue[i].tolist()
            sample_text.append(f"红球: {reds}  蓝球: {blues}")

        axes[1].text(0.1, 0.5, '\n'.join(sample_text), fontsize=10,
                     family='monospace', verticalalignment='center')
        axes[1].set_title('Generated Samples (Last Epoch)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves_v2.png"), dpi=150)
        print(f"✓ 训练曲线图已保存: {save_dir}/training_curves_v2.png")
    except Exception as e:
        print(f"⚠ 绘图失败: {e}")

    return gan, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练双色球 GAN v2 模型")
    parser.add_argument("--csv", type=str, default="lottery_data.csv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_periods", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")

    args = parser.parse_args()

    train(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_periods=args.test_periods,
        save_dir=args.save_dir,
        device=args.device,
        resume=args.resume,
    )
