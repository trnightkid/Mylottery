"""
GAN 训练脚本
使用 WGAN-GP 稳定训练，分别训练红球和蓝球 GAN
支持断点续训
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gan_model import LotteryGAN
from dataset import LotteryDataset, prepare_backtest_data


def validate_red_uniqueness(red_indices: torch.Tensor) -> float:
    """检查生成的红球是否互不重复，返回重复率（越低越好）"""
    batch_size = red_indices.size(0)
    expected = 6
    # 每行独立检查有多少个不同的球
    unique_per_row = torch.stack([torch.unique(red_indices[i]) for i in range(batch_size)])
    n_unique = unique_per_row.size(1)
    # 理想情况每行6个不同球
    duplicate_rate = 1.0 - (n_unique / expected)
    return duplicate_rate


def train(
    csv_path: str,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-4,
    d_steps: int = 5,
    test_periods: int = 200,
    save_dir: str = "output",
    device: str = None,
    resume: bool = False,
):
    """
    完整训练流程

    Args:
        csv_path: 历史数据 CSV 路径
        epochs: 训练轮次
        batch_size: 批大小
        lr: 学习率
        d_steps: 判别器每训练1步，生成器训练 d_steps 步
        test_periods: 最后多少期作为测试集
        save_dir: 模型保存目录
        device: 训练设备（auto检测）
        resume: 是否断点续训
    """
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = os.path.join(save_dir, "lottery_gan")

    # ---- 设备选择 ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 加载数据 ----
    (train_red, train_blue, train_cond), (test_red, test_blue, test_cond), dataset = \
        prepare_backtest_data(csv_path, test_periods=test_periods)

    train_red = train_red.to(device)
    train_blue = train_blue.to(device)
    train_cond = train_cond.to(device)
    test_red = test_red.to(device)
    test_blue = test_blue.to(device)
    test_cond = test_cond.to(device)

    n_samples = train_red.size(0)
    print(f"训练样本数: {n_samples}, 批大小: {batch_size}, 轮次: {epochs}")

    # ---- 初始化模型 ----
    cond_dim = 99  # 条件向量实际维度（dataset.py 输出 99 维）
    gan = LotteryGAN(noise_dim=64, cond_dim=cond_dim, device=device)

    if resume:
        ckpt_path = f"{save_prefix}_gan.pt"
        if os.path.exists(ckpt_path):
            gan.load(save_prefix)
            print(f"✓ 断点续训: {ckpt_path}")

    # ---- 训练循环 ----
    d_loss_history = []
    g_loss_history = []

    for epoch in range(1, epochs + 1):
        # 打乱训练数据
        perm = torch.randperm(n_samples)
        train_red = train_red[perm]
        train_blue = train_blue[perm]
        train_cond = train_cond[perm]

        epoch_d_loss_red = 0.0
        epoch_g_loss_red = 0.0
        epoch_d_loss_blue = 0.0
        epoch_g_loss_blue = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            real_red = train_red[i:i+batch_size]
            real_blue = train_blue[i:i+batch_size]
            cond = train_cond[i:i+batch_size]

            # ---- 红球训练 ----
            for _ in range(d_steps):
                d_r, g_r = gan.train_step_red(real_red, cond)
            epoch_d_loss_red += d_r
            epoch_g_loss_red += g_r

            # ---- 蓝球训练 ----
            d_b, g_b = gan.train_step_blue(real_blue, cond)
            epoch_d_loss_blue += d_b
            epoch_g_loss_blue += g_b

            n_batches += 1

        avg_d_r = epoch_d_loss_red / n_batches
        avg_g_r = epoch_g_loss_red / n_batches
        avg_d_b = epoch_d_loss_blue / n_batches
        avg_g_b = epoch_g_loss_blue / n_batches

        d_loss_history.append((avg_d_r + avg_d_b) / 2)
        g_loss_history.append((avg_g_r + avg_g_b) / 2)

        if epoch % 10 == 0 or epoch == 1:
            # 生成样本检查
            with torch.no_grad():
                fake_red, fake_blue = gan.generate(min(8, batch_size), test_cond[:8])
                dup_rate = validate_red_uniqueness(fake_red)

            print(f"Epoch {epoch:4d} | "
                  f"D_red={avg_d_r:.4f} G_red={avg_g_r:.4f} | "
                  f"D_blue={avg_d_b:.4f} G_blue={avg_g_b:.4f} | "
                  f"重复率={dup_rate:.2%} | "
                  f"设备={device}")

        # ---- 保存模型 ----
        if epoch % 50 == 0:
            gan.save(save_prefix)
            print(f"  ✓ 模型已保存 (Epoch {epoch})")

    # ---- 最终保存 ----
    gan.save(save_prefix)
    print(f"\n✓ 训练完成！模型保存至: {save_prefix}")

    # ---- 训练损失曲线 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(d_loss_history, label='D loss')
        axes[0].plot(g_loss_history, label='G loss')
        axes[0].set_title('Training Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()

        # 展示最后一期的生成样本
        with torch.no_grad():
            fake_red, fake_blue = gan.generate(5, test_cond[-5:])
            fake_red = fake_red.cpu().numpy() + 1  # 转回1-33
            fake_blue = fake_blue.cpu().numpy() + 1  # 转回1-16

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
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
        print(f"✓ 训练曲线图已保存: {save_dir}/training_curves.png")
    except Exception as e:
        print(f"⚠ 绘图失败（可能缺少 matplotlib）: {e}")

    return gan, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练双色球 GAN 模型")
    parser.add_argument("--csv", type=str, default="lottery_data.csv",
                        help="历史数据 CSV 路径")
    parser.add_argument("--epochs", type=int, default=300,
                        help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_steps", type=int, default=5,
                        help="判别器每步，生成器训练次数")
    parser.add_argument("--test_periods", type=int, default=200,
                        help="最后多少期作为测试集")
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="断点续训")

    args = parser.parse_args()

    train(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_steps=args.d_steps,
        test_periods=args.test_periods,
        save_dir=args.save_dir,
        device=args.device,
        resume=args.resume,
    )
