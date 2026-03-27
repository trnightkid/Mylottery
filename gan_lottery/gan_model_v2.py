"""
GAN 彩票预测模型 v2 - 解决 Mode Collapse
关键改进：
1. Softmax + 贪心采样替代 topk（防止锁定在固定号码）
2. G/D 学习率比调整（生成器更快）
3. 多样性惩罚（Generated Diversity Loss）
4. 特征匹配损失（Feature Matching Loss）
5. D:G 训练步数比 = 1:2（生成器训练更频繁）
6. 判别器使用 logits 而非 one-hot（更稳定）
7. Dropout in D + 更大的 noise_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RedBallGeneratorV2(nn.Module):
    """红球生成器 v2：33选6，Softmax + 贪心采样"""
    def __init__(self, noise_dim: int = 128, cond_dim: int = 99):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 128, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 33),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """返回 (batch, 6) 红球索引，0-indexed"""
        cond128 = self.cond_proj(cond)
        x = torch.cat([z, cond128], dim=1)
        logits = self.fc(x)  # (batch, 33)
        probs = F.softmax(logits / temperature, dim=1)

        batch_size = z.size(0)
        selected = []
        probs_clone = probs.clone()

        for _ in range(6):
            row_sums = probs_clone.sum(dim=1, keepdim=True).clamp(min=1e-10)
            probs_norm = probs_clone / row_sums
            indices = torch.multinomial(probs_norm, 1).squeeze(-1)
            selected.append(indices)
            probs_clone.scatter_(1, indices.unsqueeze(1), 0.0)

        return torch.stack(selected, dim=1)

    def forward_logits(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """返回 logits，用于判别器和多样性损失"""
        cond128 = self.cond_proj(cond)
        x = torch.cat([z, cond128], dim=1)
        return self.fc(x)


class BlueBallGeneratorV2(nn.Module):
    """蓝球生成器 v2：16选1，Softmax 输出"""
    def __init__(self, noise_dim: int = 128, cond_dim: int = 99):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 16),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """返回 (batch,) 蓝球索引，0-indexed"""
        cond64 = self.cond_proj(cond)
        x = torch.cat([z, cond64], dim=1)
        logits = self.fc(x)
        probs = F.softmax(logits / temperature, dim=1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def forward_logits(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond64 = self.cond_proj(cond)
        x = torch.cat([z, cond64], dim=1)
        return self.fc(x)


class RedDiscriminatorV2(nn.Module):
    """红球判别器 v2：接收 logits，拼接完整概率向量"""
    def __init__(self, cond_dim: int = 99):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.LeakyReLU(0.2),
        )
        self.proj = nn.Sequential(
            nn.Linear(33, 32),   # 压缩33维概率
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 + 64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, logits_or_probs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        logits_or_probs: (batch, 33) Generator 输出的 logits 或 softmax 概率
        """
        probs = torch.softmax(logits_or_probs, dim=1)  # (batch, 33)
        prob_feat = self.proj(probs)  # (batch, 32)
        cond64 = self.cond_proj(cond)  # (batch, 64)
        x = torch.cat([prob_feat, cond64], dim=1)  # (batch, 96)
        return self.fc(x)


class BlueDiscriminatorV2(nn.Module):
    """蓝球判别器 v2"""
    def __init__(self, cond_dim: int = 99):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 + 32, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, logits_or_probs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits_or_probs, dim=1)
        cond32 = self.cond_proj(cond)
        x = torch.cat([probs, cond32], dim=1)
        return self.fc(x)


class LotteryGANV2(nn.Module):
    """
    双色球 GAN v2 - 解决 Mode Collapse

    训练策略：
    - G 训练步数是 D 的 2 倍
    - G 学习率比 D 高（2e-4 vs 5e-5）
    - 多样性损失 + 特征匹配损失
    - D 使用 Dropout（防止过拟合压制 G）
    - D 接收 logits 而非 one-hot
    """

    def __init__(
        self,
        noise_dim: int = 128,
        cond_dim: int = 99,
        red_lr: float = 2e-4,
        blue_lr: float = 2e-4,
        d_lr: float = 5e-5,
        device: str = "cpu",
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.device = device

        self.gen_red = RedBallGeneratorV2(noise_dim, cond_dim).to(device)
        self.disc_red = RedDiscriminatorV2(cond_dim).to(device)
        self.gen_blue = BlueBallGeneratorV2(noise_dim, cond_dim).to(device)
        self.disc_blue = BlueDiscriminatorV2(cond_dim).to(device)

        self.opt_g_red = torch.optim.Adam(self.gen_red.parameters(), lr=red_lr, betas=(0.5, 0.9))
        self.opt_d_red = torch.optim.Adam(self.disc_red.parameters(), lr=d_lr, betas=(0.5, 0.9))
        self.opt_g_blue = torch.optim.Adam(self.gen_blue.parameters(), lr=blue_lr, betas=(0.5, 0.9))
        self.opt_d_blue = torch.optim.Adam(self.disc_blue.parameters(), lr=d_lr, betas=(0.5, 0.9))

        self.to(device)

    def generate(self, batch_size: int, cond: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成一批红球+蓝球号码（batch_size >= 2 以避免 BatchNorm 问题）"""
        if batch_size < 2:
            batch_size = 2
            cond = cond.expand(batch_size, -1)
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        red_indices = self.gen_red(z, cond, temperature)
        blue_idx = self.gen_blue(z, cond, temperature)
        return red_indices, blue_idx

    def generate_logits(self, batch_size: int, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成 logits（用于特征匹配）"""
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        red_logits = self.gen_red.forward_logits(z, cond)
        blue_logits = self.gen_blue.forward_logits(z, cond)
        return red_logits, blue_logits

    def real_to_probs(self, real_red_idx: torch.Tensor, real_blue_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将真实索引转为概率向量（每个位置 one-hot 然后平均）"""
        batch_size, n_balls = real_red_idx.shape
        device = real_red_idx.device
        
        idx_exp = real_red_idx.unsqueeze(-1)
        oh = torch.zeros(batch_size, n_balls, 33, device=device)
        oh.scatter_(2, idx_exp, 1.0)
        red_probs = oh.mean(dim=1)  # (batch, 33)

        oh_blue = torch.zeros(batch_size, 16, device=device)
        oh_blue.scatter_(1, real_blue_idx.unsqueeze(1), 1.0)
        blue_probs = oh_blue

        return red_probs, blue_probs

    def diversity_loss(self, red_logits: torch.Tensor) -> torch.Tensor:
        """鼓励 logits 熵高（不要太集中）"""
        probs = torch.softmax(red_logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return -(entropy.mean() * 0.01)

    def feature_matching_loss(self, fake_logits: torch.Tensor, real_probs: torch.Tensor) -> torch.Tensor:
        """生成分布均值接近真实分布均值"""
        fake_probs = torch.softmax(fake_logits, dim=1)
        return F.mse_loss(fake_probs.mean(dim=0), real_probs.mean(dim=0))

    def gradient_penalty_red(self, real_probs: torch.Tensor, fake_logits: torch.Tensor, cond: torch.Tensor):
        batch_size = real_probs.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        real_p = torch.softmax(real_probs, dim=1)
        fake_p = torch.softmax(fake_logits, dim=1)
        interp = (alpha * real_p + (1 - alpha) * fake_p).requires_grad_(True)
        d_interp = self.disc_red(interp, cond)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, allow_unused=True,
        )[0]
        if grads is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def gradient_penalty_blue(self, real_probs: torch.Tensor, fake_logits: torch.Tensor, cond: torch.Tensor):
        batch_size = real_probs.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        real_p = torch.softmax(real_probs, dim=1)
        fake_p = torch.softmax(fake_logits, dim=1)
        interp = (alpha * real_p + (1 - alpha) * fake_p).requires_grad_(True)
        d_interp = self.disc_blue(interp, cond)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, allow_unused=True,
        )[0]
        if grads is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def train_step_red(
        self,
        real_red_idx: torch.Tensor,
        cond: torch.Tensor,
        real_red_probs: torch.Tensor,
        gp_lambda: float = 10.0,
        d_steps: int = 1,
        g_steps: int = 2,
    ):
        batch_size = real_red_idx.size(0)
        loss_d_item, loss_g_item = 0.0, 0.0

        # ---- 训练判别器 ----
        for _ in range(d_steps):
            self.opt_d_red.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_logits = self.gen_red.forward_logits(z, cond).detach()
            d_real = self.disc_red(real_red_probs, cond)
            d_fake = self.disc_red(fake_logits, cond)
            gp = self.gradient_penalty_red(real_red_probs, fake_logits, cond)
            loss_d = -(d_real.mean() - d_fake.mean()) + gp_lambda * gp
            loss_d.backward()
            self.opt_d_red.step()
            loss_d_item = loss_d.item()

        # ---- 训练生成器 ----
        for _ in range(g_steps):
            self.opt_g_red.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_logits = self.gen_red.forward_logits(z, cond)
            fake_idx = self.gen_red(z, cond)

            d_fake = self.disc_red(fake_logits, cond)
            loss_g_gan = -d_fake.mean()
            loss_g_div = self.diversity_loss(fake_logits)
            loss_g_fm = self.feature_matching_loss(fake_logits, real_red_probs)
            loss_g = loss_g_gan + loss_g_div + loss_g_fm
            loss_g.backward()
            self.opt_g_red.step()
            loss_g_item = loss_g.item()

        return loss_d_item, loss_g_item

    def train_step_blue(
        self,
        real_blue_idx: torch.Tensor,
        cond: torch.Tensor,
        real_blue_probs: torch.Tensor,
        gp_lambda: float = 10.0,
        d_steps: int = 1,
        g_steps: int = 2,
    ):
        batch_size = real_blue_idx.size(0)
        loss_d_item, loss_g_item = 0.0, 0.0

        for _ in range(d_steps):
            self.opt_d_blue.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_logits = self.gen_blue.forward_logits(z, cond).detach()
            d_real = self.disc_blue(real_blue_probs, cond)
            d_fake = self.disc_blue(fake_logits, cond)
            gp = self.gradient_penalty_blue(real_blue_probs, fake_logits, cond)
            loss_d = -(d_real.mean() - d_fake.mean()) + gp_lambda * gp
            loss_d.backward()
            self.opt_d_blue.step()
            loss_d_item = loss_d.item()

        for _ in range(g_steps):
            self.opt_g_blue.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_logits = self.gen_blue.forward_logits(z, cond)
            d_fake = self.disc_blue(fake_logits, cond)
            loss_g_gan = -d_fake.mean()
            loss_g_div = self.diversity_loss(torch.zeros(batch_size, 33, device=self.device))
            loss_g_fm = self.feature_matching_loss(fake_logits, real_blue_probs)
            loss_g = loss_g_gan + loss_g_div + loss_g_fm
            loss_g.backward()
            self.opt_g_blue.step()
            loss_g_item = loss_g.item()

        return loss_d_item, loss_g_item

    def save(self, path_prefix: str):
        torch.save({
            'gen_red': self.gen_red.state_dict(),
            'disc_red': self.disc_red.state_dict(),
            'gen_blue': self.gen_blue.state_dict(),
            'disc_blue': self.disc_blue.state_dict(),
        }, f"{path_prefix}_gan.pt")
        torch.save({
            'opt_g_red': self.opt_g_red.state_dict(),
            'opt_d_red': self.opt_d_red.state_dict(),
            'opt_g_blue': self.opt_g_blue.state_dict(),
            'opt_d_blue': self.opt_d_blue.state_dict(),
        }, f"{path_prefix}_optimizer.pt")

    def load(self, path_prefix: str):
        ckpt = torch.load(f"{path_prefix}_gan.pt", map_location=self.device, weights_only=False)
        self.gen_red.load_state_dict(ckpt['gen_red'])
        self.disc_red.load_state_dict(ckpt['disc_red'])
        self.gen_blue.load_state_dict(ckpt['gen_blue'])
        self.disc_blue.load_state_dict(ckpt['disc_blue'])
        opt_ckpt = torch.load(f"{path_prefix}_optimizer.pt", map_location=self.device, weights_only=False)
        self.opt_g_red.load_state_dict(opt_ckpt['opt_g_red'])
        self.opt_d_red.load_state_dict(opt_ckpt['opt_d_red'])
        self.opt_g_blue.load_state_dict(opt_ckpt['opt_g_blue'])
        self.opt_d_blue.load_state_dict(opt_ckpt['opt_d_blue'])
