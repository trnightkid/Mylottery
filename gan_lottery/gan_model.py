"""
GAN 彩票预测模型 - 条件生成对抗网络
双色球：红球33选6，蓝球16选1

Generator: 学习双色球号码的分布规律，生成候选号码
Discriminator: 区分真实开奖号码和生成号码
采用 WGAN-GP 梯度惩罚，训练更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RedBallGenerator(nn.Module):
    """红球生成器：33选6（输出6个互不重复的红球）"""
    def __init__(self, noise_dim: int = 64, cond_dim: int = 99):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 64)
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 33),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, noise_dim) 随机噪声
            cond: (batch, cond_dim) 条件向量
        Returns:
            (batch, 6) 红球索引，0-indexed
        """
        cond64 = self.cond_proj(cond)
        x = torch.cat([z, cond64], dim=1)
        scores = self.fc(x)  # (batch, 33)
        _, indices = torch.topk(scores, 6, dim=1)
        return indices


class BlueBallGenerator(nn.Module):
    """蓝球生成器：16选1"""
    def __init__(self, noise_dim: int = 64, cond_dim: int = 99):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 64)
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond64 = self.cond_proj(cond)
        x = torch.cat([z, cond64], dim=1)
        scores = self.fc(x)  # (batch, 16)
        blue_idx = torch.argmax(scores, dim=1)  # (batch,)
        return blue_idx


class RedDiscriminator(nn.Module):
    """红球判别器：判断6个红球是否像真实开奖"""
    def __init__(self, cond_dim: int = 99):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 64)
        self.fc = nn.Sequential(
            nn.Linear(33 * 6 + 64, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, red: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            red: 红球 - 索引 (batch, 6, dtype=long) 或 one-hot (batch, 198, dtype=float)
            cond: (batch, cond_dim)
        """
        batch_size = red.size(0)
        device = red.device

        if red.dtype == torch.float32 and red.size(1) == 198:
            # 已经是 one-hot (batch, 198)，直接用
            onehot_flat = red
        else:
            # 索引 (batch, 6) -> 转 one-hot (batch, 198)
            red_idx = red.long()
            idx_exp = red_idx.unsqueeze(-1)                           # (batch, 6, 1)
            oh = torch.zeros(batch_size, 6, 33, device=device)
            oh.scatter_(2, idx_exp, 1.0)                               # (batch, 6, 33)
            onehot_flat = oh.view(batch_size, -1)                     # (batch, 198)

        cond64 = self.cond_proj(cond)
        x = torch.cat([onehot_flat, cond64], dim=1)
        return self.fc(x)


class BlueDiscriminator(nn.Module):
    """蓝球判别器：判断1个蓝球是否像真实开奖"""
    def __init__(self, cond_dim: int = 99):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 64)
        self.fc = nn.Sequential(
            nn.Linear(16 + 64, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, blue: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blue: 蓝球索引 (batch,) dtype=long 或 one-hot (batch, 16) dtype=float
            cond: (batch, cond_dim)
        """
        batch_size = blue.size(0)
        device = blue.device

        if blue.dtype == torch.float32 and blue.size(1) == 16:
            oh = blue
        else:
            blue_idx = blue.long() if blue.dtype == torch.long else blue
            oh = torch.zeros(batch_size, 16, device=device)
            oh.scatter_(1, blue_idx.unsqueeze(1), 1.0)

        cond64 = self.cond_proj(cond)
        x = torch.cat([oh, cond64], dim=1)
        return self.fc(x)


class LotteryGAN(nn.Module):
    """
    双色球 GAN（CGAN 条件版本）
    训练目标：min_G max_D E[log D(real)] + E[log(1 - D(G(z)))]
    采用 WGAN-GP 梯度惩罚，训练更稳定
    """

    def __init__(
        self,
        noise_dim: int = 64,
        cond_dim: int = 99,
        red_lr: float = 1e-4,
        blue_lr: float = 1e-4,
        device: str = "cpu",
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.device = device

        self.gen_red = RedBallGenerator(noise_dim, cond_dim).to(device)
        self.disc_red = RedDiscriminator(cond_dim).to(device)
        self.gen_blue = BlueBallGenerator(noise_dim, cond_dim).to(device)
        self.disc_blue = BlueDiscriminator(cond_dim).to(device)

        self.opt_g_red = torch.optim.Adam(self.gen_red.parameters(), lr=red_lr, betas=(0.5, 0.9))
        self.opt_d_red = torch.optim.Adam(self.disc_red.parameters(), lr=red_lr, betas=(0.5, 0.9))
        self.opt_g_blue = torch.optim.Adam(self.gen_blue.parameters(), lr=blue_lr, betas=(0.5, 0.9))
        self.opt_d_blue = torch.optim.Adam(self.disc_blue.parameters(), lr=blue_lr, betas=(0.5, 0.9))

        self.to(device)

    def generate(self, batch_size: int, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成一批红球+蓝球号码"""
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        red_indices = self.gen_red(z, cond)
        blue_idx = self.gen_blue(z, cond)
        return red_indices, blue_idx

    # ---- 辅助方法 ----

    def _indices_to_onehot_red(self, indices: torch.Tensor) -> torch.Tensor:
        """(batch, 6) long -> (batch, 198) float"""
        batch_size = indices.size(0)
        device = indices.device
        idx_exp = indices.unsqueeze(-1).long()
        oh = torch.zeros(batch_size, 6, 33, device=device)
        oh.scatter_(2, idx_exp, 1.0)
        return oh.view(batch_size, -1)

    def _indices_to_onehot_blue(self, indices: torch.Tensor) -> torch.Tensor:
        """(batch,) long -> (batch, 16) float"""
        batch_size = indices.size(0)
        device = indices.device
        oh = torch.zeros(batch_size, 16, device=device)
        oh.scatter_(1, indices.long().unsqueeze(1), 1.0)
        return oh

    # ---- 梯度惩罚（WGAN-GP）----

    def gradient_penalty_red(self, real_red: torch.Tensor, fake_red: torch.Tensor, cond: torch.Tensor):
        """WGAN-GP 红球梯度惩罚，在 one-hot 空间做插值"""
        batch_size = real_red.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)

        real_oh = self._indices_to_onehot_red(real_red)
        fake_oh = self._indices_to_onehot_red(fake_red)

        # one-hot 空间插值
        interp_oh = (alpha * real_oh + (1 - alpha) * fake_oh).requires_grad_(True)
        d_interp = self.disc_red(interp_oh, cond)

        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp_oh,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            allow_unused=True,
        )[0]

        if grads is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        grads_flat = grads.view(batch_size, -1)
        gradient_norm = grads_flat.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

    def gradient_penalty_blue(self, real_blue: torch.Tensor, fake_blue: torch.Tensor, cond: torch.Tensor):
        """WGAN-GP 蓝球梯度惩罚"""
        batch_size = real_blue.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)

        real_oh = self._indices_to_onehot_blue(real_blue)
        fake_oh = self._indices_to_onehot_blue(fake_blue)

        interp_oh = (alpha * real_oh + (1 - alpha) * fake_oh).requires_grad_(True)
        d_interp = self.disc_blue(interp_oh, cond)

        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp_oh,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            allow_unused=True,
        )[0]

        if grads is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        grads_flat = grads.view(batch_size, -1)
        gradient_norm = grads_flat.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

    # ---- 训练步 ----

    def train_step_red(self, real_red: torch.Tensor, cond: torch.Tensor, gp_lambda: float = 10.0):
        """训练红球 GAN 一步"""
        self.opt_d_red.zero_grad()
        z = torch.randn(real_red.size(0), self.noise_dim, device=self.device)
        fake_red = self.gen_red(z, cond).detach()

        d_real = self.disc_red(real_red, cond)
        d_fake = self.disc_red(fake_red, cond)
        gp = self.gradient_penalty_red(real_red, fake_red, cond)
        loss_d = d_fake.mean() - d_real.mean() + gp_lambda * gp
        loss_d.backward()
        self.opt_d_red.step()

        self.opt_g_red.zero_grad()
        z = torch.randn(real_red.size(0), self.noise_dim, device=self.device)
        fake_red = self.gen_red(z, cond)
        d_fake = self.disc_red(fake_red, cond)
        loss_g = -d_fake.mean()
        loss_g.backward()
        self.opt_g_red.step()

        return loss_d.item(), loss_g.item()

    def train_step_blue(self, real_blue: torch.Tensor, cond: torch.Tensor, gp_lambda: float = 10.0):
        """训练蓝球 GAN 一步"""
        self.opt_d_blue.zero_grad()
        z = torch.randn(real_blue.size(0), self.noise_dim, device=self.device)
        fake_blue = self.gen_blue(z, cond).detach()

        d_real = self.disc_blue(real_blue, cond)
        d_fake = self.disc_blue(fake_blue, cond)
        gp = self.gradient_penalty_blue(real_blue, fake_blue, cond)
        loss_d = d_fake.mean() - d_real.mean() + gp_lambda * gp
        loss_d.backward()
        self.opt_d_blue.step()

        self.opt_g_blue.zero_grad()
        z = torch.randn(real_blue.size(0), self.noise_dim, device=self.device)
        fake_blue = self.gen_blue(z, cond)
        d_fake = self.disc_blue(fake_blue, cond)
        loss_g = -d_fake.mean()
        loss_g.backward()
        self.opt_g_blue.step()

        return loss_d.item(), loss_g.item()

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
        ckpt = torch.load(f"{path_prefix}_gan.pt", map_location=self.device)
        self.gen_red.load_state_dict(ckpt['gen_red'])
        self.disc_red.load_state_dict(ckpt['disc_red'])
        self.gen_blue.load_state_dict(ckpt['gen_blue'])
        self.disc_blue.load_state_dict(ckpt['disc_blue'])
        opt_ckpt = torch.load(f"{path_prefix}_optimizer.pt", map_location=self.device)
        self.opt_g_red.load_state_dict(opt_ckpt['opt_g_red'])
        self.opt_d_red.load_state_dict(opt_ckpt['opt_d_red'])
        self.opt_g_blue.load_state_dict(opt_ckpt['opt_g_blue'])
        self.opt_d_blue.load_state_dict(opt_ckpt['opt_d_blue'])
