"""
数据集加载和预处理
把 CSV 历史数据转成 GAN 训练格式，
构建条件向量（频率/遗漏/近期活跃度），
输出红球、蓝球、条件的 Tensor
"""

import csv
import math
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


# 双色球参数
RED_COUNT = 33      # 红球 1-33
RED_PICK = 6        # 选6个
BLUE_COUNT = 16     # 蓝球 1-16


class LotteryDataset:
    """
    历史开奖数据集
    支持：频率统计、遗漏值计算、条件向量构建
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.records = []      # List[dict]，每期开奖记录
        self.red_freq = np.zeros(RED_COUNT, dtype=np.float32)   # 累计频率
        self.blue_freq = np.zeros(BLUE_COUNT, dtype=np.float32)
        self.red_miss = np.zeros(RED_COUNT, dtype=np.float32)   # 当前遗漏值
        self.blue_miss = np.zeros(BLUE_COUNT, dtype=np.float32)

        self._load_csv()
        self._compute_base_stats()

    def _load_csv(self):
        """加载 CSV"""
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    period = int(row['period'])
                    red = [int(row[f'red{i}']) for i in range(1, 7)]
                    blue = int(row['blue'])
                    self.records.append({
                        'period': period,
                        'red': red,
                        'blue': blue,
                    })
                except (KeyError, ValueError):
                    continue
        # 按期号升序
        self.records.sort(key=lambda x: x['period'])

    def _compute_base_stats(self):
        """计算全局频率"""
        for rec in self.records:
            for r in rec['red']:
                self.red_freq[r - 1] += 1
            self.blue_freq[rec['blue'] - 1] += 1

        total = len(self.records)
        self.red_freq /= max(total, 1)
        self.blue_freq /= max(total, 1)

    def get_condition_vector(self, period_idx: int, window: int = 200) -> np.ndarray:
        """
        为指定期号构建条件向量（共 97 维）

        条件向量构成：
        - [0:33]    红球频率（近 window 期，归一化到和=1）
        - [33:66]   红球遗漏值（当期遗漏/窗口长度）
        - [66:82]   蓝球频率（归一化到和=1）
        - [82:98]   蓝球遗漏值
        - [98]      热号比例（近期10期出现在高频区的比例）
        - [99]      冷号比例（近期10期出现在低频区的比例）
        - [100]     平均连号数
        总: 101 维，取前 99 维
        """
        end_idx = period_idx
        start_idx = max(0, end_idx - window)
        window_records = self.records[start_idx:end_idx]

        # 默认向量（全0）
        default_cond = np.zeros(101, dtype=np.float32)

        if len(window_records) == 0:
            return default_cond[:99]

        # ---- 红球频率（归一化到和=1）----
        rf = np.zeros(RED_COUNT, dtype=np.float32)
        for rec in window_records:
            for r in rec['red']:
                rf[r - 1] += 1
        rf_sum = rf.sum()
        if rf_sum > 0:
            rf = rf / rf_sum

        # ---- 红球遗漏值（当期遗漏/窗口长度）----
        # 对每期计算遗漏（当期索引 - 上次出现索引）
        last_appear = {i: -1 for i in range(RED_COUNT)}
        for j, rec in enumerate(self.records[start_idx:end_idx]):
            for r in rec['red']:
                last_appear[r - 1] = j

        rm = np.zeros(RED_COUNT, dtype=np.float32)
        window_len = max(len(window_records), 1)
        for i in range(RED_COUNT):
            if last_appear[i] == -1:
                rm[i] = 1.0  # 从未出现
            else:
                miss = (end_idx - start_idx - 1) - last_appear[i]
                rm[i] = miss / window_len

        # ---- 蓝球频率 ----
        bf = np.zeros(BLUE_COUNT, dtype=np.float32)
        for rec in window_records:
            bf[rec['blue'] - 1] += 1
        bf_sum = bf.sum()
        if bf_sum > 0:
            bf = bf / bf_sum

        # ---- 蓝球遗漏值 ----
        last_blue = {i: -1 for i in range(BLUE_COUNT)}
        for j, rec in enumerate(self.records[start_idx:end_idx]):
            last_blue[rec['blue'] - 1] = j

        bm = np.zeros(BLUE_COUNT, dtype=np.float32)
        for i in range(BLUE_COUNT):
            if last_blue[i] == -1:
                bm[i] = 1.0
            else:
                miss = (end_idx - start_idx - 1) - last_blue[i]
                bm[i] = miss / window_len

        # ---- 近期模式特征（3维）----
        recent_10 = self.records[max(0, end_idx - 10):end_idx]
        hot_count = 0
        cold_count = 0
        consecutive = 0

        for rec in recent_10:
            reds = sorted(rec['red'])
            for r in reds:
                if self.red_freq[r - 1] > 0.05:
                    hot_count += 1
            for r in reds:
                if self.red_freq[r - 1] < 0.02:
                    cold_count += 1
            for i in range(len(reds) - 1):
                if reds[i + 1] - reds[i] == 1:
                    consecutive += 1

        hot_ratio = hot_count / max(len(recent_10) * 6, 1)
        cold_ratio = cold_count / max(len(recent_10) * 6, 1)
        avg_consecutive = consecutive / max(len(recent_10), 1)

        # ---- 拼接成条件向量 ----
        cond = np.concatenate([
            rf,           # 33维 红球频率
            rm,           # 33维 红球遗漏
            bf,           # 16维 蓝球频率
            bm,           # 16维 蓝球遗漏
            np.array([hot_ratio, cold_ratio, avg_consecutive], dtype=np.float32),  # 3维
        ])  # 共 101 维

        return cond[:99]  # 取前99维以保持一致

    def get_training_data(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取训练数据（start 到 end 期，不含 end）

        Returns:
            red_tensor: (N, 6) 红球索引，0-indexed
            blue_tensor: (N,)  蓝球索引，0-indexed
            cond_tensor: (N, 99) 条件向量
        """
        records = self.records[start:end]
        n = len(records)
        reds = np.zeros((n, 6), dtype=np.int64)
        blues = np.zeros(n, dtype=np.int64)
        conds = []

        for i, rec in enumerate(records):
            start_idx = start + i
            cond = self.get_condition_vector(start_idx, window=200)
            reds[i] = [r - 1 for r in rec['red']]  # 转0索引
            blues[i] = rec['blue'] - 1
            conds.append(cond)

        red_tensor = torch.from_numpy(reds).long()
        blue_tensor = torch.from_numpy(blues).long()
        cond_tensor = torch.from_numpy(np.stack(conds)).float()

        return red_tensor, blue_tensor, cond_tensor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        cond = self.get_condition_vector(idx + 1, window=200)
        return {
            'red': [r - 1 for r in rec['red']],
            'blue': rec['blue'] - 1,
            'cond': cond,
            'period': rec['period'],
        }


def prepare_backtest_data(csv_path: str, test_periods: int = 200) -> Tuple:
    """
    准备回测数据：
    - 训练集：所有数据减去 test_periods
    - 测试集：最后 test_periods 期

    Returns:
        train_data: (train_red, train_blue, train_cond)
        test_data: (test_red, test_blue, test_cond)
        dataset: LotteryDataset 实例
    """
    dataset = LotteryDataset(csv_path)
    total = len(dataset)
    train_end = total - test_periods

    if train_end <= 0:
        raise ValueError(f"数据量不足，当前 {total} 期，需要至少 {test_periods + 1} 期")

    train_red, train_blue, train_cond = dataset.get_training_data(0, train_end)
    test_red, test_blue, test_cond = dataset.get_training_data(train_end, total)

    print(f"总数据: {total} 期")
    print(f"训练集: {train_end} 期 ({dataset.records[0]['period']} ~ {dataset.records[train_end-1]['period']})")
    print(f"测试集: {test_periods} 期 ({dataset.records[train_end]['period']} ~ {dataset.records[-1]['period']})")

    return (train_red, train_blue, train_cond), (test_red, test_blue, test_cond), dataset
