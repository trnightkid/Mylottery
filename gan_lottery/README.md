# GAN 彩票预测模型

基于 WGAN-GP 条件生成对抗网络的双色球预测，包含数据预处理、模型训练、回测验证、预测输出完整流程。

## 目录结构

```
gan_lottery/
├── gan_model.py      # GAN 核心架构（WGAN-GP，红球+蓝球独立GAN）
├── dataset.py        # 数据加载、特征工程、条件向量构建
├── train.py          # 训练脚本（断点续训、损失曲线）
├── backtest.py       # 回测脚本（200期回测、各类准确率指标）
├── predict.py        # 预测脚本（生成推荐号码）
├── run_pipeline.py   # 一键运行完整流程
└── requirements.txt  # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载历史数据到项目根目录

```bash
# 在 Mylottery 仓库根目录运行
# 或直接把 lottery_data.csv 复制到 gan_lottery/ 同级目录
```

### 3. 一键运行（训练 → 回测 → 预测）

```bash
python gan_lottery/run_pipeline.py --csv ../lottery_data.csv
```

## 分步运行

### 训练
```bash
python gan_lottery/train.py \
  --csv lottery_data.csv \
  --epochs 300 \
  --batch_size 64 \
  --test_periods 200 \
  --save_dir output
```

### 回测
```bash
python gan_lottery/backtest.py \
  --csv lottery_data.csv \
  --model_prefix output/lottery_gan \
  --test_periods 200 \
  --n_candidates 50 \
  --top_k 5
```

### 预测下一期
```bash
python gan_lottery/predict.py \
  --csv lottery_data.csv \
  --model_prefix output/lottery_gan \
  --n 5
```

## 核心指标说明

| 指标 | 含义 |
|------|------|
| 红球命中率 | 预测的6个红球中中了几个（随机期望 1.82/6） |
| 蓝球命中率 | 蓝球是否命中（随机期望 6.25%） |
| hit_3+ rate | 红球命中≥3期的比例（重要参考） |
| 综合准确率 | 红球≥3命中 或 蓝球命中（主要目标） |

## 关于 80% 准确率目标

彩票是随机事件，真正的随机基准：
- 红球33选6，随机期望命中 6/33 ≈ **18.2%**
- 蓝球16选1，随机期望命中率 **6.25%**

如果 GAN 能把红球平均命中从 1.82 提升到 3+（50%+），就已经是重大提升。

## 与现有系统结合

GAN 生成的候选号码可以再用你原有的系统做二次筛选：
```
GAN 生成50个候选
  ↓
graph_predictor.py PageRank 打分
  ↓
predict_v4.py 蒙特卡洛 + 胆拖优化
  ↓
最终推荐
```
