# 双色球预测分析系统

基于历史数据分析的双色球彩票预测工具，采用统计学方法和蒙特卡洛采样算法，对双色球历史开奖数据进行深度分析，生成预测推荐号码。

## 功能特点

- ✅ 历史数据分析（频率、热号、冷号、遗漏值、连号等）
- ✅ 图算法增强预测（PageRank + 共现网络）
- ✅ 时间加权多窗口分析
- ✅ 胆拖投注优化策略
- ✅ 数据可视化图表
- ✅ GUI图形界面

---

## 目录结构

```
Mylottery/
├── main.py                    # 命令行主程序入口
├── lottery_ui_v1.2.py         # GUI主程序（Tkinter桌面应用）
├── graph_predictor.py          # 图算法分析模块（PageRank + 共现网络）
│
├── fetch_history_v3.py        # 数据爬虫（500star.com，支持BeautifulSoup）
├── predict_v4.py              # 预测核心模块（时间加权优化版）
│
├── config.py                  # 配置文件
├── init_db.py                  # 数据库初始化
├── import_data.py             # 数据导入工具
│
├── output/                     # 输出目录
│   ├── predictions_v4.json    # 预测结果
│   └── *.png                  # 可视化图表
│
├── lottery_data.csv           # 历史开奖数据
├── .env                       # 环境变量配置
└── mysql_table_schema.sql     # 数据库表结构
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy matplotlib requests beautifulsoup4 pillow python-dotenv
```

### 2. 爬取最新数据

```bash
# 检查最新期号
python fetch_history_v3.py --check

# 爬取最新数据
python fetch_history_v3.py --latest

# 全量爬取（首次运行）
python fetch_history_v3.py --start 10001 --end 26032
```

### 3. 运行预测

```bash
python predict_v4.py
```

预测结果保存在 `output/predictions_v4.json`

### 4. GUI模式

```bash
python lottery_ui_v1.2.py
```

---

## 算法说明

### 预测 v4 - 时间加权优化版

采用多维度综合评分体系：

| 因素 | 权重 | 说明 |
|------|------|------|
| 热号/冷号得分 | 50% | 多窗口频率分析 |
| PageRank | 25% | 号码网络核心度 |
| 区域分布 | 15% | 号码分区统计 |
| 遗漏加成 | 10% | 冷号回补建模 |

### 时间加权

| 周期 | 权重 |
|------|------|
| 短期 (近50期) | 60% |
| 中期 (51-200期) | 30% |
| 长期 (200期以上) | 10% |

### 图算法 (graph_predictor.py)

- **PageRank 算法**：分析号码在共现网络中的核心程度
- **号码共现矩阵**：统计号码同时出现的频率
- **聚类分析**：发现号码的热区冷区分布

### 蒙特卡洛采样

- 500次随机采样
- 精英选择策略
- 胆拖投注优化

---

## 数据说明

- **数据来源**：500star.com
- **历史覆盖**：2011年 - 2026年
- **数据量级**：2200+ 期
- **期号范围**：11071 ~ 26032

---

## 最近更新

### 2026-03-26 v2.1.0

**新增功能：**
- `fetch_history_v3.py` - 全新爬虫，支持 BeautifulSoup 和纯正则双模式
- `predict_v4.py` - 时间加权优化版预测算法

**算法改进：**
- 多窗口频率分析（50/200/1000期）
- 时间衰减权重
- 遗漏值动态加成
- 自适应区域分布

**删除旧文件：**
- `predict.py` → 由 `predict_v4.py` 替代
- `lottery_dantuo_prediction_v2.py` → 由 `predict_v4.py` 替代
- `fetch_all_history.py` → 由 `fetch_history_v3.py` 替代
- `Lottery_spider.py` → 由 `fetch_history_v3.py` 替代

### 历史版本

- **v2.0.0** - 初始版本，双色球历史数据分析，蒙特卡洛采样，GUI界面

---

## 注意事项

1. 彩票是随机事件，预测结果仅供参考
2. 请理性购彩，切勿沉迷
3. 本系统不保证预测准确性
4. 数据仅供娱乐研究使用

---

## 问题排查

**Q: 爬虫失败怎么办？**
```bash
# 检查最新期号
python fetch_history_v3.py --check

# 指定范围重试
python fetch_history_v3.py --start 26025 --end 26035
```

**Q: 图表中文乱码？**
确保系统已安装中文字体（SimHei / Microsoft YaHei）

**Q: 缺少依赖？**
```bash
pip install pandas numpy matplotlib requests beautifulsoup4
```

---

*最后更新：2026-03-26*
