#!/bin/bash
# Mylottery VPS 一键安装脚本
# 运行方式: (在VPS上) curl -sL https://... | bash 或直接粘贴下面内容

set -e
cd /home/clawd/Mylottery

echo "=========================================="
echo "Mylottery VPS 部署"
echo "=========================================="

# 检查是否是 root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行: sudo bash install.sh"
    exit 1
fi

# 1. 安装 MariaDB
echo "[1/4] 安装 MariaDB..."
yum install -y mariadb-server mariadb > /dev/null 2>&1
systemctl enable mariadb
systemctl start mariadb

# 2. 配置数据库
echo "[2/4] 配置数据库..."
mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED BY 'Lottery2026!';" 2>/dev/null || true
mysql -e "CREATE DATABASE IF NOT EXISTS lottery_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
mysql -e "FLUSH PRIVILEGES;"

# 3. 安装 Python 依赖
echo "[3/4] 安装 Python 依赖..."
pip3 install pymysql pandas numpy scipy matplotlib seaborn requests beautifulsoup4 pillow python-dotenv sqlalchemy --quiet

# 4. 设置 cron 定时任务
echo "[4/4] 设置定时任务..."
(crontab -l 2>/dev/null | grep -v "lottery_spider"; echo "0 20 * * * cd /home/clawd/Mylottery && /usr/bin/python3 lottery_spider_v2.py >> /home/clawd/Mylottery/spider_cron.log 2>&1") | crontab -

echo ""
echo "=========================================="
echo "✅ 部署完成!"
echo "=========================================="
echo ""
echo "数据库密码: Lottery2026!"
echo ""
echo "快速命令:"
echo "  更新数据: python3 lottery_spider_v2.py"
echo "  运行预测: python3 lottery_dantuo_prediction_v2.py"
echo "  图算法预测: python3 graph_predictor.py"
echo "  查看日志: tail -f spider_cron.log"
echo ""
