#!/bin/bash
# Mylottery VPS 部署安装脚本
# 运行方式: bash setup_vps.sh

set -e

echo "=========================================="
echo "Mylottery VPS 部署脚本"
echo "=========================================="

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检测操作系统
if [ -f /etc/rocky-release ] || [ -f /etc/centos-release ]; then
    echo -e "${GREEN}✓ 检测到 Rocky/CentOS 系统${NC}"
    PKG_MANAGER="yum"
elif [ -f /etc/debian_version ]; then
    echo -e "${GREEN}✓ 检测到 Debian/Ubuntu 系统${NC}"
    PKG_MANAGER="apt-get"
else
    echo -e "${YELLOW}⚠ 未识别的操作系统，使用 yum${NC}"
    PKG_MANAGER="yum"
fi

# 1. 安装 Python3 pip
echo -e "\n[1/5] 安装 Python3 pip..."
$PKG_MANAGER install -y python3-pip python3-venv

# 2. 安装 MySQL/MariaDB
echo -e "\n[2/5] 安装 MariaDB..."
$PKG_MANAGER install -y mariadb-server mariadb-devel

# 3. 启动 MariaDB
echo -e "\n[3/5] 启动 MariaDB..."
systemctl enable mariadb
systemctl start mariadb

# 4. 配置 MySQL root 密码并创建数据库
echo -e "\n[4/5] 配置数据库..."
mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED BY 'Lottery2026!';"
mysql -e "CREATE DATABASE IF NOT EXISTS lottery_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
mysql -e "FLUSH PRIVILEGES;"

# 5. 安装 Python 依赖
echo -e "\n[5/5] 安装 Python 依赖..."
pip3 install --upgrade pip
pip3 install pymysql pandas numpy scipy matplotlib seaborn requests beautifulsoup4 pillow python-dotenv sqlalchemy

echo -e "\n=========================================="
echo -e "${GREEN}✅ 安装完成！${NC}"
echo "=========================================="
echo ""
echo "数据库信息:"
echo "  Host: localhost"
echo "  Port: 3306"
echo "  User: root"
echo "  Password: Lottery2026!"
echo "  Database: lottery_db"
echo ""
echo "下一步:"
echo "  1. 初始化数据库: python3 init_db.py"
echo "  2. 运行爬虫: python3 lottery_spider_v2.py"
echo "  3. 生成预测: python3 lottery_dantuo_prediction_v2.py"
echo ""
echo "设置定时任务 (每天自动更新):"
echo "  crontab -e"
echo "  添加: 0 20 * * * cd /home/clawd/Mylottery && python3 lottery_spider_v2.py >> spider_cron.log 2>&1"
echo "=========================================="
