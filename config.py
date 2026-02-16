import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """配置类"""

    # MySQL数据库配置
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'Mylottery')

    # 表名配置
    TABLE_NAME = 'lottery_data'

    # 红球/蓝球范围
    RED_BALL_RANGE = (1, 33)
    BLUE_BALL_RANGE = (1, 16)

    # 预测配置
    RECENT_WEIGHT = 0.7  # 近期数据权重
    HISTORY_WEIGHT = 0.3  # 历史数据权重

    # 输出路径
    OUTPUT_DIR = 'output'