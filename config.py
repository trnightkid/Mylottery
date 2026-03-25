"""
配置文件 - Mylottery
支持环境变量覆盖
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 数据库配置
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'Lottery2026!')
    DB_NAME = os.getenv('DB_NAME', 'lottery_db')
    
    # 预测参数
    RECENT_WEIGHT = 0.7  # 近期数据权重 (0-1)
    
    # 输出路径
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/home/clawd/Mylottery/output')
    PREDICTION_DIR = os.getenv('PREDICTION_DIR', '/home/clawd/Mylottery/dan_tuo_prediction')
    
    @classmethod
    def get_db_config(cls):
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD,
            'database': cls.DB_NAME,
            'charset': 'utf8mb4'
        }
