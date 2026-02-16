import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Dict, Any
from config import Config
from urllib.parse import quote
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """数据库连接器"""

    def __init__(self):
        self.engine = self._create_engine()

    def _create_engine(self):
        """创建数据库连接引擎"""
        # 对密码进行URL编码，防止特殊字符（如@）导致连接失败
        encoded_password = quote(Config.DB_PASSWORD, safe='')
        connection_string = (
            f"mysql+pymysql://{Config.DB_USER}:{encoded_password}@"
            f"{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
        )
        return create_engine(connection_string)

    def load_lottery_data(self) -> pd.DataFrame:
        """加载双色球数据"""
        query = f"""
        SELECT 
            period,
            red1, red2, red3, red4, red5, red6,
            blue,
            jackpot,
            first_prize_count,
            first_prize_amount,
            second_prize_count,
            second_prize_amount,
            total_bet_amount,
            draw_date
        FROM {Config.TABLE_NAME}
        ORDER BY CAST(period AS UNSIGNED)
        """

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"成功加载 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def get_all_red_ball_numbers(self, df: pd.DataFrame) -> pd.Series:
        """获取所有红球号码（展开为一维序列）"""
        red_columns = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
        return df[red_columns].values.flatten()

    def get_blue_ball_numbers(self, df: pd.DataFrame) -> pd.Series:
        """获取所有蓝球号码"""
        return df['blue']