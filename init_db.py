"""
数据库初始化脚本 - 创建正确的表结构
"""
import pymysql

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'reven@0504',
    'database': 'lottery_db',
    'charset': 'utf8mb4'
}

# 创建表的SQL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS lottery_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    period VARCHAR(10) NOT NULL UNIQUE COMMENT '期号',
    red1 INT NOT NULL COMMENT '红球1',
    red2 INT NOT NULL COMMENT '红球2',
    red3 INT NOT NULL COMMENT '红球3',
    red4 INT NOT NULL COMMENT '红球4',
    red5 INT NOT NULL COMMENT '红球5',
    red6 INT NOT NULL COMMENT '红球6',
    blue INT NOT NULL COMMENT '蓝球',
    jackpot BIGINT DEFAULT 0 COMMENT '奖池奖金',
    first_prize_count INT DEFAULT 0 COMMENT '一等奖注数',
    first_prize_amount BIGINT DEFAULT 0 COMMENT '一等奖奖金',
    second_prize_count INT DEFAULT 0 COMMENT '二等奖注数',
    second_prize_amount BIGINT DEFAULT 0 COMMENT '二等奖奖金',
    total_bet_amount BIGINT DEFAULT 0 COMMENT '总投注额',
    draw_date DATE COMMENT '开奖日期',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_period (period),
    INDEX idx_draw_date (draw_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='双色球开奖数据';
"""

# 删除旧表的SQL（如果结构完全错误）
DROP_TABLE_SQL = "DROP TABLE IF EXISTS lottery_data;"


def init_database():
    print("=" * 60)
    print("🗄️ 数据库初始化")
    print("=" * 60)

    try:
        # 连接数据库
        print("\n🔌 正在连接数据库...")
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            charset=DB_CONFIG['charset']
        )
        print("   ✅ 连接成功")

        cursor = conn.cursor()

        # 检查数据库是否存在
        cursor.execute("SHOW DATABASES LIKE 'lottery_db'")
        if not cursor.fetchone():
            print("   📦 创建数据库 lottery_db...")
            cursor.execute("CREATE DATABASE lottery_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print("   ✅ 数据库创建成功")

        # 选择数据库
        cursor.execute("USE lottery_db")

        # 检查表是否存在
        cursor.execute("SHOW TABLES LIKE 'lottery_data'")
        if cursor.fetchone():
            print("   ℹ️ 表 lottery_data 已存在")

            # 检查表结构
            cursor.execute("DESCRIBE lottery_data")
            columns = [row[0] for row in cursor.fetchall()]
            print(f"   当前列: {columns}")

            if 'period' not in columns:
                print("   ⚠️ 表结构不正确，需要重建")
                print("   🗑️ 删除旧表...")
                cursor.execute(DROP_TABLE_SQL)
                print("   ✅ 旧表已删除")
            else:
                print("   ✅ 表结构正确")
                cursor.close()
                conn.close()
                return True

        # 创建新表
        print("\n   🆕 创建新表 lottery_data...")
        cursor.execute(CREATE_TABLE_SQL)
        print("   ✅ 表创建成功")

        # 验证
        cursor.execute("DESCRIBE lottery_data")
        columns = cursor.fetchall()
        print("\n   📋 表结构:")
        for col in columns:
            print(f"      {col[0]:20} {col[1]:15} {col[2]}")

        cursor.close()
        conn.close()

        print("\n" + "=" * 60)
        print("✅ 数据库初始化完成！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False


if __name__ == "__main__":
    init_database()
