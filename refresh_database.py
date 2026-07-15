"""
CSV数据自动刷新到MySQL数据库
"""
import csv
import pymysql
from datetime import datetime
import os

# ============== 配置区域 ==============
CSV_FILE = r"D:\Mydevelopment\MultiContentProject\Mylottery\lottery_data_from_web.csv"

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'reven@0504',
    'database': 'lottery_db',
    'charset': 'utf8mb4'
}


# =====================================

def create_table_if_not_exists(cursor):
    """创建表（如果不存在）"""
    sql = """
    CREATE TABLE IF NOT EXISTS lottery_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        period VARCHAR(10) NOT NULL UNIQUE COMMENT '期号',
        red1 TINYINT UNSIGNED NOT NULL COMMENT '红球1',
        red2 TINYINT UNSIGNED NOT NULL COMMENT '红球2',
        red3 TINYINT UNSIGNED NOT NULL COMMENT '红球3',
        red4 TINYINT UNSIGNED NOT NULL COMMENT '红球4',
        red5 TINYINT UNSIGNED NOT NULL COMMENT '红球5',
        red6 TINYINT UNSIGNED NOT NULL COMMENT '红球6',
        blue TINYINT UNSIGNED NOT NULL COMMENT '蓝球',
        jackpot DECIMAL(15,2) DEFAULT 0 COMMENT '奖池',
        first_prize_count INT DEFAULT 0 COMMENT '一等奖注数',
        first_prize_amount DECIMAL(15,2) DEFAULT 0 COMMENT '一等奖金额',
        second_prize_count INT DEFAULT 0 COMMENT '二等奖注数',
        second_prize_amount DECIMAL(15,2) DEFAULT 0 COMMENT '二等奖金额',
        total_bet_amount DECIMAL(15,2) DEFAULT 0 COMMENT '总投注额',
        draw_date DATE COMMENT '开奖日期',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_period (period),
        INDEX idx_draw_date (draw_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='双色球开奖数据';
    """

    try:
        cursor.execute(sql)
        print("✅ 表 lottery_data 已准备就绪")
        return True
    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        return False


def read_csv(file_path):
    """读取CSV文件"""
    data_list = []

    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None

        print(f"📖 正在读取CSV文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
            # 尝试读取表头
            sample = f.read(500)
            f.seek(0)

            # 检测是否有表头
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            print(f"📋 CSV字段: {fieldnames}")

            # 读取数据
            for row in reader:
                try:
                    # 跳过空行或无效数据
                    if not row.get('period'):
                        continue

                    # 转换数据类型
                    data = {
                        'period': str(row.get('period', '')).strip().zfill(6),
                        'red1': int(row.get('red1', 0)),
                        'red2': int(row.get('red2', 0)),
                        'red3': int(row.get('red3', 0)),
                        'red4': int(row.get('red4', 0)),
                        'red5': int(row.get('red5', 0)),
                        'red6': int(row.get('red6', 0)),
                        'blue': int(row.get('blue', 0)),
                        'jackpot': float(row.get('jackpot', 0)) if row.get('jackpot') else 0,
                        'first_prize_count': int(row.get('first_prize_count', 0)) if row.get(
                            'first_prize_count') else 0,
                        'first_prize_amount': float(row.get('first_prize_amount', 0)) if row.get(
                            'first_prize_amount') else 0,
                        'second_prize_count': int(row.get('second_prize_count', 0)) if row.get(
                            'second_prize_count') else 0,
                        'second_prize_amount': float(row.get('second_prize_amount', 0)) if row.get(
                            'second_prize_amount') else 0,
                        'total_bet_amount': float(row.get('total_bet_amount', 0)) if row.get('total_bet_amount') else 0,
                        'draw_date': parse_date(row.get('draw_date', ''))
                    }

                    data_list.append(data)

                except Exception as e:
                    continue

        print(f"✅ 成功读取 {len(data_list)} 条数据")

        if data_list:
            print("\n📊 数据预览 (前3条):")
            for row in data_list[:3]:
                print(
                    f"   {row['period']}: {row['red1']:02d}-{row['red2']:02d}-{row['red3']:02d}-{row['red4']:02d}-{row['red5']:02d}-{row['red6']:02d} | {row['draw_date']}")

        return data_list

    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return None


def parse_date(date_str):
    """解析日期"""
    if not date_str:
        return None

    date_str = str(date_str).strip()

    # 已经是标准格式
    if len(date_str) == 10 and '-' in date_str:
        return date_str

    formats = ['%Y/%m/%d', '%Y年%m月%d日', '%m/%d/%Y', '%d/%m/%Y']

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except:
            continue

    return date_str


def sync_to_database(data_list, db_config):
    """同步数据到数据库"""
    if not data_list:
        print("❌ 没有数据可同步")
        return 0

    try:
        print("\n🔄 正在连接数据库...")
        conn = pymysql.connect(**db_config, autocommit=False)
        cursor = conn.cursor()

        # 创建表
        if not create_table_if_not_exists(cursor):
            return 0

        # 插入/更新SQL
        sql = """
        INSERT INTO lottery_data 
        (period, red1, red2, red3, red4, red5, red6, blue, 
         jackpot, first_prize_count, first_prize_amount, 
         second_prize_count, second_prize_amount, total_bet_amount, draw_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        red1 = VALUES(red1),
        red2 = VALUES(red2),
        red3 = VALUES(red3),
        red4 = VALUES(red4),
        red5 = VALUES(red5),
        red6 = VALUES(red6),
        blue = VALUES(blue),
        jackpot = VALUES(jackpot),
        first_prize_count = VALUES(first_prize_count),
        first_prize_amount = VALUES(first_prize_amount),
        second_prize_count = VALUES(second_prize_count),
        second_prize_amount = VALUES(second_prize_amount),
        total_bet_amount = VALUES(total_bet_amount),
        draw_date = VALUES(draw_date)
        """

        # 统计
        inserted = 0
        updated = 0
        errors = 0

        print("📊 正在同步数据到数据库...")

        for row in data_list:
            try:
                values = (
                    row['period'],
                    row['red1'], row['red2'], row['red3'],
                    row['red4'], row['red5'], row['red6'],
                    row['blue'],
                    row['jackpot'],
                    row['first_prize_count'],
                    row['first_prize_amount'],
                    row['second_prize_count'],
                    row['second_prize_amount'],
                    row['total_bet_amount'],
                    row['draw_date']
                )

                cursor.execute(sql, values)

                # 检查是插入还是更新
                if cursor.lastrowid:
                    inserted += 1
                else:
                    updated += 1

            except pymysql.err.IntegrityError:
                # 唯一键冲突，说明是更新
                updated += 1
            except Exception as e:
                errors += 1
                if errors <= 5:  # 只显示前5个错误
                    print(f"   ⚠️ 错误 [{row['period']}]: {e}")
                continue

        # 提交事务
        conn.commit()

        # 关闭连接
        cursor.close()
        conn.close()

        # 统计数据库中的总数据
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM lottery_data")
        total_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        print("\n" + "=" * 50)
        print("✅ 数据库同步完成!")
        print("=" * 50)
        print(f"📊 新增: {inserted} 条")
        print(f"📊 更新: {updated} 条")
        print(f"📊 失败: {errors} 条")
        print(f"📊 数据库总记录: {total_count} 条")
        print("=" * 50)

        return inserted + updated

    except Exception as e:
        print(f"❌ 数据库错误: {e}")
        return 0


def main():
    """主函数"""
    print("=" * 70)
    print("📥 CSV数据刷新到MySQL数据库")
    print("=" * 70)
    print(f"📄 CSV文件: {CSV_FILE}")
    print("=" * 70)

    # 1. 读取CSV
    print("\n📖 步骤1: 读取CSV文件")
    data_list = read_csv(CSV_FILE)

    if data_list is None:
        return

    # 2. 同步到数据库
    print("\n🔄 步骤2: 同步到数据库")
    sync_to_database(data_list, DB_CONFIG)


if __name__ == "__main__":
    main()
