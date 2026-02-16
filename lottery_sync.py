"""
CSV数据导入MySQL工具
功能：将 lottery_data_from_web.csv 导入到数据库
"""
import csv
import pymysql
from datetime import datetime

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

TABLE_NAME = "lottery_db"


# ======================================


def create_table(cursor):
    """创建表"""
    sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        period VARCHAR(10) NOT NULL UNIQUE,
        red1 TINYINT UNSIGNED NOT NULL, red2 TINYINT UNSIGNED NOT NULL,
        red3 TINYINT UNSIGNED NOT NULL, red4 TINYINT UNSIGNED NOT NULL,
        red5 TINYINT UNSIGNED NOT NULL, red6 TINYINT UNSIGNED NOT NULL,
        blue TINYINT UNSIGNED NOT NULL,
        jackpot DECIMAL(15,2) DEFAULT 0, 
        first_prize_count INT DEFAULT 0,
        first_prize_amount DECIMAL(15,2) DEFAULT 0, 
        second_prize_count INT DEFAULT 0,
        second_prize_amount DECIMAL(15,2) DEFAULT 0, 
        total_bet_amount DECIMAL(15,2) DEFAULT 0,
        draw_date DATE, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_period (period)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cursor.execute(sql)
    print(f"   ✅ 表 {TABLE_NAME} 已创建/确认存在")


def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG, autocommit=True)


def read_csv_file(filename):
    """读取CSV文件"""
    print(f"\n📂 读取CSV文件: {filename}")

    data_list = []
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)

            print(f"   CSV列名: {reader.fieldnames}")

            for row in reader:
                try:
                    # 检查必要字段
                    period = row.get('period', '').strip()
                    if not period:
                        continue

                    # 解析红球
                    reds = []
                    for i in range(1, 7):
                        key = f'red{i}'
                        val = row.get(key, '').strip()
                        if val:
                            reds.append(int(val))

                    if len(reds) < 6:
                        continue

                    # 解析蓝球
                    blue = int(row.get('blue', 0)) if row.get('blue') else 0

                    data = {
                        'period': period.zfill(5) if len(period) < 5 else period,
                        'red1': reds[0],
                        'red2': reds[1],
                        'red3': reds[2],
                        'red4': reds[3],
                        'red5': reds[4],
                        'red6': reds[5],
                        'blue': blue,
                        'jackpot': float(row.get('jackpot', 0) or 0),
                        'first_prize_count': int(row.get('first_prize_count', 0) or 0),
                        'first_prize_amount': float(row.get('first_prize_amount', 0) or 0),
                        'second_prize_count': int(row.get('second_prize_count', 0) or 0),
                        'second_prize_amount': float(row.get('second_prize_amount', 0) or 0),
                        'total_bet_amount': float(row.get('total_bet_amount', 0) or 0),
                        'draw_date': row.get('draw_date', '') or datetime.now().strftime('%Y-%m-%d')
                    }
                    data_list.append(data)

                except Exception as e:
                    print(f"   ⚠️ 解析行失败: {e}")
                    continue

        print(f"   ✅ 成功读取 {len(data_list)} 条数据")

        if data_list:
            print(f"\n📊 数据预览 (前3条):")
            for item in data_list[:3]:
                print(f"   {item['period']}: ", end="")
                print(f"{item['red1']:02d} {item['red2']:02d} {item['red3']:02d} "
                      f"{item['red4']:02d} {item['red5']:02d} {item['red6']:02d} | "
                      f"蓝 {item['blue']:02d}")

        return data_list

    except FileNotFoundError:
        print(f"   ❌ 文件不存在: {filename}")
        return []
    except Exception as e:
        print(f"   ❌ 读取CSV失败: {e}")
        return []


def sync_to_database(data_list):
    """同步数据到数据库"""
    if not data_list:
        print("   ⚠️ 没有数据可同步")
        return 0

    print("\n🔄 同步到数据库...")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 创建表
        create_table(cursor)

        # 检查数据库中已有数据
        cursor.execute(f"SELECT MAX(CAST(period AS UNSIGNED)) FROM {TABLE_NAME}")
        db_latest = cursor.fetchone()[0]
        print(f"   📊 数据库最新期号: {db_latest}")

        # 过滤已存在的数据（只导入比数据库更新的数据）
        if db_latest:
            new_data = [row for row in data_list if int(row['period']) > int(db_latest)]
            print(f"   📊 CSV数据: {len(data_list)} 条")
            print(f"   📊 需新增: {len(new_data)} 条")
        else:
            new_data = data_list
            print(f"   📊 数据库为空，导入全部 {len(new_data)} 条")

        if not new_data:
            print("   ✅ 数据已是最新，无需导入")
            cursor.close()
            conn.close()
            return 0

        # 插入数据
        sql = f"""
        INSERT INTO {TABLE_NAME} 
        (period, red1, red2, red3, red4, red5, red6, blue, 
         jackpot, first_prize_count, first_prize_amount, 
         second_prize_count, second_prize_amount, total_bet_amount, draw_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        red1 = VALUES(red1), red2 = VALUES(red2), red3 = VALUES(red3),
        red4 = VALUES(red4), red5 = VALUES(red5), red6 = VALUES(red6),
        blue = VALUES(blue), draw_date = VALUES(draw_date)
        """

        inserted = 0
        for row in new_data:
            try:
                values = (
                    row['period'], row['red1'], row['red2'], row['red3'],
                    row['red4'], row['red5'], row['red6'], row['blue'],
                    row['jackpot'], row['first_prize_count'], row['first_prize_amount'],
                    row['second_prize_count'], row['second_prize_amount'],
                    row['total_bet_amount'], row['draw_date']
                )
                cursor.execute(sql, values)
                inserted += 1
            except Exception as e:
                print(f"   ❌ 插入失败 {row['period']}: {e}")
                continue

        # 获取总数
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        total = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        print(f"\n{'=' * 50}")
        print(f"✅ 数据库同步完成!")
        print(f"   新增: {inserted} 条")
        print(f"   总计: {total} 条")
        print(f"{'=' * 50}")

        return inserted

    except Exception as e:
        print(f"\n❌ 数据库错误: {e}")
        import traceback
        traceback.print_exc()
        return 0


def check_database_status():
    """检查数据库状态"""
    print("\n" + "=" * 50)
    print("📊 数据库状态检查")
    print("=" * 50)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 检查表是否存在
        cursor.execute(f"SHOW TABLES LIKE '{TABLE_NAME}'")
        if cursor.fetchone():
            print(f"   ✅ 表 {TABLE_NAME} 存在")

            # 获取数据量
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            count = cursor.fetchone()[0]
            print(f"   📊 数据量: {count} 条")

            # 获取最新期号
            cursor.execute(f"SELECT MAX(CAST(period AS UNSIGNED)) FROM {TABLE_NAME}")
            latest = cursor.fetchone()[0]
            print(f"   📊 最新期号: {latest}")

            # 获取最早期号
            cursor.execute(f"SELECT MIN(CAST(period AS UNSIGNED)) FROM {TABLE_NAME}")
            earliest = cursor.fetchone()[0]
            print(f"   📊 最早期号: {earliest}")

        else:
            print(f"   ⚠️ 表 {TABLE_NAME} 不存在")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"   ❌ 连接失败: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 CSV数据导入MySQL工具")
    print("=" * 70)
    print(f"📄 CSV文件: {CSV_FILE}")
    print(f"🗄️  数据库: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"📋 数据表: {TABLE_NAME}")
    print("=" * 70)

    start_time = datetime.now()

    # 1. 检查数据库状态
    check_database_status()

    # 2. 读取CSV
    data_list = read_csv_file(CSV_FILE)

    if not data_list:
        print("\n❌ 没有读取到数据，程序退出")
        return

    # 3. 同步到数据库
    sync_to_database(data_list)

    # 4. 最终检查
    check_database_status()

    # 完成
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  耗时: {duration:.2f} 秒")


if __name__ == "__main__":
    main()
