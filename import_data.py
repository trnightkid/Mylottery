import pymysql
import csv
import re
from datetime import datetime

# ============== 配置区域 ==============
CSV_FILE = r"D:\Mydevelopment\MultiContentProject\Mylottery\lottery_data_clean.csv"
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'reven@0504',  # Docker MySQL密码
    'database': 'lottery_db',
    'charset': 'utf8mb4'
}


# =====================================

def parse_date(date_str):
    """解析 YYYY/M/D 或 YYYY/MM/DD 格式的日期"""
    if not date_str:
        return None
    try:
        # 处理中文斜杠格式
        parts = date_str.strip().split('/')
        if len(parts) == 3:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception as e:
        print(f"日期解析错误: {date_str} -> {e}")
    return None


def convert_value(value, field_type):
    """根据字段类型转换值"""
    value = value.strip()
    if value == '' or value is None:
        return None

    try:
        if field_type == 'int':
            return int(value)
        elif field_type == 'bigint':
            return int(value)
        elif field_type == 'date':
            return parse_date(value)
        else:
            return value
    except ValueError:
        return None


print("=" * 70)
print("🚀 双色球数据导入工具")
print("=" * 70)

try:
    print("\n📡 正在连接数据库...")
    conn = pymysql.connect(**DB_CONFIG, autocommit=False)
    cursor = conn.cursor()
    print("   ✅ 数据库连接成功")

    print(f"\n📖 正在读取CSV文件: {CSV_FILE}")
    insert_count = 0
    skip_count = 0

    with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 跳过表头

        print(f"   📋 CSV表头: {headers}")
        print(f"   📊 字段数量: {len(headers)}")

        # 定义字段类型映射
        field_types = [
            'int',  # period
            'int',  # red1
            'int',  # red2
            'int',  # red3
            'int',  # red4
            'int',  # red5
            'int',  # red6
            'int',  # blue
            'bigint',  # jackpot
            'int',  # first_prize_count
            'bigint',  # first_prize_amount
            'int',  # second_prize_count
            'bigint',  # second_prize_amount
            'bigint',  # total_bet_amount
            'date'  # draw_date
        ]

        sql = """
        INSERT INTO Mylottery 
        (period, red1, red2, red3, red4, red5, red6, blue, 
         jackpot, first_prize_count, first_prize_amount, 
         second_prize_count, second_prize_amount, total_bet_amount, draw_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        batch_data = []
        batch_size = 100

        for i, row in enumerate(reader, 1):
            try:
                # 确保数据完整性
                if len(row) < 15:
                    print(f"   ⚠️ 第{i}行数据不完整，跳过")
                    skip_count += 1
                    continue

                # 转换每行数据
                converted_row = []
                for j, value in enumerate(row[:15]):
                    converted = convert_value(value, field_types[j])
                    converted_row.append(converted)

                # 验证关键字段
                if converted_row[0] is None:  # period为空
                    print(f"   ⚠️ 第{i}行期号为空，跳过")
                    skip_count += 1
                    continue

                batch_data.append(converted_row)

                # 批量插入
                if len(batch_data) >= batch_size:
                    cursor.executemany(sql, batch_data)
                    conn.commit()
                    insert_count += len(batch_data)
                    batch_data = []
                    print(f"   📈 已导入 {insert_count} 行...")

            except Exception as e:
                skip_count += 1
                if skip_count <= 5:
                    print(f"   ❌ 第{i}行出错: {e}")

        # 插入剩余数据
        if batch_data:
            cursor.executemany(sql, batch_data)
            conn.commit()
            insert_count += len(batch_data)

    cursor.close()
    conn.close()

    print("\n" + "=" * 70)
    print("✅ 导入完成！")
    print("=" * 70)
    print(f"   📊 成功导入: {insert_count} 行")
    print(f"   ⚠️ 跳过: {skip_count} 行")
    print("=" * 70)

    # 验证数据
    print("\n📊 数据验证:")
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM Mylottery")
    total = cursor.fetchone()[0]
    print(f"   总记录数: {total}")

    cursor.execute("SELECT MIN(draw_date), MAX(draw_date) FROM My20")
    date_range = cursor.fetchone()
    print(f"   日期范围: {date_range[0]} ~ {date_range[1]}")

    cursor.execute("SELECT period, draw_date FROM Mylottery ORDER BY draw_date DESC LIMIT 5")
    latest = cursor.fetchall()
    print("   最新5期:")
    for row in latest:
        print(f"      {row[0]}期 - {row[1]}")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"\n❌ 错误: {e}")
    print("\n请检查:")
    print("1. Docker MySQL是否正在运行")
    print("2. 数据库密码是否正确")
    print("3. CSV文件路径是否正确")
