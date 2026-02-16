"""
500彩票网 - 双色球历史数据爬虫
版本：v2.3 (修复HTML解析 + GBK编码)
"""
import requests
import csv
import pymysql
import time
from bs4 import BeautifulSoup

# ============== 配置区域 ==============
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'reven@0504',
    'database': 'lottery_db',
    'charset': 'utf8mb4'
}

OUTPUT_CSV = r"D:\Mydevelopment\MultiContentProject\Mylottery\lottery_data_from_web.csv"
DEBUG_FILE = r"D:\Mydevelopment\MultiContentProject\Mylottery\debug_response.html"

DEFAULT_START_PERIOD = "15001"
DEFAULT_END_PERIOD = "26010"


# =====================================


def get_session():
    """创建Session"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://datachart.500star.com/ssq/history/history.shtml',
    })
    return session


def visit_homepage(session):
    """访问主页获取Cookie"""
    print("🏠 正在访问主页...")
    home_url = "https://datachart.500star.com/ssq/history/history.shtml"
    try:
        response = session.get(home_url, timeout=30)
        response.encoding = 'gbk'  # ← 关键：设置GBK编码
        print(f"   ✅ 主页访问成功")
        return True
    except Exception as e:
        print(f"   ❌ 主页访问失败: {e}")
        return False


def fetch_data(session, start_period, end_period):
    """获取数据"""
    print(f"\n📡 正在获取数据: {start_period} ~ {end_period}")

    url = f"https://datachart.500star.com/ssq/history/newinc/history.php?start={start_period}&end={end_period}"

    try:
        response = session.get(url, timeout=30)
        response.encoding = 'gbk'  # ← 关键：设置GBK编码

        print(f"   响应: {response.status_code}, {len(response.text)} 字符")

        # 保存调试文件
        with open(DEBUG_FILE, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"   💾 调试文件已保存")

        return response.text
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")
        return None


def parse_html(html_content):
    """解析HTML表格数据"""
    print("\n🔍 正在解析HTML...")

    data_list = []

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # 找到数据表格 tbody
        tbody = soup.find('tbody', id='tdata')

        if not tbody:
            print("   ❌ 未找到数据表格")
            return []

        rows = tbody.find_all('tr')
        print(f"   找到 {len(rows)} 行数据")

        for row in rows:
            cells = row.find_all('td')

            if len(cells) < 16:
                continue

            try:
                # 提取数据
                period = cells[0].get_text(strip=True)

                # 验证期号格式
                if not period.isdigit() or len(period) != 5:
                    continue

                # 红球 (列1-6，索引1-6)
                reds = []
                for i in range(1, 7):
                    ball = cells[i].get_text(strip=True).zfill(2)
                    reds.append(int(ball))

                # 蓝球 (列7，索引7)
                blue = int(cells[7].get_text(strip=True).zfill(2))

                # 验证球号范围
                if not all(1 <= r <= 33 for r in reds):
                    continue
                if not (1 <= blue <= 16):
                    continue

                # 其他字段
                draw_date = cells[15].get_text(strip=True)

                # 构建数据行
                data_row = {
                    'period': period,
                    'red1': reds[0],
                    'red2': reds[1],
                    'red3': reds[2],
                    'red4': reds[3],
                    'red5': reds[4],
                    'red6': reds[5],
                    'blue': blue,
                    'jackpot': 0,
                    'first_prize_count': int(cells[10].get_text(strip=True).replace(',', '')) if cells[10].get_text(
                        strip=True) else 0,
                    'first_prize_amount': int(cells[11].get_text(strip=True).replace(',', '')) if cells[11].get_text(
                        strip=True) else 0,
                    'second_prize_count': int(cells[12].get_text(strip=True).replace(',', '')) if cells[12].get_text(
                        strip=True) else 0,
                    'second_prize_amount': int(cells[13].get_text(strip=True).replace(',', '')) if cells[13].get_text(
                        strip=True) else 0,
                    'total_bet_amount': int(cells[14].get_text(strip=True).replace(',', '')) if cells[14].get_text(
                        strip=True) else 0,
                    'draw_date': draw_date
                }

                data_list.append(data_row)

            except Exception as e:
                continue

        print(f"   ✅ 解析成功: {len(data_list)} 条")

    except Exception as e:
        print(f"   ❌ 解析错误: {e}")

    return data_list


def save_to_csv(data_list, filename):
    """保存到CSV"""
    if not data_list:
        print("   ⚠️ 无数据可保存")
        return

    fieldnames = ['period', 'red1', 'red2', 'red3', 'red4', 'red5', 'red6',
                  'blue', 'jackpot', 'first_prize_count', 'first_prize_amount',
                  'second_prize_count', 'second_prize_amount', 'total_bet_amount', 'draw_date']

    with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)

    print(f"   ✅ CSV已保存: {filename}")


def save_to_database(data_list, db_config):
    """保存到数据库"""
    if not data_list:
        print("   ⚠️ 无数据可保存")
        return 0

    try:
        conn = pymysql.connect(**db_config, autocommit=False)
        cursor = conn.cursor()

        sql = """INSERT INTO lottery_data
        (period, red1, red2, red3, red4, red5, red6, blue, jackpot,
         first_prize_count, first_prize_amount, second_prize_count,
         second_prize_amount, total_bet_amount, draw_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        red1=VALUES(red1), red2=VALUES(red2), red3=VALUES(red3),
        red4=VALUES(red4), red5=VALUES(red5), red6=VALUES(red6),
        blue=VALUES(blue), draw_date=VALUES(draw_date)"""

        count = 0
        for row in data_list:
            try:
                cursor.execute(sql, (
                    row['period'], row['red1'], row['red2'], row['red3'],
                    row['red4'], row['red5'], row['red6'], row['blue'],
                    row['jackpot'], row['first_prize_count'], row['first_prize_amount'],
                    row['second_prize_count'], row['second_prize_amount'],
                    row['total_bet_amount'], row['draw_date']
                ))
                count += 1
            except Exception as e:
                print(f"   插入失败: {row['period']} - {e}")
                continue

        conn.commit()
        cursor.close()
        conn.close()

        print(f"   ✅ 数据库保存成功: {count} 条")
        return count

    except Exception as e:
        print(f"   ❌ 数据库错误: {e}")
        return 0


def main():
    print("=" * 70)
    print("🚀 500彩票网 - 双色球历史数据采集器 (v2.3)")
    print("=" * 70)

    # 检查数据库
    print("\n📊 检查数据库...")
    try:
        conn = pymysql.connect(**DB_CONFIG, autocommit=False)
        cursor = conn.cursor()
        cursor.execute("SELECT period FROM lottery_data ORDER BY CAST(period AS UNSIGNED) DESC LIMIT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            print(f"   数据库最新期数: {result[0]}")
        else:
            print("   ⚠️ 数据库暂无数据")
    except Exception as e:
        print(f"   ⚠️ 数据库连接失败: {e}")

    # 创建Session并访问主页
    session = get_session()
    visit_homepage(session)

    # 确认采集范围
    print("\n" + "=" * 50)
    print("📝 确认采集范围")
    print("=" * 50)
    print(f"   起始期数: {DEFAULT_START_PERIOD}")
    print(f"   结束期数: {DEFAULT_END_PERIOD}")

    confirm = input("   确认开始采集? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 延时后获取数据
    time.sleep(1)
    html_content = fetch_data(session, DEFAULT_START_PERIOD, DEFAULT_END_PERIOD)

    if not html_content:
        print("\n❌ 获取数据失败")
        return

    # 解析数据
    data_list = parse_html(html_content)

    if not data_list:
        print("\n❌ 解析数据失败，请查看调试文件")
        return

    # 保存数据
    print("\n💾 正在保存数据...")
    save_to_csv(data_list, OUTPUT_CSV)
    save_to_database(data_list, DB_CONFIG)

    print(f"\n✅ 采集完成! 共获取 {len(data_list)} 条记录")
    print(f"   最新期号: {data_list[0]['period']}" if data_list else "")


if __name__ == "__main__":
    main()
