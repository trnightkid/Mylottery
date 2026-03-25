"""
双色球数据爬虫 v2
数据源：新浪财经 + 备用源聚合
"""
import requests
import csv
import time
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup

# ============== 配置 ==============
OUTPUT_CSV = "/home/clawd/Mylottery/lottery_data.csv"
LAST_PERIOD_FILE = "/home/clawd/Mylottery/.last_period"
LOG_FILE = "/home/clawd/Mylottery/spider.log"

# 新浪财经数据接口
SINA_API = "https://interface.sina.cn/lottery/api/lottery.fcgi"
# 备用：腾讯彩票
TENCENT_API = "https://xw.qq.com/sports/api/lottery/ssq"

# ============== 新浪财经爬虫 ==============
def fetch_from_sina(start_period=None, end_period=None):
    """从新浪财经获取数据"""
    print("📡 正在从新浪财经获取数据...")
    
    all_data = []
    page = 1
    
    while True:
        try:
            # 新浪的双色球历史接口
            url = f"https://datachart.sina.com.cn/ssq/history/newinc/history.php"
            params = {
                'page': page,
                'perPage': 50,
                'callback': 'jsonpCallback'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://finance.sina.com.cn/other/lottery/ssq.html',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            # 尝试解析JSON
            text = response.text
            # 去掉 JSONP 包装
            if 'jsonpCallback' in text:
                text = re.sub(r'jsonpCallback\(|\)', '', text)
            
            data = json.loads(text)
            
            if 'result' not in data or 'data' not in data['result']:
                break
                
            items = data['result']['data']
            if not items:
                break
                
            for item in items:
                period = str(item.get('period', ''))
                if not period.isdigit() or len(period) != 5:
                    continue
                
                # 过滤范围
                if start_period and period < start_period:
                    continue
                if end_period and period > end_period:
                    continue
                
                red_balls = []
                for i in range(1, 7):
                    red = item.get(f'red{i}')
                    if red:
                        red_balls.append(int(red))
                
                blue = item.get('blue')
                if not blue or len(red_balls) != 6:
                    continue
                
                record = {
                    'period': period,
                    'red1': red_balls[0],
                    'red2': red_balls[1],
                    'red3': red_balls[2],
                    'red4': red_balls[3],
                    'red5': red_balls[4],
                    'red6': red_balls[5],
                    'blue': int(blue),
                    'draw_date': item.get('date', ''),
                    'jackpot': item.get('pool', 0),
                    'first_prize_count': item.get('firstPrizeCount', 0),
                    'first_prize_amount': item.get('firstPrizeAmount', 0),
                    'second_prize_count': item.get('secondPrizeCount', 0),
                    'second_prize_amount': item.get('secondPrizeAmount', 0),
                    'total_bet_amount': item.get('totalBetAmount', 0),
                }
                all_data.append(record)
            
            print(f"   第{page}页: 获取{len(items)}条")
            
            # 如果当前页数据少于50条，说明到底了
            if len(items) < 50:
                break
            page += 1
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ 第{page}页失败: {e}")
            break
    
    # 按期号排序
    all_data.sort(key=lambda x: x['period'], reverse=True)
    return all_data


# ============== 腾讯彩票备用源 ==============
def fetch_from_tencent():
    """从腾讯彩票获取数据（备用）"""
    print("📡 正在从腾讯彩票获取数据...")
    
    all_data = []
    page = 1
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://xw.qq.com/sports/lottery/',
        }
        
        # 腾讯体育彩票历史数据
        url = f"https://datachart.qq.com/ssq/history/list"
        params = {'page': page, 'limit': 50}
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        
        if 'data' in data:
            for item in data['data']:
                period = str(item.get('issue', ''))
                if not period.isdigit():
                    continue
                
                reds = item.get('reds', [])
                blue = item.get('blue', 0)
                
                if len(reds) != 6 or not blue:
                    continue
                
                record = {
                    'period': period,
                    'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
                    'red4': reds[3], 'red5': reds[4], 'red6': reds[5],
                    'blue': blue,
                    'draw_date': item.get('date', ''),
                    'jackpot': item.get('pool', 0),
                    'first_prize_count': item.get('first', {}).get('count', 0),
                    'first_prize_amount': item.get('first', {}).get('money', 0),
                    'second_prize_count': item.get('second', {}).get('count', 0),
                    'second_prize_amount': item.get('second', {}).get('money', 0),
                    'total_bet_amount': item.get('total', 0),
                }
                all_data.append(record)
                
    except Exception as e:
        print(f"   ❌ 腾讯源失败: {e}")
    
    all_data.sort(key=lambda x: x['period'], reverse=True)
    return all_data


# ============== 手动更新源（最可靠） ==============
def fetch_latest_from_web手动():
    """通过访问网页手动解析最新数据"""
    print("📡 正在从多个网页源获取最新数据...")
    
    all_data = []
    
    # 源1：必应彩票
    sources = [
        ("必应彩票", "https://www.bingxihui.com/lottery/ssq/history"),
        ("澳客网", "https://www.okooo.com/ssq/history/"),
        ("360彩票", "https://www.360caipiao.com/ssq/history/"),
    ]
    
    for name, url in sources:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找表格
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')[1:][:10]  # 取前10行
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 8:
                        period = cells[0].get_text(strip=True)
                        if period.isdigit() and len(period) == 5:
                            reds = [int(cells[i].get_text(strip=True)) for i in range(1, 7)]
                            blue = int(cells[7].get_text(strip=True))
                            
                            record = {
                                'period': period,
                                'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
                                'red4': reds[3], 'red5': reds[4], 'red6': reds[5],
                                'blue': blue,
                                'draw_date': cells[8].get_text(strip=True) if len(cells) > 8 else '',
                                'jackpot': 0, 'first_prize_count': 0, 'first_prize_amount': 0,
                                'second_prize_count': 0, 'second_prize_amount': 0, 'total_bet_amount': 0,
                            }
                            all_data.append(record)
                print(f"   ✅ {name}: 获取{len(rows)}条")
                break
        except Exception as e:
            print(f"   ⚠️ {name}失败: {e}")
            continue
    
    # 去重
    seen = set()
    unique_data = []
    for d in all_data:
        if d['period'] not in seen:
            seen.add(d['period'])
            unique_data.append(d)
    
    unique_data.sort(key=lambda x: x['period'], reverse=True)
    return unique_data


# ============== CSV操作 ==============
def load_existing_csv(filename):
    """加载已有CSV，返回(数据列表, 最新期号)"""
    if not os.path.exists(filename):
        return [], None
    
    data = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    latest = data[0]['period'] if data else None
    return data, latest


def save_to_csv(data_list, filename):
    """保存到CSV"""
    if not data_list:
        print("   ⚠️ 无数据可保存")
        return 0
    
    fieldnames = ['period', 'red1', 'red2', 'red3', 'red4', 'red5', 'red6',
                  'blue', 'jackpot', 'first_prize_count', 'first_prize_amount',
                  'second_prize_count', 'second_prize_amount', 'total_bet_amount', 'draw_date']
    
    with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)
    
    print(f"   ✅ 保存{len(data_list)}条到 {filename}")
    return len(data_list)


import os

def get_last_period():
    """获取上次爬取的最新期号"""
    if os.path.exists(LAST_PERIOD_FILE):
        with open(LAST_PERIOD_FILE, 'r') as f:
            return f.read().strip()
    return None


def save_last_period(period):
    """保存最新期号"""
    with open(LAST_PERIOD_FILE, 'w') as f:
        f.write(str(period))


def log(msg):
    """写日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {msg}\n")


def merge_and_deduplicate(existing_data, new_data):
    """合并数据并去重"""
    seen = set()
    merged = []
    
    # 先加新数据
    for d in new_data:
        if d['period'] not in seen:
            seen.add(d['period'])
            merged.append(d)
    
    # 再加已有数据
    for d in existing_data:
        if d['period'] not in seen:
            seen.add(d['period'])
            merged.append(d)
    
    # 按期号排序
    merged.sort(key=lambda x: x['period'], reverse=True)
    return merged


def main():
    print("=" * 60)
    print("🚀 双色球数据采集器 v2 (多源聚合)")
    print("=" * 60)
    
    # 1. 加载已有数据
    existing_data, latest_existing = load_existing_csv(OUTPUT_CSV)
    print(f"\n📂 已有数据: {len(existing_data)}条, 最新期号: {latest_existing}")
    
    # 2. 获取上次成功爬取的期号
    last_period = get_last_period()
    print(f"📌 上次爬取期号: {last_period or '首次运行'}")
    
    # 3. 尝试多源获取
    all_new_data = []
    
    # 源1: 新浪财经
    try:
        new_data = fetch_from_sina(start_period=last_period)
        if new_data:
            all_new_data.extend(new_data)
            print(f"   📊 新浪获取{len(new_data)}条新数据")
    except Exception as e:
        print(f"   ❌ 新浪失败: {e}")
    
    time.sleep(1)
    
    # 源2: 腾讯彩票(备用)
    try:
        new_data = fetch_from_tencent()
        if new_data:
            all_new_data.extend(new_data)
            print(f"   📊 腾讯获取{len(new_data)}条")
    except Exception as e:
        print(f"   ⚠️ 腾讯备用源失败: {e}")
    
    # 4. 去重
    if all_new_data:
        all_new_data.sort(key=lambda x: x['period'], reverse=True)
        # 按期号去重，保留第一条（最新的）
        seen = set()
        unique_new = []
        for d in all_new_data:
            if d['period'] not in seen:
                seen.add(d['period'])
                unique_new.append(d)
        all_new_data = unique_new
        
        new_count = len(all_new_data)
        print(f"\n📊 去重后新数据: {new_count}条")
        if all_new_data:
            print(f"   最新期号: {all_new_data[0]['period']}")
        
        # 5. 合并
        merged_data = merge_and_deduplicate(existing_data, all_new_data)
        
        # 6. 保存
        save_to_csv(merged_data, OUTPUT_CSV)
        
        # 7. 更新最新期号
        if all_new_data:
            save_last_period(all_new_data[0]['period'])
            log(f"成功更新{new_count}条数据，最新期号:{all_new_data[0]['period']}")
        
        print(f"\n✅ 完成! 合并后共{len(merged_data)}条数据")
    else:
        print("\n⚠️ 所有源都获取失败，使用已有数据")
        if existing_data:
            save_to_csv(existing_data, OUTPUT_CSV)
            print(f"   保留{len(existing_data)}条已有数据")


if __name__ == "__main__":
    main()
