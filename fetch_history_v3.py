#!/usr/bin/env python3
"""
双色球全量历史数据爬虫 v3 (纯正则版，无需BeautifulSoup)
数据源: 500star.com
"""
import requests
import re
import csv
import time
import os
import argparse

DEFAULT_OUTPUT = 'lottery_data.csv'
REQUEST_DELAY = 0.3


def fetch_page(start, end):
    url = f"https://datachart.500star.com/ssq/history/newinc/history.php?start={start}&end={end}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://datachart.500star.com/ssq/history/',
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.encoding = 'gbk'
        return r.text
    except Exception as e:
        print(f"    ❌ {e}")
        return None


def parse_page(html):
    """解析HTML，提取数据"""
    if not html:
        return []
    
    # 定位到数据区域
    match = re.search(r'<tbody id="tdata">(.*?)</tbody>', html, re.DOTALL)
    if not match:
        return []
    
    tbody = match.group(1)
    
    # 匹配每行数据
    row_pattern = r'<tr[^>]*>(.*?)</tr>'
    rows = re.findall(row_pattern, tbody, re.DOTALL)
    
    data = []
    
    for row_html in rows:
        # 每个cell的内容
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
        
        # 结构: 0=注释, 1=period, 2-7=红球, 8=蓝球, 9=快乐星期天, 10=奖池, 11=一等奖注数, 12=一等奖金额, 13=二等奖注数, 14=二等奖金额, 15=总投注额, 16=日期
        if len(cells) < 16:
            continue
        
        try:
            # 期号 (cells[1])
            period_match = re.search(r'\d{5}', cells[1])
            if not period_match:
                continue
            period = period_match.group()
            
            # 红球 (cells[2:8])
            reds = []
            for cell in cells[2:8]:
                num_match = re.search(r'\d{2}', cell)
                if num_match:
                    reds.append(int(num_match.group()))
            
            if len(reds) != 6:
                continue
            
            # 蓝球 (cells[8])
            blue_match = re.search(r'\d{2}', cells[8])
            blue = int(blue_match.group()) if blue_match else 0
            
            # 奖池 (cells[10])
            jackpot_text = re.sub(r'\D', '', cells[10])
            jackpot = int(jackpot_text) if jackpot_text else 0
            
            # 一等奖注数 (cells[11])
            first_count_text = re.sub(r'\D', '', cells[11])
            first_count = int(first_count_text) if first_count_text else 0
            
            # 一等奖金额 (cells[12])
            first_amount_text = re.sub(r'\D', '', cells[12])
            first_amount = int(first_amount_text) if first_amount_text else 0
            
            # 二等奖注数 (cells[13])
            second_count_text = re.sub(r'\D', '', cells[13])
            second_count = int(second_count_text) if second_count_text else 0
            
            # 二等奖金额 (cells[14])
            second_amount_text = re.sub(r'\D', '', cells[14])
            second_amount = int(second_amount_text) if second_amount_text else 0
            
            # 总投注额 (cells[15])
            total_text = re.sub(r'\D', '', cells[15])
            total = int(total_text) if total_text else 0
            
            # 日期 (cells[16])
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', cells[16])
            date = date_match.group() if date_match else ''
            
            data.append({
                'period': period,
                'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
                'red4': reds[3], 'red5': reds[4], 'red6': reds[5],
                'blue': blue,
                'jackpot': jackpot,
                'first_prize_count': first_count,
                'first_prize_amount': first_amount,
                'second_prize_count': second_count,
                'second_prize_amount': second_amount,
                'total_bet_amount': total,
                'draw_date': date
            })
        except Exception as e:
            continue
    
    return data


def load_existing(filepath):
    if not os.path.exists(filepath):
        return {}, None, None
    existing = {}
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row['period']] = row
        periods = sorted(existing.keys(), key=int)
        return existing, periods[0], periods[-1] if periods else (None, None)
    except:
        return {}, None, None


def save_data(data_list, filepath):
    fields = ['period', 'red1', 'red2', 'red3', 'red4', 'red5', 'red6',
              'blue', 'jackpot', 'first_prize_count', 'first_prize_amount',
              'second_prize_count', 'second_prize_amount', 'total_bet_amount', 'draw_date']
    data_list.sort(key=lambda x: int(x['period']))
    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data_list)
    return len(data_list)


def get_latest():
    html = fetch_page(26000, 99999)
    if html:
        parsed = parse_page(html)
        if parsed:
            return parsed[0]['period']
    return None


def crawl(start_period, end_period, output_file, batch_size=100):
    print("=" * 60)
    print("🚀 双色球全量历史数据爬虫 v3")
    print("=" * 60)
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    existing, existing_start, existing_end = load_existing(output_file)
    print(f"\n📂 已有数据: {len(existing)} 条")
    if existing_start and existing_end:
        print(f"   范围: {existing_start} ~ {existing_end}")
    
    if start_period is None:
        start_period = int(existing_end) + 1 if existing_end else 10001
    if end_period is None:
        end_period = 99999
    
    print(f"\n🎯 目标: {start_period} ~ {end_period}")
    
    all_data = dict(existing)
    total_new = 0
    failed = []
    batch_num = 0
    
    current = start_period
    while current <= end_period:
        batch_end = min(current + batch_size - 1, end_period)
        batch_num += 1
        
        print(f"\n  批次{batch_num:3d}: 爬取 {current} ~ {batch_end} ...", end=' ')
        
        html = fetch_page(current, batch_end)
        
        if html:
            parsed = parse_page(html)
            if parsed:
                new_count = sum(1 for row in parsed if row['period'] not in all_data)
                for row in parsed:
                    all_data[row['period']] = row
                total_new += new_count
                print(f"✅ {len(parsed)} 条 (新增 {new_count})")
                
                # 数据量骤减说明快到头
                if len(parsed) < batch_size * 0.3:
                    print(f"   📍 数据量骤减，可能已到最新期号")
                    break
            else:
                print("⚠️ 解析0条")
                failed.append((current, batch_end))
        else:
            print("❌ 请求失败")
            failed.append((current, batch_end))
        
        current = batch_end + 1
        time.sleep(REQUEST_DELAY)
    
    print(f"\n💾 保存到 {output_file} ...")
    data_list = list(all_data.values())
    data_list.sort(key=lambda x: int(x['period']))
    total = save_data(data_list, output_file)
    
    print(f"\n{'='*60}")
    print(f"📊 完成! 总计: {total} 条, 新增: {total_new} 条")
    if data_list:
        print(f"   范围: {data_list[0]['period']} ({data_list[0]['draw_date']}) ~ {data_list[-1]['period']} ({data_list[-1]['draw_date']})")
    if failed:
        print(f"⚠️ 失败: {failed[:3]}")
    
    return total, total_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--latest', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    
    if args.check:
        print("🔍 最新期号:", get_latest())
    elif args.latest:
        latest = get_latest()
        if latest:
            crawl(int(latest), int(latest), args.output)
    else:
        crawl(args.start, args.end, args.output, args.batch)
