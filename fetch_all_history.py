#!/usr/bin/env python3
"""
双色球全量历史数据爬虫 - 完整版
抓取 2010年(期号10001) 至今所有数据
"""
import requests, re, csv, time, os
from datetime import datetime

OUTPUT_FILE = '/home/clawd/Mylottery/lottery_data_full.csv'
BATCH = 100  # 每批100期

def fetch_batch(start, end):
    url = f"https://datachart.500star.com/ssq/history/newinc/history.php?start={start}&end={end}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    r = requests.get(url, headers=headers, timeout=20)
    r.encoding = 'gbk'
    return r.text

def parse_batch(html):
    pattern = r'<td>(\d{5})</td><td class="t_cfont2">(\d{2})</td><td class="t_cfont2">(\d{2})</td><td class="t_cfont2">(\d{2})</td><td class="t_cfont2">(\d{2})</td><td class="t_cfont2">(\d{2})</td><td class="t_cfont2">(\d{2})</td><td class="t_cfont4">(\d{2})</td><td class="t_cfont4">&nbsp;</td><td>([\d,]+)</td><td>(\d+)</td><td>[\d,]+</td><td>(\d+)</td><td>[\d,]+</td><td>[\d,]+</td><td>(\d{4}-\d{2}-\d{2})</td>'
    
    matches = re.findall(pattern, html)
    data = []
    for m in matches:
        try:
            period = m[0]
            reds = [int(m[i]) for i in range(1, 7)]
            blue = int(m[7])
            jackpot = int(m[8].replace(',', ''))
            first_count = int(m[9])
            second_count = int(m[10])
            date = m[11]
            
            data.append({
                'period': period,
                'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
                'red4': reds[3], 'red5': reds[4], 'red6': reds[5],
                'blue': blue,
                'jackpot': jackpot,
                'first_prize_count': first_count,
                'first_prize_amount': 0,
                'second_prize_count': second_count,
                'second_prize_amount': 0,
                'total_bet_amount': 0,
                'draw_date': date
            })
        except:
            continue
    return data

def load_existing():
    if not os.path.exists(OUTPUT_FILE):
        return {}
    existing = {}
    with open(OUTPUT_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row['period']] = row
    return existing

def save_all(data_list):
    data_list.sort(key=lambda x: int(x['period']))
    fields = ['period','red1','red2','red3','red4','red5','red6','blue',
               'jackpot','first_prize_count','first_prize_amount',
               'second_prize_count','second_prize_amount','total_bet_amount','draw_date']
    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data_list)
    return len(data_list)

def main():
    print("=" * 60)
    print("🚀 双色球全量历史数据爬虫")
    print("=" * 60)
    
    existing = load_existing()
    print(f"\n已有数据: {len(existing)} 条")
    if existing:
        periods = sorted(existing.keys(), key=int)
        print(f"  范围: {periods[0]} ~ {periods[-1]}")
    
    # 确定范围：10001=2010年, 26030=最新
    START, END = 10001, 26030
    
    all_data = dict(existing)
    total_new = 0
    
    print(f"\n目标: {START} ~ {END} (约 {END-START} 期)")
    print(f"每批: {BATCH} 期\n")
    
    batch_num = 0
    for start in range(START, END + 1, BATCH):
        end = min(start + BATCH - 1, END)
        batch_num += 1
        
        html = fetch_batch(start, end)
        parsed = parse_batch(html)
        
        for row in parsed:
            if row['period'] not in all_data:
                all_data[row['period']] = row
                total_new += 1
        
        last_period = parsed[-1]['period'] if parsed else '?'
        first_period = parsed[0]['period'] if parsed else '?'
        print(f"  批次{batch_num:3d}: {start}-{end} → 获取{len(parsed):3d}条 (范围:{first_period}~{last_period}) 累计:{len(all_data)}条")
        
        time.sleep(0.3)  # 礼貌限速
    
    print(f"\n💾 保存数据...")
    total = save_all(list(all_data.values()))
    
    data_list = list(all_data.values())
    data_list.sort(key=lambda x: int(x['period']))
    
    print(f"  ✅ 总计: {total} 条")
    print(f"  🆕 新增: {total_new} 条")
    print(f"  📅 {data_list[0]['period']} ({data_list[0]['draw_date']}) ~ {data_list[-1]['period']} ({data_list[-1]['draw_date']})")
    print(f"\n📁 保存到: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
