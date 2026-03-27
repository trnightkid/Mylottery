import csv
import os

# 中文到英文列名映射
COLUMN_MAP = {
    '期号': 'period',
    '红球1': 'red1', '红球2': 'red2', '红球3': 'red3',
    '红球4': 'red4', '红球5': 'red5', '红球6': 'red6',
    '蓝球': 'blue',
    '奖池奖金': 'jackpot',
    '一等奖注数': 'first_prize_count',
    '一等奖奖金': 'first_prize_amount',
    '二等奖注数': 'second_prize_count',
    '二等奖奖金': 'second_prize_amount',
    '总投注额': 'total_bet_amount',
    '开奖日期': 'draw_date'
}

# 英文到中文列名映射（输出用）
REVERSE_MAP = {v: k for k, v in COLUMN_MAP.items()}

def fix_csv(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + '.new'
    
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        original_columns = reader.fieldnames
        rows = list(reader)
    
    # 检查是否已经是英文列名
    if 'period' in original_columns:
        print(f'已经是英文列名，无需转换')
        return
    
    print(f'原始列名: {original_columns}')
    
    # 转换
    new_rows = []
    for row in rows:
        new_row = {COLUMN_MAP.get(k, k): v for k, v in row.items()}
        new_rows.append(new_row)
    
    # 写出
    fieldnames = list(COLUMN_MAP.values())
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    
    print(f'转换完成: {len(new_rows)}条 -> {output_file}')
    
    # 备份原文件
    os.rename(input_file, input_file + '.bak')
    os.rename(output_file, input_file)
    print(f'已替换原文件，原文件备份为 {input_file}.bak')

fix_csv('lottery_data.csv')
