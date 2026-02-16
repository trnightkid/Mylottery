import csv

input_file = r"D:\Mydevelopment\MultiContentProject\Mylottery\lottery_data.csv"
output_file = (r"D:\Mydevelopment\MultiContentProject\Mylottery\lottery_data_clean.csv")

print("正在清理CSV文件...")

with open(input_file, 'r', encoding='utf-8-sig') as infile, \
        open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 写入表头
    header = next(reader)
    writer.writerow(header)

    skip_count = 0
    write_count = 0

    for i, row in enumerate(reader, 2):
        # 检查period是否为空
        if len(row) > 0 and row[0].strip() != '':
            writer.writerow(row)
            write_count += 1
        else:
            skip_count += 1
            print(f"跳过第{i}行: period为空")

print()
print("=" * 50)
print(f"✅ 清理完成！")
print(f"   保留行数: {write_count}")
print(f"   跳过行数: {skip_count}")
print(f"   输出文件: {output_file}")
print("=" * 50)
