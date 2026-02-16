"""
快速诊断脚本 - 检查哪个环节出问题
"""
import requests
import re
from bs4 import BeautifulSoup
import json

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
})

print("=" * 60)
print("🔍 诊断开始")
print("=" * 60)

# 测试1: 访问主页面
print("\n1️⃣ 测试访问 500.com/ssq/")
try:
    r = session.get("https://www.500.com/ssq/", timeout=15)
    print(f"   状态码: {r.status_code}")
    print(f"   响应长度: {len(r.text)}")

    # 提取期号
    periods = re.findall(r'(\d{5,6})', r.text)
    valid_periods = [p for p in periods if 3000 <= int(p) <= 300000]
    if valid_periods:
        max_p = max(set(valid_periods), key=lambda x: int(x))
        if len(max_p) == 6:
            max_p = max_p[1:]
        print(f"   页面中最大期号: {max_p}")

except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试2: 访问图表页
print("\n2️⃣ 测试访问图表页")
try:
    r = session.get(
        "https://datachart.500star.com/ssq/history/history.shtml",
        timeout=15
    )
    print(f"   状态码: {r.status_code}")
    print(f"   响应长度: {len(r.text)}")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试3: 测试API
print("\n3️⃣ 测试API接口")
try:
    r = session.get(
        "https://datachart.500star.com/ssq/history/newinc/history.php",
        params={'start': '26000', 'end': '26008'},
        headers={
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://datachart.500star.com/ssq/history/history.shtml',
        },
        timeout=30
    )
    print(f"   状态码: {r.status_code}")
    print(f"   响应长度: {len(r.text)}")
    print(f"   前300字符: {r.text[:300]}")

    # 尝试解析
    try:
        data = json.loads(r.text)
        print(f"   JSON解析成功，类型: {type(data).__name__}")
        if isinstance(data, dict):
            print(f"   键: {list(data.keys())}")
        elif isinstance(data, list):
            print(f"   列表长度: {len(data)}")
            if data:
                print(f"   第一个元素: {data[0]}")
    except json.JSONDecodeError:
        print("   ⚠️ 非JSON格式")
except Exception as e:
    print(f"   ❌ 错误: {e}")

print("\n" + "=" * 60)
print("诊断完成，请把输出结果发给我")
print("=" * 60)
