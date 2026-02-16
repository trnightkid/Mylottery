import requests

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://datachart.500star.com/ssq/history/history.shtml',
})

url = "https://datachart.500star.com/ssq/history/newinc/history.php?start=25132&end=26010"

print("=" * 60)
print("🧪 500彩票数据测试")
print("=" * 60)
print(f"URL: {url}")
print("-" * 60)

response = session.get(url, timeout=30)

print(f"状态码: {response.status_code}")
print(f"字符数: {len(response.text)}")
print("-" * 60)

# 显示内容
print("\n📄 响应内容 (前4000字符):\n")
print(response.text[:4000])
print("\n" + "-" * 60)

if len(response.text) > 4000:
    print(f"\n📄 响应内容 (最后1000字符):\n")
    print(response.text[-1000:])

print("\n" + "=" * 60)
