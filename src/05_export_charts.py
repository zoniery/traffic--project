import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

stats_path = Path("data/outputs/traffic_stats.csv")
if not stats_path.exists():
    print("缺少 data/outputs/traffic_stats.csv，请先运行 traffic_status 脚本。")
    raise SystemExit(0)

# 读取数据
stats = pd.read_csv(stats_path)
stats['time_window'] = pd.to_datetime(stats['time_window'])
if stats.empty:
    print("traffic_stats.csv 为空，已跳过图表导出。")
    raise SystemExit(0)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建趋势图
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(stats['time_window'], stats['vehicle_count'], 
        marker='o', linewidth=2, markersize=6, color='#2E86AB')

# 添加阈值线
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='畅通阈值')
ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='拥堵阈值')

# 填充不同状态区域
ax.fill_between(stats['time_window'], 0, 5, alpha=0.2, color='green', label='畅通区间')
ax.fill_between(stats['time_window'], 5, 15, alpha=0.2, color='yellow', label='一般区间')
ax.fill_between(stats['time_window'], 15, stats['vehicle_count'].max() + 5, 
                alpha=0.2, color='red', label='拥堵区间')

# 格式化
ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('车流量 (辆/分钟)', fontsize=12)
ax.set_title('实时交通流量趋势与热度分析', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# 格式化X轴时间显示
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('data/outputs/traffic_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# 生成柱状图
fig, ax = plt.subplots(figsize=(10, 6))
status_counts = stats['status'].value_counts()
colors = {'畅通': '#2ECC71', '一般': '#F39C12', '拥堵': '#E74C3C'}
bars = ax.bar(status_counts.index, status_counts.values, 
              color=[colors.get(x, 'gray') for x in status_counts.index])

ax.set_xlabel('交通状态', fontsize=12)
ax.set_ylabel('时间窗口数量', fontsize=12)
ax.set_title('各交通状态持续时间统计', fontsize=14, fontweight='bold')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('data/outputs/status_distribution.png', dpi=300, bbox_inches='tight')

print("图表已生成：")
print("- traffic_trend.png: 流量趋势图")
print("- status_distribution.png: 状态分布图")
