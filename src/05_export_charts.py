import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import font_manager
from pathlib import Path

stats_path = Path("data/outputs/traffic_stats.csv")
output_dir = Path("data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

def configure_matplotlib():
    warnings.filterwarnings(
        "ignore",
        message=r"Glyph .* missing from font\(s\).*",
        category=UserWarning,
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [
        "PingFang SC",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    fonts = [f for f in preferred if f in available]
    if fonts:
        plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib()

def save_empty_charts():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('实时交通流量趋势与热度分析（无数据）', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('车流量 (辆/分钟)', fontsize=12)
    ax.text(0.5, 0.5, '暂无可用统计数据', transform=ax.transAxes, ha='center', va='center', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'traffic_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('各交通状态持续时间统计（无数据）', fontsize=14, fontweight='bold')
    ax.set_xlabel('交通状态', fontsize=12)
    ax.set_ylabel('时间窗口数量', fontsize=12)
    ax.text(0.5, 0.5, '暂无可用统计数据', transform=ax.transAxes, ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'status_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
if not stats_path.exists():
    print("缺少 data/outputs/traffic_stats.csv，请先运行 traffic_status 脚本。")
    save_empty_charts()
    raise SystemExit(0)

# 读取数据
stats = pd.read_csv(stats_path)
use_seconds_axis = "time_window_s" in stats.columns and pd.api.types.is_numeric_dtype(stats["time_window_s"])
if use_seconds_axis:
    stats["x_min"] = stats["time_window_s"].astype(float) / 60.0
else:
    stats['time_window'] = pd.to_datetime(stats['time_window'], errors="coerce")
    stats = stats.dropna(subset=["time_window"])
if stats.empty:
    print("traffic_stats.csv 为空，已跳过图表导出。")
    save_empty_charts()
    raise SystemExit(0)

# 创建趋势图
fig, ax = plt.subplots(figsize=(12, 6))

if use_seconds_axis:
    x = stats["x_min"]
else:
    x = stats["time_window"]

ax.plot(x, stats['vehicle_count'], marker='o', linewidth=2, markersize=6, color='#2E86AB')

# 添加阈值线
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='畅通阈值')
ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='拥堵阈值')

# 填充不同状态区域
ax.fill_between(x, 0, 5, alpha=0.2, color='green', label='畅通区间')
ax.fill_between(x, 5, 15, alpha=0.2, color='yellow', label='一般区间')
ax.fill_between(x, 15, stats['vehicle_count'].max() + 5, alpha=0.2, color='red', label='拥堵区间')

# 格式化
ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('车流量 (辆/分钟)', fontsize=12)
ax.set_title('实时交通流量趋势与热度分析', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

if use_seconds_axis:
    def fmt_mmss(x_val, _pos):
        total_s = int(round(x_val * 60))
        mm = total_s // 60
        ss = total_s % 60
        return f"{mm:02d}:{ss:02d}"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_mmss))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=False))
    plt.xticks(rotation=45)
else:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'traffic_trend.png', dpi=300, bbox_inches='tight')
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
plt.savefig(output_dir / 'status_distribution.png', dpi=300, bbox_inches='tight')

print("图表已生成：")
print("- traffic_trend.png: 流量趋势图")
print("- status_distribution.png: 状态分布图")
