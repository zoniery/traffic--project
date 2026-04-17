import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

events_path = Path("data/outputs/events.csv")
if not events_path.exists():
    print("缺少 data/outputs/events.csv，请先运行计数脚本。")
    raise SystemExit(0)

# 读取事件日志
df = pd.read_csv(events_path)

# 转换时间戳（兼容秒数或时间字符串）
if pd.api.types.is_numeric_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
else:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
if df.empty:
    print("events.csv 中没有可用时间数据，已跳过交通状态统计。")
    raise SystemExit(0)

# 按1分钟窗口统计车流量
df['time_window'] = df['timestamp'].dt.floor('1min')  # 1分钟粒度
traffic_stats = df.groupby('time_window').size().reset_index(name='vehicle_count')

# 定义热度阈值（根据实际数据调整）
LOW_THRESHOLD = 5    # 低于5辆/分钟为畅通
HIGH_THRESHOLD = 15  # 高于15辆/分钟为拥堵

def get_status(count):
    if count < LOW_THRESHOLD:
        return '畅通'
    elif count < HIGH_THRESHOLD:
        return '一般'
    else:
        return '拥堵'

traffic_stats['status'] = traffic_stats['vehicle_count'].apply(get_status)

# 保存统计结果
traffic_stats.to_csv('data/outputs/traffic_stats.csv', index=False)

print("交通热度统计：")
print(traffic_stats)
