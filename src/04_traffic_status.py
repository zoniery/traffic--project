import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

events_path = Path("data/outputs/events.csv")
output_path = Path("data/outputs/traffic_stats.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
if not events_path.exists():
    print("缺少 data/outputs/events.csv，请先运行计数脚本。")
    pd.DataFrame(columns=["time_window_s", "vehicle_count", "status"]).to_csv(output_path, index=False)
    raise SystemExit(0)

# 读取事件日志
df = pd.read_csv(events_path)

time_s = None
if "time_s" in df.columns and pd.api.types.is_numeric_dtype(df["time_s"]):
    time_s = df["time_s"].astype(float)
elif "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
    time_s = df["timestamp"].astype(float)
elif "timestamp" in df.columns:
    parsed = pd.to_datetime(df["timestamp"], errors="coerce")
    parsed = parsed.dropna()
    if not parsed.empty:
        time_s = (parsed - parsed.min()).dt.total_seconds()

if time_s is None or time_s.empty:
    print("events.csv 中没有可用时间数据，已跳过交通状态统计。")
    pd.DataFrame(columns=["time_window_s", "vehicle_count", "status"]).to_csv(output_path, index=False)
    raise SystemExit(0)

df = df.loc[time_s.index].copy()
df["time_s"] = time_s
df["time_window_s"] = (df["time_s"] // 60).astype(int) * 60
traffic_stats = df.groupby("time_window_s").size().reset_index(name="vehicle_count")

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
traffic_stats.to_csv(output_path, index=False)

print("交通热度统计：")
print(traffic_stats)
