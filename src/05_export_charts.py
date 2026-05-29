import os
from pathlib import Path

import pandas as pd
import tempfile
import warnings
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import font_manager

events_path = Path("data/outputs/events.csv")
output_dir = Path("data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "realtime_flow_1s.csv"
chart_path = output_dir / "realtime_flow_1s.png"

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

cols = ["second", "in_count", "out_count", "total_count", "cum_in", "cum_out", "cum_total"]
if not events_path.exists():
    print("缺少 data/outputs/events.csv，请先运行计数脚本。")
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)
    if chart_path.exists():
        chart_path.unlink()
    raise SystemExit(0)

df = pd.read_csv(events_path)
if df.empty or "time_s" not in df.columns:
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)
    if chart_path.exists():
        chart_path.unlink()
    raise SystemExit(0)

df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
df = df.dropna(subset=["time_s"])
if df.empty:
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)
    if chart_path.exists():
        chart_path.unlink()
    raise SystemExit(0)

df["second"] = df["time_s"].astype(float).apply(lambda v: int(v // 1))
df["direction"] = df["direction"].astype(str).str.lower()
df["count_delta"] = pd.to_numeric(df.get("count_delta", 1), errors="coerce").fillna(1).astype(int)

grouped = df.groupby(["second", "direction"])["count_delta"].sum().unstack(fill_value=0)
in_series = grouped["in"] if "in" in grouped.columns else 0
out_series = grouped["out"] if "out" in grouped.columns else 0

max_second = int(df["second"].max())
index = pd.Index(range(0, max_second + 1), name="second")
in_series = pd.Series(in_series, index=index).fillna(0).astype(int)
out_series = pd.Series(out_series, index=index).fillna(0).astype(int)

table = pd.DataFrame({"second": index})
table["in_count"] = in_series.values
table["out_count"] = out_series.values
table["total_count"] = table["in_count"] + table["out_count"]
table["cum_in"] = table["in_count"].cumsum()
table["cum_out"] = table["out_count"].cumsum()
table["cum_total"] = table["total_count"].cumsum()

table.to_csv(output_path, index=False)

try:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(table["second"], table["total_count"], linewidth=2, color="#2E86AB", label="total/s")
    ax.plot(table["second"], table["in_count"], linewidth=1.5, color="#2ECC71", alpha=0.9, label="in/s")
    ax.plot(table["second"], table["out_count"], linewidth=1.5, color="#E74C3C", alpha=0.9, label="out/s")
    ax.set_title("每秒实时车流量", fontsize=14, fontweight="bold")
    ax.set_xlabel("秒 (s)", fontsize=12)
    ax.set_ylabel("车辆数 (辆/秒)", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=220, bbox_inches="tight")
finally:
    plt.close()

print("实时流量结果已生成：")
print(f"- {output_path}")
print(f"- {chart_path}")
