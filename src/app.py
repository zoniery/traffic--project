import os
import tempfile

os.environ.setdefault("STREAMLIT_HOME", os.path.join(tempfile.gettempdir(), "streamlit"))
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib
matplotlib.use("Agg", force=True)
from ultralytics import YOLO
import supervision as sv
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib import font_manager

try:
    from src.counting_core import (
        MIN_RECOMMEND_SCORE,
        recommend_counting_line as recommend_counting_line_core,
        update_crossing_counts as update_crossing_counts_core,
    )
except Exception:
    from counting_core import (
        MIN_RECOMMEND_SCORE,
        recommend_counting_line as recommend_counting_line_core,
        update_crossing_counts as update_crossing_counts_core,
    )

OUTPUT_DIR = "data/outputs"
EVENTS_PATH = os.path.join(OUTPUT_DIR, "events.csv")
VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "processed_video.mp4")
CHART_PATH = os.path.join(OUTPUT_DIR, "traffic_trend.png")

MODEL_CANDIDATES = [
    ("yolov8n.pt", "轻量快速"),
    ("yolov8s.pt", "均衡"),
    ("yolov8m.pt", "精确"),
]

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


def find_local_model(model_name: str) -> str | None:
    candidates = [
        os.path.join(os.getcwd(), model_name),
        os.path.join(os.path.dirname(__file__), model_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def recommend_counting_line(video_path: str, model_path: str, max_frames: int = 200):
    return recommend_counting_line_core(video_path, model_path, max_frames=max_frames)


def draw_detected_boxes(frame, boxes):
    annotated = frame.copy()
    if boxes.xyxy is None:
        return annotated

    classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
    names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    for index, xyxy in enumerate(boxes.xyxy.cpu().tolist()):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        class_id = classes[index] if index < len(classes) else -1
        track_text = f" #{track_ids[index]}" if index < len(track_ids) else ""
        label = f"{names.get(class_id, 'vehicle')}{track_text}"
        foot_x = int((x1 + x2) / 2)
        foot_y = y2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.circle(annotated, (foot_x, foot_y), 5, (0, 255, 255), -1)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2,
        )

    return annotated


def update_crossing_counts(
    track_last_seen,
    track_last_side,
    track_last_count,
    frame_index,
    track_ids,
    boxes_xyxy,
    orientation,
    line_value,
    time_s,
    margin,
    ttl=90,
    cooldown=3,
    band_box_scale=0.12,
    band_margin_scale=0.8,
):
    return update_crossing_counts_core(
        track_last_seen,
        track_last_side,
        track_last_count,
        frame_index,
        track_ids,
        boxes_xyxy,
        orientation,
        line_value,
        time_s,
        margin,
        ttl=ttl,
        cooldown=cooldown,
        band_box_scale=band_box_scale,
        band_margin_scale=band_margin_scale,
    )

def process_video(video_path, line_pos, orientation, model_name, ttl=90, cooldown=3, band_box_scale=0.12, band_margin_scale=0.8):
    """
    处理视频的完整流程
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(model_name)
    tracker = sv.ByteTrack()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件，请确认上传文件有效。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("视频元信息读取失败（fps/宽高无效）。")

    fourcc_options = [
        cv2.VideoWriter_fourcc(*'mp4v'),
        cv2.VideoWriter_fourcc(*'avc1'),
        cv2.VideoWriter_fourcc(*'XVID')
    ]

    out = None
    for fourcc in fourcc_options:
        candidate = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))
        if candidate.isOpened():
            out = candidate
            break
        candidate.release()

    if out is None:
        cap.release()
        raise RuntimeError("无法创建输出视频，请检查系统视频编码支持。")

    if orientation == "horizontal":
        line_value = int(height * line_pos / 100)
        margin = max(5, height // 100)
    else:
        line_value = int(width * line_pos / 100)
        margin = max(5, width // 100)

    events = []
    count_in = 0
    count_out = 0
    frame_idx = 0
    track_last_seen = {}
    track_last_side = {}
    track_last_count = {}
    unique_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[2, 3, 5, 7], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            unique_ids.update(detections.tracker_id.tolist())

        time_s = frame_idx / fps

        new_in, new_out, new_events = update_crossing_counts(
            track_last_seen,
            track_last_side,
            track_last_count,
            frame_idx,
            detections.tracker_id.tolist() if detections.tracker_id is not None else [],
            detections.xyxy.tolist() if detections.xyxy is not None else [],
            orientation,
            line_value,
            time_s,
            margin,
            ttl=ttl,
            cooldown=cooldown,
            band_box_scale=band_box_scale,
            band_margin_scale=band_margin_scale,
        )
        count_in += new_in
        count_out += new_out
        events.extend(new_events)

        annotated = draw_detected_boxes(frame, results[0].boxes)
        if orientation == "horizontal":
            cv2.line(annotated, (0, line_value), (width, line_value), (0, 255, 255), 2)
            tip = "Above->Below: In, Below->Above: Out"
            tip_pos = (10, max(25, line_value - 10))
        else:
            cv2.line(annotated, (line_value, 0), (line_value, height), (0, 255, 255), 2)
            tip = "Left->Right: In, Right->Left: Out"
            tip_pos = (max(10, line_value + 10), 25)

        cv2.putText(
            annotated,
            tip,
            tip_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            f'In: {count_in} | Out: {count_out}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()

    events_df = pd.DataFrame(events)
    if events_df.empty:
        events_df = pd.DataFrame(columns=["time_s", "direction", "count_delta"])
    events_df.to_csv(EVENTS_PATH, index=False)
    generate_charts(events_df)
    return {"in_count": count_in, "out_count": count_out, "event_count": len(events_df), "unique_vehicles": len(unique_ids)}

def generate_charts(events_df=None):
    """生成统计图表"""
    if events_df is None:
        if not os.path.exists(EVENTS_PATH):
            return
        events_df = pd.read_csv(EVENTS_PATH)

    if events_df.empty:
        return

    df = events_df.copy()
    if "time_s" in df.columns and pd.api.types.is_numeric_dtype(df["time_s"]):
        df["minute"] = (df["time_s"] // 60).astype(int)
        stats = df.groupby("minute").size().reset_index(name="count")
        stats["minute_label"] = stats["minute"].apply(lambda m: f"{m:02d}:00")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return
        df['minute'] = df['timestamp'].dt.floor('1min')
        stats = df.groupby('minute').size().reset_index(name='count')

    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        if "minute_label" in stats.columns:
            ax.plot(stats["minute"], stats["count"], marker="o")
            ax.set_xticks(stats["minute"].tolist())
            ax.set_xticklabels(stats["minute_label"].tolist(), rotation=45)
        else:
            ax.plot(stats['minute'], stats['count'], marker='o')
        ax.set_title('车流量趋势')
        ax.set_xlabel('时间')
        ax.set_ylabel('车辆数')
        plt.tight_layout()
        plt.savefig(CHART_PATH)
    except Exception:
        return
    finally:
        plt.close()

def main():
    try:
        from src.web_app import main as ui_main
    except Exception:
        from web_app import main as ui_main
    ui_main()


if __name__ == "__main__":
    main()
