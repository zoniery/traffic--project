import os
import tempfile

os.environ.setdefault("STREAMLIT_HOME", os.path.join(tempfile.gettempdir(), "streamlit"))
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import cv2
import streamlit as st
from ultralytics import YOLO
import supervision as sv
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib import font_manager

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
        message=r"Glyph .* missing from font\\(s\\).*",
        category=UserWarning,
        module=r"matplotlib.*",
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None

    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    bounds = {}

    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, classes=[2, 3, 5, 7], verbose=False)
        detections = sv.Detections.from_ultralytics(res[0])
        detections = tracker.update_with_detections(detections)
        if detections.tracker_id is None or detections.xyxy is None or len(detections) == 0:
            continue
        ids = detections.tracker_id.tolist()
        xyxy = detections.xyxy.tolist()
        for track_id, b in zip(ids, xyxy):
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            item = bounds.get(track_id)
            if item is None:
                bounds[track_id] = [cx, cx, cy, cy]
            else:
                item[0] = min(item[0], cx)
                item[1] = max(item[1], cx)
                item[2] = min(item[2], cy)
                item[3] = max(item[3], cy)

    cap.release()
    if not bounds:
        return None

    def best_overlap(orientation: str):
        dim = height if orientation == "horizontal" else width
        margin = max(5, dim // 100)
        points = []
        for min_x, max_x, min_y, max_y in bounds.values():
            lo = (min_y + margin) if orientation == "horizontal" else (min_x + margin)
            hi = (max_y - margin) if orientation == "horizontal" else (max_x - margin)
            if lo < hi:
                points.append((lo, 1))
                points.append((hi, -1))
        if not points:
            return (0, dim // 2)
        points.sort(key=lambda x: x[0])
        best_score = 0
        best_pos = dim // 2
        score = 0
        idx = 0
        while idx < len(points):
            pos = points[idx][0]
            while idx < len(points) and points[idx][0] == pos:
                score += points[idx][1]
                idx += 1
            if idx < len(points):
                next_pos = points[idx][0]
                if pos < next_pos and score > best_score:
                    best_score = score
                    best_pos = (pos + next_pos) / 2
        return (best_score, best_pos)

    score_h, pos_h = best_overlap("horizontal")
    score_v, pos_v = best_overlap("vertical")
    if score_v > score_h:
        orientation = "vertical"
        position = int(round(pos_v / width * 100))
        score = score_v
    else:
        orientation = "horizontal"
        position = int(round(pos_h / height * 100))
        score = score_h

    position = max(0, min(100, position))
    return {"score": int(score), "orientation": orientation, "position": position, "tracks": len(bounds)}


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
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
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


def update_crossing_counts(track_sides, track_last_seen, frame_index, track_ids, boxes_xyxy, orientation, line_value, time_s, margin, ttl=30):
    events = []
    new_in = 0
    new_out = 0

    if not track_ids or not boxes_xyxy:
        return new_in, new_out, events
    active_ids = set()

    for track_id, xyxy in zip(track_ids, boxes_xyxy):
        active_ids.add(track_id)
        track_last_seen[track_id] = frame_index
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2

        if orientation == "horizontal":
            if center_y < line_value - margin:
                current_side = "above"
            elif center_y > line_value + margin:
                current_side = "below"
            else:
                current_side = None
        else:
            if center_x < line_value - margin:
                current_side = "left"
            elif center_x > line_value + margin:
                current_side = "right"
            else:
                current_side = None

        previous_side = track_sides.get(track_id)
        if current_side is None:
            continue

        if previous_side == "above" and current_side == "below":
            new_in += 1
            events.append({"time_s": time_s, "direction": "in", "count_delta": 1})
        elif previous_side == "below" and current_side == "above":
            new_out += 1
            events.append({"time_s": time_s, "direction": "out", "count_delta": 1})
        elif previous_side == "left" and current_side == "right":
            new_in += 1
            events.append({"time_s": time_s, "direction": "in", "count_delta": 1})
        elif previous_side == "right" and current_side == "left":
            new_out += 1
            events.append({"time_s": time_s, "direction": "out", "count_delta": 1})

        track_sides[track_id] = current_side

    expired = [track_id for track_id, last_seen in list(track_last_seen.items()) if frame_index - last_seen > ttl]
    for track_id in expired:
        track_last_seen.pop(track_id, None)
        track_sides.pop(track_id, None)

    return new_in, new_out, events

# 页面配置
st.set_page_config(page_title="车流量识别系统", layout="wide")
st.title("🚗 低空车流量实时识别与分析系统")

def process_video(video_path, line_pos, orientation, model_name):
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
    track_sides = {}
    track_last_seen = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[2, 3, 5, 7], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)

        time_s = frame_idx / fps

        new_in, new_out, new_events = update_crossing_counts(
            track_sides,
            track_last_seen,
            frame_idx,
            detections.tracker_id.tolist() if detections.tracker_id is not None else [],
            detections.xyxy.tolist() if detections.xyxy is not None else [],
            orientation,
            line_value,
            time_s,
            margin,
        )
        count_in += new_in
        count_out += new_out
        events.extend(new_events)

        annotated = draw_detected_boxes(frame, results[0].boxes)
        if orientation == "horizontal":
            cv2.line(annotated, (0, line_value), (width, line_value), (0, 255, 255), 2)
            tip = "上->下记为进入，下->上记为离开"
            tip_pos = (10, max(25, line_value - 10))
        else:
            cv2.line(annotated, (line_value, 0), (line_value, height), (0, 255, 255), 2)
            tip = "左->右记为进入，右->左记为离开"
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
    return {"in_count": count_in, "out_count": count_out, "event_count": len(events_df)}

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

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

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
    plt.close()

# 侧边栏
st.sidebar.header("配置选项")
uploaded_file = st.sidebar.file_uploader("上传视频文件", type=['mp4', 'avi', 'mov'])

line_orientation = st.sidebar.radio(
    "计数线方向",
    ["水平线（上下穿越）", "垂直线（左右穿越）"],
    index=0,
    key="line_orientation",
)

line_position_label = "计数线位置（画面高度%）" if line_orientation.startswith("水平") else "计数线位置（画面宽度%）"
line_position = st.sidebar.slider(line_position_label, 0, 100, 60, key="line_position")
show_preview = st.sidebar.checkbox("显示计数线预览", value=True, key="show_preview")
auto_use_recommended = st.sidebar.checkbox("开始分析前自动推荐并使用计数线", value=True, key="auto_use_recommended")

available_models = []
model_desc = {}
missing_models = []
for name, desc in MODEL_CANDIDATES:
    path = find_local_model(name)
    model_desc[name] = desc
    if path:
        available_models.append(path)
    else:
        missing_models.append(name)

if not available_models:
    st.sidebar.error("未找到本地模型权重文件（例如 yolov8n.pt）。请将 .pt 放到项目根目录后重试。")
    st.stop()

def model_label(path: str) -> str:
    base = os.path.basename(path)
    return f"{base}（{model_desc.get(base, '模型')}，本地）"

model_path = st.sidebar.selectbox(
    "选择模型（离线仅显示本地可用）",
    options=available_models,
    format_func=model_label,
    key="model_path",
)

if missing_models:
    st.sidebar.info("未检测到：" + "、".join(missing_models) + "（离线环境下不会自动下载）")

# 主界面
if uploaded_file is not None:
    # 保存上传的文件
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.flush()

    st.video(tfile.name)

    orientation = "horizontal" if line_orientation.startswith("水平") else "vertical"
    if show_preview:
        cap_preview = cv2.VideoCapture(tfile.name)
        ok, preview_frame = cap_preview.read()
        preview_width = int(cap_preview.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        preview_height = int(cap_preview.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap_preview.release()

        if ok and preview_width > 0 and preview_height > 0:
            if orientation == "horizontal":
                line_value = int(preview_height * line_position / 100)
                cv2.line(preview_frame, (0, line_value), (preview_width, line_value), (0, 255, 255), 3)
            else:
                line_value = int(preview_width * line_position / 100)
                cv2.line(preview_frame, (line_value, 0), (line_value, preview_height), (0, 255, 255), 3)

            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            st.image(preview_frame, caption="计数线预览（黄色）", use_container_width=True)

    recommend_clicked = st.button("自动推荐计数线（基于前200帧）")
    if recommend_clicked:
        with st.spinner("正在分析视频轨迹并推荐计数线..."):
            rec = recommend_counting_line(tfile.name, model_path, max_frames=200)
        if not rec:
            st.warning("无法从当前视频中提取稳定轨迹，建议手动调整计数线。")
        else:
            st.info(f"推荐：{('水平线' if rec['orientation']=='horizontal' else '垂直线')}，位置 {rec['position']}%，候选轨迹 {rec['tracks']}，覆盖 {rec['score']} 条轨迹")
            st.session_state["line_orientation"] = "水平线（上下穿越）" if rec["orientation"] == "horizontal" else "垂直线（左右穿越）"
            st.session_state["line_position"] = int(rec["position"])
            st.rerun()

    if st.button("开始分析", type="primary"):
        try:
            with st.spinner("正在处理视频..."):
                used_orientation = orientation
                used_position = line_position
                if auto_use_recommended:
                    rec = recommend_counting_line(tfile.name, model_path, max_frames=200)
                    if rec and rec["score"] > 0:
                        used_orientation = rec["orientation"]
                        used_position = int(rec["position"])
                        st.info(f"本次使用推荐计数线：{('水平线' if used_orientation=='horizontal' else '垂直线')} {used_position}%（覆盖 {rec['score']} 条轨迹）")
                    else:
                        st.info("未能自动推荐计数线，本次使用当前滑块设置。")
                stats = process_video(tfile.name, used_position, used_orientation, model_path)
        except Exception as exc:
            st.error(f"处理失败：{exc}")
        else:
            st.success("分析完成！")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("处理结果")
                if os.path.exists(EVENTS_PATH):
                    events_df = pd.read_csv(EVENTS_PATH)
                    st.write(f"总过线次数: {len(events_df)}")
                    st.write(f"进入车辆: {stats['in_count']}")
                    st.write(f"离开车辆: {stats['out_count']}")
                    st.dataframe(events_df.head(10))

                if os.path.exists(CHART_PATH):
                    st.image(CHART_PATH)
                else:
                    st.info("暂无图表数据。")

            with col2:
                st.subheader("下载结果")
                if os.path.exists(VIDEO_OUTPUT_PATH) and os.path.getsize(VIDEO_OUTPUT_PATH) > 0:
                    file_size = os.path.getsize(VIDEO_OUTPUT_PATH) / (1024 * 1024)
                    st.write(f"处理后的视频大小: {file_size:.2f} MB")
                    with open(VIDEO_OUTPUT_PATH, 'rb') as video_file:
                        st.download_button(
                            label="下载处理后的视频",
                            data=video_file,
                            file_name="processed_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                else:
                    st.warning("处理后的视频文件不存在或为空")

                if os.path.exists(EVENTS_PATH):
                    with open(EVENTS_PATH, 'rb') as f:
                        st.download_button(
                            label="下载事件日志(CSV)",
                            data=f,
                            file_name="traffic_events.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                if os.path.exists(CHART_PATH):
                    with open(CHART_PATH, 'rb') as f:
                        st.download_button(
                            label="下载趋势图表",
                            data=f,
                            file_name="traffic_trend.png",
                            mime="image/png",
                            use_container_width=True
                        )

            st.subheader("处理日志")
            st.write(f"检测到 {stats['event_count']} 个车辆过线事件")
            if stats["event_count"] == 0:
                st.warning("当前计数线位置下未检测到穿线事件，建议调整计数线位置后重试。")
