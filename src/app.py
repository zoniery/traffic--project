import os
import tempfile

os.environ.setdefault("STREAMLIT_HOME", os.path.join(tempfile.gettempdir(), "streamlit"))
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import cv2
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "data/outputs"
EVENTS_PATH = os.path.join(OUTPUT_DIR, "events.csv")
VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "processed_video.mp4")
CHART_PATH = os.path.join(OUTPUT_DIR, "traffic_trend.png")


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


def update_crossing_counts(track_sides, boxes, orientation, line_value, current_ts, margin):
    events = []
    new_in = 0
    new_out = 0

    if boxes.id is None or boxes.xyxy is None:
        return new_in, new_out, events

    track_ids = boxes.id.int().cpu().tolist()
    boxes_xyxy = boxes.xyxy.cpu().tolist()
    active_ids = set()

    for track_id, xyxy in zip(track_ids, boxes_xyxy):
        active_ids.add(track_id)
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
            events.append({"timestamp": current_ts, "direction": "in", "count_delta": 1})
        elif previous_side == "below" and current_side == "above":
            new_out += 1
            events.append({"timestamp": current_ts, "direction": "out", "count_delta": 1})
        elif previous_side == "left" and current_side == "right":
            new_in += 1
            events.append({"timestamp": current_ts, "direction": "in", "count_delta": 1})
        elif previous_side == "right" and current_side == "left":
            new_out += 1
            events.append({"timestamp": current_ts, "direction": "out", "count_delta": 1})

        track_sides[track_id] = current_side

    stale_ids = [track_id for track_id in track_sides if track_id not in active_ids]
    for track_id in stale_ids:
        del track_sides[track_id]

    return new_in, new_out, events

# 页面配置
st.set_page_config(page_title="车流量识别系统", layout="wide")
st.title("🚗 低空车流量实时识别与分析系统")

def process_video(video_path, line_pos, orientation, model_name):
    """
    处理视频的完整流程
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(model_name.replace(" (轻量快速)", "").replace(" (均衡)", "").replace(" (精确)", ""))
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
        margin = max(8, height // 50)
    else:
        line_value = int(width * line_pos / 100)
        margin = max(8, width // 50)

    events = []
    count_in = 0
    count_out = 0
    frame_idx = 0
    base_time = pd.Timestamp.now().floor("s")
    track_sides = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker='bytetrack.yaml',
            classes=[2, 3, 5, 7]
        )

        current_ts = base_time + pd.to_timedelta(frame_idx / fps, unit="s")
        frame_idx += 1

        new_in, new_out, new_events = update_crossing_counts(
            track_sides,
            results[0].boxes,
            orientation,
            line_value,
            current_ts,
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

    cap.release()
    out.release()

    events_df = pd.DataFrame(events)
    if events_df.empty:
        events_df = pd.DataFrame(columns=["timestamp", "direction", "count_delta"])
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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('1min')
    stats = df.groupby('minute').size().reset_index(name='count')

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stats['minute'], stats['count'], marker='o')
    ax.set_title('车流量趋势')
    ax.set_xlabel('时间')
    ax.set_ylabel('车辆数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_PATH)
    plt.close()

# 侧边栏
st.sidebar.header("配置选项")
uploaded_file = st.sidebar.file_uploader("上传视频文件", type=['mp4', 'avi', 'mov'])

line_orientation = st.sidebar.radio(
    "计数线方向",
    ["水平线（上下穿越）", "垂直线（左右穿越）"],
    index=0
)

line_position_label = "计数线位置（画面高度%）" if line_orientation.startswith("水平") else "计数线位置（画面宽度%）"
line_position = st.sidebar.slider(line_position_label, 0, 100, 60)
show_preview = st.sidebar.checkbox("显示计数线预览", value=True)

# 模型选择
model_option = st.sidebar.selectbox(
    "选择模型",
    ["yolov8n.pt (轻量快速)", "yolov8s.pt (均衡)", "yolov8m.pt (精确)"]
)

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

    if st.button("开始分析", type="primary"):
        try:
            with st.spinner("正在处理视频..."):
                stats = process_video(tfile.name, line_position, orientation, model_option)
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
