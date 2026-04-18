import cv2
import csv
import os
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

try:
    from src.counting_core import MIN_RECOMMEND_SCORE, recommend_counting_line, update_crossing_counts
except Exception:
    from counting_core import MIN_RECOMMEND_SCORE, recommend_counting_line, update_crossing_counts

# 初始化
model = YOLO('yolov8n.pt')
video_candidates = sorted([p for p in Path("data/raw_videos").glob("**/*.mp4") if p.is_file()])
if not video_candidates:
    raise FileNotFoundError("未找到可用视频文件，请检查 data/raw_videos 目录。")
video_path = str(video_candidates[0])
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"视频无法读取: {video_path}")

# 视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("视频元信息无效，无法继续处理。")

os.makedirs('data/outputs', exist_ok=True)

# 创建输出
out = cv2.VideoWriter(
    'data/outputs/test1_counted.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (width, height)
)

rec = recommend_counting_line(video_path, model, max_frames=200)
if rec and rec["score"] > 0 and rec["score"] < MIN_RECOMMEND_SCORE:
    print(f"推荐得分偏低（{rec['score']} < {MIN_RECOMMEND_SCORE}），建议手动调整计数线；本次仍使用推荐线。")

if rec and rec["score"] > 0:
    orientation = rec["orientation"]
    if orientation == "horizontal":
        line_value = int(height * rec["position"] / 100)
        margin = max(5, height // 100)
    else:
        line_value = int(width * rec["position"] / 100)
        margin = max(5, width // 100)
    print(
        f"推荐计数线：{orientation} {rec['position']}%"
        f"（候选轨迹 {rec['tracks']}，覆盖 {rec.get('coverage', 0)}，置信 {rec['score']}）"
    )
else:
    orientation = "horizontal"
    line_value = int(height * 0.6)
    margin = max(5, height // 100)
    print("未能自动推荐计数线，使用默认水平线 60%")
track_last_seen = {}
track_states = {}
track_last_count = {}
frame_index = 0

count_in = 0  # 进入计数
count_out = 0  # 出去计数

# 准备CSV日志
events_log = []
tracker_main = sv.ByteTrack()
unique_ids = set()
max_process_frames = int(os.environ.get("TP_MAX_PROCESS_FRAMES", "0") or "0")
cooldown_frames = int(os.environ.get("TP_COOLDOWN", "3") or "3")
ttl_frames = int(os.environ.get("TP_TTL", "90") or "90")
band_box_scale = float(os.environ.get("TP_BAND_BOX_SCALE", "0.12") or "0.12")
band_margin_scale = float(os.environ.get("TP_BAND_MARGIN_SCALE", "0.8") or "0.8")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if max_process_frames > 0 and frame_index >= max_process_frames:
        break
    
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    detections = tracker_main.update_with_detections(detections)
    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        unique_ids.update(detections.tracker_id.tolist())
    
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    new_count_in, new_count_out, new_events = update_crossing_counts(
        track_last_seen,
        track_states,
        track_last_count,
        frame_index,
        detections.tracker_id.tolist() if detections.tracker_id is not None else [],
        detections.xyxy.tolist() if detections.xyxy is not None else [],
        orientation,
        line_value,
        timestamp,
        margin,
        ttl=ttl_frames,
        cooldown=cooldown_frames,
        band_box_scale=band_box_scale,
        band_margin_scale=band_margin_scale,
    )
    count_in += new_count_in
    count_out += new_count_out
    events_log.extend(new_events)
    frame_index += 1

    annotated_frame = results[0].plot()
    if detections.xyxy is not None:
        for xyxy in detections.xyxy.tolist():
            x1, _, x2, y2 = xyxy
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            cv2.circle(annotated_frame, (foot_x, foot_y), 5, (0, 255, 255), -1)
    if orientation == "horizontal":
        cv2.line(annotated_frame, (0, line_value), (width, line_value), (0, 255, 255), 2)
        tip = 'Above->Below = In, Below->Above = Out'
        tip_pos = (10, max(25, line_value - 10))
    else:
        cv2.line(annotated_frame, (line_value, 0), (line_value, height), (0, 255, 255), 2)
        tip = 'Left->Right = In, Right->Left = Out'
        tip_pos = (max(10, line_value + 10), 25)
    cv2.putText(
        annotated_frame,
        tip,
        tip_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 2
    )
    
    # 显示计数
    cv2.putText(
        annotated_frame,
        f'In: {count_in} | Out: {count_out}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )
    
    out.write(annotated_frame)
    
    # 显示窗口在无GUI环境下可能失败，这里仅输出文件

cap.release()
out.release()

# 保存事件日志
with open('data/outputs/events.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['time_s', 'direction', 'count_delta'])
    writer.writeheader()
    writer.writerows(events_log)

print(f"处理完成，总计进入: {count_in} 辆，总计离开: {count_out} 辆")
print(f"跟踪到的车辆ID数量: {len(unique_ids)}")
