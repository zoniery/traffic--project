import cv2
import csv
import os
from pathlib import Path
from ultralytics import YOLO


def update_crossing_counts(track_sides, boxes, line_y, timestamp, margin):
    new_in = 0
    new_out = 0
    events = []

    if boxes.id is None or boxes.xyxy is None:
        return new_in, new_out, events

    track_ids = boxes.id.int().cpu().tolist()
    boxes_xyxy = boxes.xyxy.cpu().tolist()
    active_ids = set()

    for track_id, xyxy in zip(track_ids, boxes_xyxy):
        active_ids.add(track_id)
        center_y = (xyxy[1] + xyxy[3]) / 2
        if center_y < line_y - margin:
            current_side = "above"
        elif center_y > line_y + margin:
            current_side = "below"
        else:
            current_side = None

        previous_side = track_sides.get(track_id)
        if current_side is None:
            continue

        if previous_side == "above" and current_side == "below":
            new_in += 1
            events.append({"timestamp": timestamp, "event": "enter", "count": 1})
        elif previous_side == "below" and current_side == "above":
            new_out += 1
            events.append({"timestamp": timestamp, "event": "exit", "count": 1})

        track_sides[track_id] = current_side

    stale_ids = [track_id for track_id in track_sides if track_id not in active_ids]
    for track_id in stale_ids:
        del track_sides[track_id]

    return new_in, new_out, events

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

line_y = int(height * 0.6)
margin = max(8, height // 50)
track_sides = {}

count_in = 0  # 进入计数
count_out = 0  # 出去计数

# 准备CSV日志
events_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 检测+跟踪
    results = model.track(
        frame,
        persist=True,
        tracker='bytetrack.yaml',
        classes=[2, 3, 5, 7]
    )
    
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    new_count_in, new_count_out, new_events = update_crossing_counts(
        track_sides,
        results[0].boxes,
        line_y,
        timestamp,
        margin,
    )
    count_in += new_count_in
    count_out += new_count_out
    events_log.extend(new_events)

    annotated_frame = results[0].plot()
    cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
    cv2.putText(
        annotated_frame,
        'Above->Below = In, Below->Above = Out',
        (10, max(25, line_y - 10)),
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
    writer = csv.DictWriter(f, fieldnames=['timestamp', 'event', 'count'])
    writer.writeheader()
    writer.writerows(events_log)

print(f"处理完成，总计进入: {count_in} 辆")
