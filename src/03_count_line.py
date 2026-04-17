import cv2
import csv
import os
from pathlib import Path
from ultralytics import YOLO
import supervision as sv


def update_crossing_counts(track_sides, track_last_seen, frame_index, track_ids, boxes_xyxy, orientation, line_value, timestamp, margin, ttl=30):
    new_in = 0
    new_out = 0
    events = []

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
            events.append({"time_s": timestamp, "direction": "in", "count_delta": 1})
        elif previous_side == "below" and current_side == "above":
            new_out += 1
            events.append({"time_s": timestamp, "direction": "out", "count_delta": 1})
        elif previous_side == "left" and current_side == "right":
            new_in += 1
            events.append({"time_s": timestamp, "direction": "in", "count_delta": 1})
        elif previous_side == "right" and current_side == "left":
            new_out += 1
            events.append({"time_s": timestamp, "direction": "out", "count_delta": 1})

        track_sides[track_id] = current_side

    expired = [track_id for track_id, last_seen in list(track_last_seen.items()) if frame_index - last_seen > ttl]
    for track_id in expired:
        track_last_seen.pop(track_id, None)
        track_sides.pop(track_id, None)

    return new_in, new_out, events


def recommend_counting_line(video_path: str, model, max_frames: int = 200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None

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
    return {"score": int(score), "orientation": orientation, "position": position, "tracks": len(bounds), "width": width, "height": height}

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
if rec and rec["score"] > 0:
    orientation = rec["orientation"]
    if orientation == "horizontal":
        line_value = int(height * rec["position"] / 100)
        margin = max(5, height // 100)
    else:
        line_value = int(width * rec["position"] / 100)
        margin = max(5, width // 100)
    print(f"推荐计数线：{orientation} {rec['position']}%（候选轨迹 {rec['tracks']}，覆盖 {rec['score']}）")
else:
    orientation = "horizontal"
    line_value = int(height * 0.6)
    margin = max(5, height // 100)
    print("未能自动推荐计数线，使用默认水平线 60%")
track_sides = {}
track_last_seen = {}
frame_index = 0

count_in = 0  # 进入计数
count_out = 0  # 出去计数

# 准备CSV日志
events_log = []
tracker_main = sv.ByteTrack()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    detections = tracker_main.update_with_detections(detections)
    
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    new_count_in, new_count_out, new_events = update_crossing_counts(
        track_sides,
        track_last_seen,
        frame_index,
        detections.tracker_id.tolist() if detections.tracker_id is not None else [],
        detections.xyxy.tolist() if detections.xyxy is not None else [],
        orientation,
        line_value,
        timestamp,
        margin,
    )
    count_in += new_count_in
    count_out += new_count_out
    events_log.extend(new_events)
    frame_index += 1

    annotated_frame = results[0].plot()
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
