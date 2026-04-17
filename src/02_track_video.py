import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 加载模型（先用轻量级模型验证）
model = YOLO('yolov8n.pt')

# 打开视频
video_candidates = sorted([p for p in Path("data/raw_videos").glob("**/*.mp4") if p.is_file()])
if not video_candidates:
    raise FileNotFoundError("未找到可用视频文件，请检查 data/raw_videos 目录。")
video_path = str(video_candidates[0])
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"视频无法读取: {video_path}")

# 获取视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("视频元信息无效，无法继续处理。")

os.makedirs("data/outputs", exist_ok=True)

# 创建视频写入器
out = cv2.VideoWriter(
    'data/outputs/test1_track.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (width, height)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 执行检测+跟踪（显式指定ByteTrack）
    results = model.track(
        frame, 
        persist=True,  # 保持跟踪器状态
        tracker='bytetrack.yaml',  # 显式指定跟踪器
        classes=[2, 3, 5, 7]  # 只保留car, motorcycle, bus, truck
    )
    
    # 绘制结果
    annotated_frame = results[0].plot()
    
    # 写入输出
    out.write(annotated_frame)
    
    # 显示窗口在无GUI环境下可能失败，这里仅输出文件

cap.release()
out.release()
print(f"处理完成，输出文件: data/outputs/test1_track.mp4，输入: {video_path}")
