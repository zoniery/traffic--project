import torch
import numpy as np
from ultralytics import YOLO

# 验证PyTorch和CUDA
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
else:
    print("GPU设备: 当前环境无CUDA，使用CPU模式")

# 验证YOLO推理
model = YOLO('yolov8n.pt')  # 下载轻量级模型
dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
results = model(dummy_image)  # 本地推理测试，避免网络依赖
print("YOLO推理测试成功")
