# 低空车流量识别与分析系统（Traffic Project）

## 1. 项目简介

本项目提供一个基于 YOLOv8 的车辆检测与跟踪、并结合“计数线穿越”策略统计车流量的可视化工具。你可以在网页端上传视频，选择计数线方向与位置，得到：

- 处理后的视频（含检测框、计数线与统计信息叠加）
- 过线事件日志（CSV）
- 简单的分钟级趋势图（PNG）

项目同时提供若干脚本，用于本地快速验证环境、跟踪输出、计数输出与统计导图。

仓库地址：`https://github.com/zoniery/traffic--project`

## 2. 功能介绍

### 2.1 Web 页面（Streamlit）

入口文件：`src/app.py`

- 上传视频并预览
- 选择模型：`yolov8n / yolov8s / yolov8m`
- 选择计数线方向：
  - 水平线（上下穿越）：上 → 下记为“进入”，下 → 上记为“离开”
  - 垂直线（左右穿越）：左 → 右记为“进入”，右 → 左记为“离开”
- 调整计数线位置（按画面高度/宽度百分比）
- 计数线预览（黄色线，基于视频第一帧）
- 输出下载：
  - `data/outputs/processed_video.mp4`
  - `data/outputs/events.csv`
  - `data/outputs/traffic_trend.png`

### 2.2 脚本工具

- `src/01_check_gpu.py`：检查 PyTorch/YOLO 是否可运行（不依赖外网）
- `src/02_track_video.py`：读取 `data/raw_videos` 下的首个 mp4，输出跟踪视频到 `data/outputs`
- `src/03_count_line.py`：读取 `data/raw_videos` 下的首个 mp4，按中心点穿线逻辑输出 `events.csv`
- `src/04_traffic_status.py`：读取 `events.csv`，生成分钟粒度的交通状态统计（当 events 为空时自动跳过）
- `src/05_export_charts.py`：基于 `traffic_stats.csv` 导出趋势图/状态分布图（当数据缺失时自动跳过）

## 3. 部署方式（本地运行）

### 3.1 环境要求

- Python 3.10+（建议 3.11/3.12；3.13 也可运行，但某些第三方包兼容性可能波动）
- 具备安装依赖与下载模型权重的网络环境（首次运行 YOLO 可能下载 `.pt`）

### 3.2 克隆与安装依赖（macOS / Linux）

```bash
git clone https://github.com/zoniery/traffic--project.git
cd traffic--project

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

pip install streamlit ultralytics supervision opencv-python pandas matplotlib torch torchvision torchaudio
```

### 3.3 克隆与安装依赖（Windows PowerShell）

```powershell
git clone https://github.com/zoniery/traffic--project.git
cd traffic--project

python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip

pip install streamlit ultralytics supervision opencv-python pandas matplotlib torch torchvision torchaudio
```

### 3.4 启动 Web 页面

```bash
streamlit run src/app.py
```

打开页面后上传视频，先看“计数线预览”，再点击“开始分析”。

### 3.5 使用本地脚本（可选）

将你的视频放到 `data/raw_videos/` 下（可多级目录），脚本会自动选择第一个 `*.mp4` 文件。

```bash
python src/01_check_gpu.py
python src/02_track_video.py
python src/03_count_line.py
python src/04_traffic_status.py
python src/05_export_charts.py
```

## 4. 数据与输出说明

### 4.1 输入

- Web：页面上传任意 mp4/avi/mov
- 脚本：`data/raw_videos/**.mp4`

说明：仓库默认 `.gitignore` 会忽略 `data/raw_videos/` 和 `*.mp4`，避免将大文件推送到 GitHub。请自行准备本地视频。

### 4.2 输出

默认输出目录：`data/outputs/`

- `processed_video.mp4`：处理后视频
- `events.csv`：过线事件（字段：`timestamp / direction / count_delta`，其中 `count_delta` 默认恒为 1）
- `traffic_trend.png`：分钟粒度事件数趋势图

注意：当前统计口径为“过线次数/事件数”，不是“车道真实流量的地面标定结果”。要得到更严格的车流量指标，通常还需要更精准的相机标定、车道区域定义与过滤策略。

## 5. 不同系统上可能出现的问题与解决方案

### 5.1 Streamlit 无法写入配置目录（PermissionError）

现象：启动时报 `PermissionError`，提示无法创建 `~/.streamlit`。

原因：部分环境对用户目录写入受限（尤其是沙箱环境或企业管控环境）。

解决：本项目已在 `src/app.py` 中将 `STREAMLIT_HOME` 设置为系统临时目录（macOS/Windows 通用）。若你需要自定义：

```bash
export STREAMLIT_HOME=/path/to/writable/dir
```

### 5.2 YOLO 权重下载失败（网络/证书）

现象：首次运行时下载 `yolov8n.pt` 失败，出现 `CERTIFICATE_VERIFY_FAILED`、超时或无法连接 GitHub/Ultralytics。

原因：网络受限、代理、公司内网或证书链不完整。

解决：

- 确保可访问外网资源（GitHub / Ultralytics）
- 或手动下载 `.pt` 放到项目根目录（与代码中的 `yolov8n.pt` 同路径）

### 5.3 输出视频无法播放 / 0KB

现象：程序跑完但 `processed_video.mp4` 无法打开或大小为 0。

原因：OpenCV 写 mp4 依赖本机编码器支持，不同系统/环境差异较大。

解决：

- 更换播放器验证（VLC 等）
- 尝试升级 OpenCV（`pip install -U opencv-python`）
- 若仍不行，可改为输出 AVI（需要你自行修改代码中的输出编码与后缀）

### 5.4 Windows 上推理很慢 / CUDA 不可用

现象：`torch.cuda.is_available()` 为 False，推理慢。

原因：安装的是 CPU 版 PyTorch，或驱动/CUDA 不匹配。

解决：

- 若有 NVIDIA 显卡：按 PyTorch 官网指引安装对应 CUDA 版本
- 若无 NVIDIA 显卡：这是正常现象，建议使用轻量模型 `yolov8n.pt`，并降低分辨率/帧率

### 5.5 “计不到数”（events.csv 为空）

现象：检测框有，但 `events.csv` 为空、页面提示无过线事件。

原因：计数逻辑基于“目标中心点穿越计数线”，常见原因是：

- 线的位置没有覆盖车流主通道
- 车辆主要左右移动但你选了水平线（或相反）
- 目标跟踪在遮挡/抖动时不稳定，中心点没有形成有效穿越

解决建议：

- 先开启“计数线预览”，把线放在车道中央
- 车流主要左右通行：切换为“垂直线（左右穿越）”
- 调整线位置（例如 30%~70% 多试几档），直到出现稳定事件

## 6. 目录结构

```text
traffic--project/
  src/
    app.py
    01_check_gpu.py
    02_track_video.py
    03_count_line.py
    04_traffic_status.py
    05_export_charts.py
  data/
    raw_videos/        # 本地视频（默认不提交）
    outputs/           # 输出结果（默认不提交）
```

