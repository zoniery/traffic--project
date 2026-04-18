import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

try:
    from src.app import (
        CHART_PATH,
        EVENTS_PATH,
        MODEL_CANDIDATES,
        MIN_RECOMMEND_SCORE,
        REALTIME_CHART_PATH,
        REALTIME_TABLE_PATH,
        VIDEO_OUTPUT_PATH,
        find_local_model,
        process_video,
        recommend_counting_line,
    )
except Exception:
    from app import (
        CHART_PATH,
        EVENTS_PATH,
        MODEL_CANDIDATES,
        MIN_RECOMMEND_SCORE,
        REALTIME_CHART_PATH,
        REALTIME_TABLE_PATH,
        VIDEO_OUTPUT_PATH,
        find_local_model,
        process_video,
        recommend_counting_line,
    )


def main():
    st.set_page_config(page_title="车流量识别系统", layout="wide")
    st.markdown(
        """
<style>
:root {
  --tp-bg: #0b1220;
  --tp-surface: #0f1a2b;
  --tp-surface-2: #111f34;
  --tp-border: rgba(255,255,255,.08);
  --tp-border-2: rgba(255,255,255,.12);
  --tp-text: #e8eefc;
  --tp-muted: rgba(232,238,252,.72);
  --tp-muted-2: rgba(232,238,252,.55);
  --tp-accent: #2f6bff;
  --tp-radius: 14px;
}

.block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1200px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--tp-border); }
h1 { letter-spacing: .2px; }
h2, h3 { letter-spacing: .15px; }

.small-muted { color: var(--tp-muted); font-size: 0.92rem; line-height: 1.55; }
.tp-divider { height: 1px; background: var(--tp-border); margin: 0.25rem 0 0.75rem 0; }

.tp-card {
  background: var(--tp-surface);
  border: 1px solid var(--tp-border);
  border-radius: var(--tp-radius);
  padding: 14px 14px;
}
.tp-card__hd { display: flex; flex-direction: column; gap: 2px; margin-bottom: 10px; }
.tp-card__title { font-size: 0.98rem; font-weight: 650; color: var(--tp-text); }
.tp-card__sub { font-size: 0.82rem; color: var(--tp-muted-2); }

.tp-kv { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,.06); }
.tp-kv:last-child { border-bottom: none; }
.tp-k { color: var(--tp-muted-2); font-size: 0.85rem; }
.tp-v { color: var(--tp-text); font-weight: 600; font-size: 0.92rem; text-align: right; }

.tp-pill { display: inline-flex; align-items: center; border: 1px solid var(--tp-border); background: rgba(255,255,255,.03); color: var(--tp-muted); border-radius: 999px; padding: 2px 8px; font-size: 0.78rem; }

.tp-callout {
  border: 1px solid var(--tp-border);
  border-radius: var(--tp-radius);
  padding: 10px 12px;
  background: rgba(255,255,255,.02);
}
.tp-callout--info { border-color: rgba(47,107,255,.35); background: rgba(47,107,255,.10); }
.tp-callout--warn { border-color: rgba(245,158,11,.35); background: rgba(245,158,11,.10); }
.tp-callout--error { border-color: rgba(239,68,68,.35); background: rgba(239,68,68,.10); }
.tp-callout--ok { border-color: rgba(34,197,94,.35); background: rgba(34,197,94,.10); }
.tp-callout__title { font-weight: 650; margin-bottom: 4px; }
.tp-callout__line { color: var(--tp-muted); font-size: 0.9rem; line-height: 1.55; }

div[data-testid="stMetric"] {
  background: var(--tp-surface);
  border: 1px solid var(--tp-border);
  padding: 0.85rem 0.95rem;
  border-radius: var(--tp-radius);
}
div[data-testid="stMetric"] * { color: var(--tp-text); }
div[data-testid="stMetric"] label { color: var(--tp-muted-2) !important; }

div.stButton > button {
  border-radius: 12px;
  padding: 0.62rem 0.9rem;
  border: 1px solid var(--tp-border);
  background: rgba(255,255,255,.03);
}
div.stButton > button:hover {
  border-color: var(--tp-border-2);
  background: rgba(255,255,255,.06);
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
  background: var(--tp-accent);
  border-color: var(--tp-accent);
  color: #ffffff;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
  filter: brightness(.96);
}

[data-testid="stTabs"] { background: var(--tp-surface); border: 1px solid var(--tp-border); border-radius: var(--tp-radius); padding: 10px 12px 12px 12px; }
[data-testid="stTabs"] [role="tablist"] { gap: 6px; }
[data-testid="stTabs"] [role="tab"] { border-radius: 999px; border: 1px solid var(--tp-border); padding: 6px 10px; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: rgba(255,255,255,.06); border-color: var(--tp-border-2); }
</style>
""",
        unsafe_allow_html=True,
    )

    def card_open(title: str, subtitle: str | None = None):
        subtitle_html = f"<div class='tp-card__sub'>{subtitle}</div>" if subtitle else ""
        st.markdown(
            f"<div class='tp-card'><div class='tp-card__hd'><div class='tp-card__title'>{title}</div>{subtitle_html}</div>",
            unsafe_allow_html=True,
        )

    def card_close():
        st.markdown("</div>", unsafe_allow_html=True)

    def kv_row(label: str, value: str):
        st.markdown(
            f"<div class='tp-kv'><div class='tp-k'>{label}</div><div class='tp-v'>{value}</div></div>",
            unsafe_allow_html=True,
        )

    def callout(kind: str, title: str, lines: list[str]):
        safe_lines = "".join([f"<div class='tp-callout__line'>{line}</div>" for line in lines])
        st.markdown(
            f"<div class='tp-callout tp-callout--{kind}'><div class='tp-callout__title'>{title}</div>{safe_lines}</div>",
            unsafe_allow_html=True,
        )

    def format_duration(seconds: float | None) -> str:
        if seconds is None or seconds <= 0:
            return "—"
        m = int(seconds // 60)
        s = int(round(seconds - m * 60))
        return f"{m}分{s:02d}秒" if m > 0 else f"{s}秒"

    def get_video_meta(video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"ok": False}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        duration_s = (frame_count / fps) if fps > 0 and frame_count > 0 else None
        return {
            "ok": True,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_s": duration_s,
        }

    st.title("低空车流量实时识别与分析")
    st.markdown(
        "<div class='small-muted'>上传视频并配置计数线，生成过线统计、趋势图与处理后视频。保持简洁路径：输入 → 设置 → 分析 → 结果。</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("控制台")
    st.sidebar.markdown(
        "<div class='small-muted'>按 <span class='tp-pill'>输入</span> → <span class='tp-pill'>设置</span> → <span class='tp-pill'>分析</span> → <span class='tp-pill'>结果</span> 的顺序操作。</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<div class='tp-divider'></div>", unsafe_allow_html=True)

    st.sidebar.subheader("输入")
    uploaded_file = st.sidebar.file_uploader("上传视频", type=["mp4", "avi", "mov"])

    st.sidebar.subheader("模型")
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
        "选择模型（离线仅显示本地文件）",
        options=available_models,
        format_func=model_label,
        key="model_path",
    )

    if missing_models:
        st.sidebar.info("未检测到：" + "、".join(missing_models) + "（离线环境下不会自动下载）")

    st.sidebar.subheader("计数线")
    line_orientation = st.sidebar.radio(
        "方向",
        ["水平线（上下穿越）", "垂直线（左右穿越）"],
        index=0,
        key="line_orientation",
    )
    line_position_label = "位置（画面高度%）" if line_orientation.startswith("水平") else "位置（画面宽度%）"
    line_position = st.sidebar.slider(line_position_label, 0, 100, 60, key="line_position")

    with st.sidebar.expander("高级选项", expanded=False):
        show_preview = st.checkbox("显示计数线预览", value=True, key="show_preview")
        auto_use_recommended = st.checkbox("开始分析前自动推荐并使用计数线", value=True, key="auto_use_recommended")
        cooldown_frames = st.slider("计数冷却帧", 1, 10, 3, key="cooldown_frames")
        sensitivity = st.selectbox(
            "贴线敏感度",
            options=["高", "中", "低"],
            index=1,
            key="band_sensitivity",
        )
        st.markdown("<div class='small-muted'>推荐线将分析视频前 200 帧轨迹，用于推断更合适的方向与位置。</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>冷却帧越小越容易计数（也更容易重复计数）；贴线敏感度越高越容易计数。</div>", unsafe_allow_html=True)

    band_box_scale = {"高": 0.08, "中": 0.12, "低": 0.16}.get(sensitivity, 0.12)

    if uploaded_file is None:
        card_open("开始之前", "上传一段视频并完成基础设置")
        callout(
            "info",
            "建议操作",
            [
                "1) 上传视频后先开启计数线预览，把线放在车道中央附近。",
                "2) 建议保持开启“开始分析前自动推荐并使用计数线”。",
                "3) 车流主要左右行驶用垂直线；主要上下行驶用水平线。",
            ],
        )
        card_close()
        st.stop()

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()

    orientation = "horizontal" if line_orientation.startswith("水平") else "vertical"

    if "last_recommendation" not in st.session_state:
        st.session_state["last_recommendation"] = None
    if "last_used_params" not in st.session_state:
        st.session_state["last_used_params"] = None
    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False

    video_meta = get_video_meta(tfile.name)
    video_sub = None
    if video_meta.get("ok") and video_meta.get("width") and video_meta.get("height"):
        wh = f"{video_meta['width']}×{video_meta['height']}"
        fps = f"{video_meta['fps']:.1f} FPS" if video_meta.get("fps") else "—"
        dur = format_duration(video_meta.get("duration_s"))
        video_sub = f"{wh} · {fps} · {dur}"

    main_left, main_right = st.columns([1.35, 1.0])
    with main_left:
        card_open("视频预览", video_sub)
        st.video(tfile.name)
        card_close()

    with main_right:
        card_open("本次配置", "开始分析前会在这里确认最终参数")
        kv_row("模型", os.path.basename(model_path))
        kv_row("计数线方向", "水平线（上→下进入）" if orientation == "horizontal" else "垂直线（左→右进入）")
        kv_row("计数线位置", f"{line_position}%")
        kv_row("自动推荐", "开启" if auto_use_recommended else "关闭")
        kv_row("冷却帧", str(cooldown_frames))
        kv_row("贴线敏感度", sensitivity)
        last_rec = st.session_state.get("last_recommendation")
        if last_rec and last_rec.get("score", 0) >= MIN_RECOMMEND_SCORE:
            rec_ori = "水平线" if last_rec["orientation"] == "horizontal" else "垂直线"
            kv_row(
                "上次推荐",
                f"{rec_ori} {int(last_rec['position'])}%（覆盖 {int(last_rec.get('coverage', 0))} 条轨迹，置信 {int(last_rec['score'])}）",
            )
        last_used = st.session_state.get("last_used_params")
        if last_used:
            used_ori = "水平线" if last_used["orientation"] == "horizontal" else "垂直线"
            badge = "推荐" if last_used.get("source") == "recommend" else "手动"
            kv_row("最终使用", f"{used_ori} {int(last_used['position'])}%（{badge}）")
        card_close()

        card_open("计数线预览", "黄色线为计数线，建议覆盖车流主通道")
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
                st.image(preview_frame, use_container_width=True)
                st.markdown(
                    "<div class='small-muted'>水平线：上→下记为进入，下→上记为离开；垂直线：左→右记为进入，右→左记为离开。</div>",
                    unsafe_allow_html=True,
                )
            else:
                callout("warn", "无法生成预览", ["未能读取视频第一帧。可尝试更换视频编码或格式后重试。"])
        else:
            st.markdown("<div class='small-muted'>你已关闭计数线预览。</div>", unsafe_allow_html=True)
        card_close()

    card_open("操作", "推荐线用于优化方向与位置；开始分析会生成输出文件")
    btn_left, btn_right = st.columns([1, 2])
    with btn_left:
        recommend_clicked = st.button("推荐线", use_container_width=True, disabled=st.session_state["is_processing"])
    with btn_right:
        run_clicked = st.button("开始分析", type="primary", use_container_width=True, disabled=st.session_state["is_processing"])
    card_close()

    if recommend_clicked:
        with st.spinner("正在分析视频轨迹并推荐计数线..."):
            rec = recommend_counting_line(tfile.name, model_path, max_frames=200)
        if not rec or rec["score"] < MIN_RECOMMEND_SCORE:
            st.session_state["last_recommendation"] = {"score": 0}
            callout("warn", "推荐不可靠", [f"推荐得分过低（< {MIN_RECOMMEND_SCORE}）。", "建议：手动调整计数线方向/位置，或换一个更清晰的视频片段。"])
        else:
            st.session_state["last_recommendation"] = rec
            callout(
                "ok",
                "推荐结果",
                [
                    f"建议使用：{('水平线' if rec['orientation']=='horizontal' else '垂直线')} {int(rec['position'])}%",
                    f"覆盖轨迹数：{int(rec.get('coverage', 0))}，推荐置信：{int(rec['score'])}",
                ],
            )
            st.session_state["line_orientation"] = "水平线（上下穿越）" if rec["orientation"] == "horizontal" else "垂直线（左右穿越）"
            st.session_state["line_position"] = int(rec["position"])
            st.rerun()

    if run_clicked:
        st.session_state["is_processing"] = True
        try:
            with st.spinner("正在处理视频..."):
                used_orientation = orientation
                used_position = line_position
                used_source = "manual"
                if auto_use_recommended:
                    rec = recommend_counting_line(tfile.name, model_path, max_frames=200)
                    if rec and rec["score"] >= MIN_RECOMMEND_SCORE:
                        used_orientation = rec["orientation"]
                        used_position = int(rec["position"])
                        used_source = "recommend"
                        st.session_state["last_recommendation"] = rec
                st.session_state["last_used_params"] = {
                    "orientation": used_orientation,
                    "position": int(used_position),
                    "source": used_source,
                }
                stats = process_video(
                    tfile.name,
                    used_position,
                    used_orientation,
                    model_path,
                    cooldown=cooldown_frames,
                    band_box_scale=band_box_scale,
                )
        except Exception as exc:
            callout("error", "处理失败", [str(exc), "建议：确认模型文件存在、视频可正常播放，或尝试更换输出编码环境。"])
        else:
            events_df = pd.read_csv(EVENTS_PATH) if os.path.exists(EVENTS_PATH) else pd.DataFrame(columns=["time_s", "direction", "count_delta"])

            card_open("结果摘要", "指标与最终使用参数")
            last_used = st.session_state.get("last_used_params")
            if last_used:
                used_ori = "水平线" if last_used["orientation"] == "horizontal" else "垂直线"
                badge = "推荐" if last_used.get("source") == "recommend" else "手动"
                st.markdown(
                    f"<div class='small-muted'>最终使用：<span class='tp-pill'>{used_ori} {int(last_used['position'])}%</span> <span class='tp-pill'>{badge}</span></div>",
                    unsafe_allow_html=True,
                )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("进入", int(stats["in_count"]))
            m2.metric("离开", int(stats["out_count"]))
            m3.metric("过线事件", int(stats["event_count"]))
            m4.metric("跟踪车辆", int(stats.get("unique_vehicles", 0)))
            if stats["event_count"] == 0:
                callout(
                    "warn",
                    "未检测到过线事件",
                    [
                        "建议 1：切换计数线方向（水平/垂直）。",
                        "建议 2：把线放到车流更集中区域（例如 30%~70% 多试几档）。",
                        "建议 3：确认车辆确实穿越计数线，而不是沿线移动。",
                    ],
                )
            card_close()

            tabs = st.tabs(["概览", "事件数据", "处理后视频", "下载"])
            with tabs[0]:
                if os.path.exists(REALTIME_CHART_PATH):
                    st.image(REALTIME_CHART_PATH, use_container_width=True)
                if os.path.exists(REALTIME_TABLE_PATH):
                    flow_df = pd.read_csv(REALTIME_TABLE_PATH)
                    st.dataframe(flow_df, use_container_width=True, height=360)
                else:
                    callout("info", "暂无实时流量表", ["当出现过线事件后会自动生成每秒流量表。"])

            with tabs[1]:
                st.dataframe(events_df, use_container_width=True, height=420)

            with tabs[2]:
                if os.path.exists(VIDEO_OUTPUT_PATH) and os.path.getsize(VIDEO_OUTPUT_PATH) > 0:
                    st.video(VIDEO_OUTPUT_PATH)
                else:
                    callout("info", "暂无输出视频", ["当视频编码器不可用时可能无法生成输出。可尝试更换系统编码环境或改用 AVI 输出。"])

            with tabs[3]:
                card_open("下载", "仅在文件成功生成后可下载")
                if os.path.exists(VIDEO_OUTPUT_PATH) and os.path.getsize(VIDEO_OUTPUT_PATH) > 0:
                    with open(VIDEO_OUTPUT_PATH, "rb") as video_file:
                        st.download_button(
                            label="下载处理后的视频",
                            data=video_file,
                            file_name="processed_video.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )
                else:
                    st.markdown("<div class='small-muted'>处理后视频：暂无可下载文件。</div>", unsafe_allow_html=True)
                if os.path.exists(EVENTS_PATH):
                    with open(EVENTS_PATH, "rb") as f:
                        st.download_button(
                            label="下载事件日志(CSV)",
                            data=f,
                            file_name="traffic_events.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                else:
                    st.markdown("<div class='small-muted'>事件日志：暂无可下载文件。</div>", unsafe_allow_html=True)
                if os.path.exists(REALTIME_TABLE_PATH):
                    with open(REALTIME_TABLE_PATH, "rb") as f:
                        st.download_button(
                            label="下载实时流量表(CSV)",
                            data=f,
                            file_name="realtime_flow_1s.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                else:
                    st.markdown("<div class='small-muted'>实时流量表：暂无可下载文件。</div>", unsafe_allow_html=True)
                if os.path.exists(REALTIME_CHART_PATH):
                    with open(REALTIME_CHART_PATH, "rb") as f:
                        st.download_button(
                            label="下载实时折线图(PNG)",
                            data=f,
                            file_name="realtime_flow_1s.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                else:
                    st.markdown("<div class='small-muted'>实时折线图：暂无可下载文件。</div>", unsafe_allow_html=True)
                card_close()
        finally:
            st.session_state["is_processing"] = False


if __name__ == "__main__":
    main()
