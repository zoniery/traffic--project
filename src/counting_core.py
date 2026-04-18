from __future__ import annotations

from collections import defaultdict

import cv2
import supervision as sv
from ultralytics import YOLO

MIN_RECOMMEND_SCORE = 5
VEHICLE_CLASSES = [2, 3, 5, 7]


def _ensure_model(model_or_path):
    if hasattr(model_or_path, "predict") and callable(getattr(model_or_path, "__call__", None)):
        return model_or_path
    return YOLO(model_or_path)


def _reference_point(xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _, x2, y2 = xyxy
    return (x1 + x2) / 2, y2


def _classify_side(coord: float, line_value: float, band: int) -> int:
    if coord <= line_value - band:
        return -1
    if coord >= line_value + band:
        return 1
    return 0


def _analysis_windows(frame_count: int, max_frames: int) -> list[tuple[int, int]]:
    if frame_count <= 0:
        return [(0, max_frames)]
    if frame_count <= max_frames * 3:
        return [(0, frame_count)]

    window = max(60, max_frames // 3)
    starts = {
        0,
        max(0, frame_count // 2 - window // 2),
        max(0, frame_count - window),
    }
    return [(start, min(frame_count, start + window)) for start in sorted(starts)]


def collect_trajectories(video_path: str, model_or_path, max_frames: int = 200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None

    model = _ensure_model(model_or_path)
    trajectories = defaultdict(list)
    processed_frames = 0

    for window_index, (start, end) in enumerate(_analysis_windows(frame_count, max_frames)):
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        tracker = sv.ByteTrack()
        frame_idx = start
        while frame_idx < end:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(frame, classes=VEHICLE_CLASSES, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            detections = tracker.update_with_detections(detections)
            if detections.tracker_id is not None and detections.xyxy is not None and len(detections) > 0:
                ids = detections.tracker_id.tolist()
                boxes = detections.xyxy.tolist()
                for track_id, box in zip(ids, boxes):
                    cx, cy = _reference_point(box)
                    pseudo_track_id = window_index * 100_000 + int(track_id)
                    trajectories[pseudo_track_id].append((cx, cy, frame_idx))

            processed_frames += 1
            frame_idx += 1

    cap.release()
    if not trajectories:
        return None

    return {
        "trajectories": dict(trajectories),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "processed_frames": processed_frames,
    }


def _candidate_stats(trajectories: dict[int, list[tuple[float, float, int]]], orientation: str, line_value: float, band: int):
    weighted_score = 0.0
    coverage = 0
    direction_balance = 0

    for points in trajectories.values():
        if len(points) < 3:
            continue

        coords = [p[1] if orientation == "horizontal" else p[0] for p in points]
        orth_coords = [p[0] if orientation == "horizontal" else p[1] for p in points]
        span = max(coords) - min(coords)
        orth_span = max(orth_coords) - min(orth_coords)
        if span < band * 3:
            continue
        if min(coords) >= line_value - band or max(coords) <= line_value + band:
            continue

        crossed = False
        last_side = None
        band_touched = False
        prev_coord = None
        for coord in coords:
            side = _classify_side(coord, line_value, band)
            if side == 0:
                if last_side is not None:
                    band_touched = True
                prev_coord = coord
                continue

            if last_side is None:
                last_side = side
                prev_coord = coord
                continue

            crossed_segment = (
                prev_coord is not None
                and (prev_coord - line_value) * (coord - line_value) < 0
                and abs(prev_coord - line_value) >= band
                and abs(coord - line_value) >= band
            )
            if side != last_side and (band_touched or crossed_segment):
                crossed = True
                break

            if side == last_side:
                band_touched = False
            last_side = side
            prev_coord = coord

        if not crossed:
            continue

        coverage += 1
        displacement = coords[-1] - coords[0]
        if abs(displacement) >= band * 2:
            direction_balance += 1 if displacement > 0 else -1

        weighted_score += 1.0
        weighted_score += min(2.0, span / max(40.0, band * 6.0))
        if span > orth_span * 1.15:
            weighted_score += 0.75
        if len(points) >= 8:
            weighted_score += 0.5

    return {
        "score": int(round(weighted_score)),
        "coverage": coverage,
        "direction_balance": direction_balance,
    }


def recommend_counting_line(video_path: str, model_or_path, max_frames: int = 200):
    data = collect_trajectories(video_path, model_or_path, max_frames=max_frames)
    if not data:
        return None

    trajectories = data["trajectories"]
    width = data["width"]
    height = data["height"]
    best = None
    best_key = None

    for orientation in ("horizontal", "vertical"):
        length = height if orientation == "horizontal" else width
        band = max(6, length // 80)
        for position in range(5, 96):
            line_value = length * position / 100.0
            stats = _candidate_stats(trajectories, orientation, line_value, band)
            candidate = {
                "score": stats["score"],
                "coverage": stats["coverage"],
                "orientation": orientation,
                "position": position,
            }
            candidate_key = (
                candidate["score"],
                candidate["coverage"],
                abs(stats["direction_balance"]),
                -abs(position - 50),
            )
            if best_key is None or candidate_key > best_key:
                best = candidate
                best_key = candidate_key

    if not best:
        return None

    return {
        "score": int(best["score"]),
        "coverage": int(best["coverage"]),
        "orientation": best["orientation"],
        "position": int(best["position"]),
        "tracks": len(trajectories),
        "width": width,
        "height": height,
        "processed_frames": int(data["processed_frames"]),
    }


def update_crossing_counts(
    track_last_seen,
    track_states,
    track_last_count,
    frame_index,
    track_ids,
    boxes_xyxy,
    orientation,
    line_value,
    timestamp,
    margin,
    ttl: int = 90,
    cooldown: int = 3,
    band_min: int = 4,
    band_margin_scale: float = 0.8,
    band_box_scale: float = 0.12,
):
    new_in = 0
    new_out = 0
    events = []

    if not track_ids or not boxes_xyxy:
        return new_in, new_out, events

    for track_id, xyxy in zip(track_ids, boxes_xyxy):
        track_last_seen[track_id] = frame_index
        point_x, point_y = _reference_point(xyxy)
        coord = point_y if orientation == "horizontal" else point_x
        axis_size = (xyxy[3] - xyxy[1]) if orientation == "horizontal" else (xyxy[2] - xyxy[0])
        band = max(band_min, int(margin * band_margin_scale), int(axis_size * band_box_scale))
        side = _classify_side(coord, line_value, band)

        state = track_states.setdefault(
            track_id,
            {
                "side": None,
                "band_touched": False,
                "prev_coord": coord,
            },
        )
        prev_side = state.get("side")
        prev_coord = state.get("prev_coord")
        state["prev_coord"] = coord

        if side == 0:
            if prev_side is not None:
                state["band_touched"] = True
            continue

        if prev_side is None:
            state["side"] = side
            state["band_touched"] = False
            continue

        crossed_segment = (
            prev_coord is not None
            and (prev_coord - line_value) * (coord - line_value) < 0
            and abs(prev_coord - line_value) >= band
            and abs(coord - line_value) >= band
        )

        if side != prev_side and (state.get("band_touched") or crossed_segment):
            last_count_frame = track_last_count.get(track_id, -10_000)
            if frame_index - last_count_frame > cooldown:
                if prev_side < side:
                    new_in += 1
                    events.append({"time_s": timestamp, "direction": "in", "count_delta": 1})
                else:
                    new_out += 1
                    events.append({"time_s": timestamp, "direction": "out", "count_delta": 1})
                track_last_count[track_id] = frame_index
            state["band_touched"] = False
        elif side == prev_side:
            state["band_touched"] = False

        state["side"] = side

    expired = [track_id for track_id, last_seen in list(track_last_seen.items()) if frame_index - last_seen > ttl]
    for track_id in expired:
        track_last_seen.pop(track_id, None)
        track_states.pop(track_id, None)
        track_last_count.pop(track_id, None)

    return new_in, new_out, events
