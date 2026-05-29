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


def _axis_coord(point: tuple[float, float, int], orientation: str) -> float:
    return point[1] if orientation == "horizontal" else point[0]


def _direction_is_consistent(
    trace: list[tuple[float, float, int]],
    orientation: str,
    direction: int,
    min_displacement: float,
) -> bool:
    if len(trace) < 2:
        return False

    coords = [_axis_coord(point, orientation) for point in trace]
    displacement = coords[-1] - coords[0]
    if abs(displacement) < min_displacement:
        return False
    if displacement * direction <= 0:
        return False

    deltas = [b - a for a, b in zip(coords, coords[1:]) if abs(b - a) >= 1]
    if not deltas:
        return False
    aligned = sum(1 for delta in deltas if delta * direction > 0)
    return aligned / len(deltas) >= 0.55


def _distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _restore_lost_track(
    lost_tracks: dict,
    frame_index: int,
    point: tuple[float, float],
    axis_size: float,
    reconnect_max_age: int,
    reconnect_distance: float,
) -> dict | None:
    if not lost_tracks:
        return None

    best_key = None
    best_distance = None
    max_distance_sq = reconnect_distance * reconnect_distance
    for lost_key, state in list(lost_tracks.items()):
        lost_frame = state.get("lost_frame", state.get("last_frame", frame_index))
        if frame_index - lost_frame > reconnect_max_age:
            continue
        last_point = state.get("last_point")
        if not last_point:
            continue

        distance = _distance_sq(point, last_point)
        if distance > max_distance_sq:
            continue

        old_axis_size = max(1.0, float(state.get("axis_size", axis_size) or axis_size or 1.0))
        size_ratio = max(axis_size, old_axis_size) / max(1.0, min(axis_size, old_axis_size))
        if size_ratio > 2.2:
            continue

        if best_distance is None or distance < best_distance:
            best_key = lost_key
            best_distance = distance

    if best_key is None:
        return None

    restored = lost_tracks.pop(best_key).copy()
    restored["reconnected_from"] = best_key
    return restored


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
    maturity_hits: int = 6,
    trace_length: int = 20,
    gate_min_frames: int = 2,
    min_displacement: int | None = None,
    lost_tracks: dict | None = None,
    reconnect_max_age: int = 60,
    reconnect_distance: int | None = None,
):
    new_in = 0
    new_out = 0
    events = []
    lost_tracks = lost_tracks if lost_tracks is not None else {}
    min_displacement = min_displacement if min_displacement is not None else max(12, int(margin * 1.5))
    reconnect_distance = reconnect_distance if reconnect_distance is not None else max(40, int(margin * 6))

    if not track_ids or not boxes_xyxy:
        expired = [track_id for track_id, last_seen in list(track_last_seen.items()) if frame_index - last_seen > ttl]
        for track_id in expired:
            state = track_states.pop(track_id, None)
            if state:
                state["lost_frame"] = track_last_seen.get(track_id, frame_index)
                lost_tracks[track_id] = state
            track_last_seen.pop(track_id, None)
            track_last_count.pop(track_id, None)
        return new_in, new_out, events

    for track_id, xyxy in zip(track_ids, boxes_xyxy):
        point_x, point_y = _reference_point(xyxy)
        point = (point_x, point_y)
        coord = point_y if orientation == "horizontal" else point_x
        axis_size = (xyxy[3] - xyxy[1]) if orientation == "horizontal" else (xyxy[2] - xyxy[0])
        band = max(band_min, int(margin * band_margin_scale), int(axis_size * band_box_scale))
        side = _classify_side(coord, line_value, band)

        if track_id not in track_states:
            restored = _restore_lost_track(
                lost_tracks,
                frame_index,
                point,
                axis_size,
                reconnect_max_age,
                reconnect_distance,
            )
            track_states[track_id] = restored or {
                "hits": 0,
                "confirmed": False,
                "counted": False,
                "gate_state": "outside",
                "enter_side": None,
                "inside_frames": 0,
                "last_nonzero_side": None,
                "trace": [],
            }

        track_last_seen[track_id] = frame_index
        state = track_states[track_id]
        state["hits"] = int(state.get("hits", 0)) + 1
        state["confirmed"] = bool(state.get("confirmed")) or state["hits"] >= maturity_hits
        state["last_frame"] = frame_index
        state["last_point"] = point
        state["axis_size"] = axis_size
        trace = state.setdefault("trace", [])
        trace.append((point_x, point_y, frame_index))
        if len(trace) > trace_length:
            del trace[: len(trace) - trace_length]

        if side == 0:
            if state.get("gate_state") != "inside":
                state["enter_side"] = state.get("last_nonzero_side")
                state["inside_frames"] = 1
                state["gate_state"] = "inside"
            else:
                state["inside_frames"] = int(state.get("inside_frames", 0)) + 1
            continue

        enter_side = state.get("enter_side")
        crossed_gate = state.get("gate_state") == "inside" and enter_side is not None and side != enter_side
        skipped_gate = (
            state.get("gate_state") != "inside"
            and state.get("last_nonzero_side") is not None
            and side != state.get("last_nonzero_side")
        )

        if (crossed_gate or skipped_gate) and not state.get("counted") and state.get("confirmed"):
            start_side = enter_side if crossed_gate else state.get("last_nonzero_side")
            direction = 1 if start_side < side else -1
            last_count_frame = track_last_count.get(track_id, -10_000)
            enough_gate_evidence = skipped_gate or int(state.get("inside_frames", 0)) >= gate_min_frames
            if (
                enough_gate_evidence
                and frame_index - last_count_frame > cooldown
                and _direction_is_consistent(trace, orientation, direction, min_displacement)
            ):
                if direction > 0:
                    new_in += 1
                    events.append({"time_s": timestamp, "direction": "in", "count_delta": 1})
                else:
                    new_out += 1
                    events.append({"time_s": timestamp, "direction": "out", "count_delta": 1})
                track_last_count[track_id] = frame_index
                state["counted"] = True
                state["gate_state"] = "passed"
            else:
                state["gate_state"] = "outside"
        elif state.get("gate_state") == "inside":
            state["gate_state"] = "outside"

        state["inside_frames"] = 0
        state["enter_side"] = None
        state["last_nonzero_side"] = side

    expired = [(track_id, last_seen) for track_id, last_seen in list(track_last_seen.items()) if frame_index - last_seen > ttl]
    for track_id, last_seen in expired:
        state = track_states.pop(track_id, None)
        if state:
            state["lost_frame"] = last_seen
            lost_tracks[track_id] = state
        track_last_seen.pop(track_id, None)
        track_last_count.pop(track_id, None)

    stale_lost = [
        track_id
        for track_id, state in list(lost_tracks.items())
        if frame_index - state.get("lost_frame", frame_index) > reconnect_max_age
    ]
    for track_id in stale_lost:
        lost_tracks.pop(track_id, None)

    return new_in, new_out, events
