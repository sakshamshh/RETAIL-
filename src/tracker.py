from deep_sort_realtime.deepsort_tracker import DeepSort

trackers = {}

def get_tracker(cam_name):
    if cam_name not in trackers:
        trackers[cam_name] = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)
    return trackers[cam_name]

def update_tracks(cam_name, boxes, frame):
    tracker = get_tracker(cam_name)
    detections = []
    for (x1, y1, x2, y2, conf) in boxes:
        w = x2 - x1
        h = y2 - y1
        detections.append(([x1, y1, w, h], conf, "person"))

    try:
        tracks = tracker.update_tracks(detections, frame=frame)
    except Exception:
        return []

    active = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        active.append((tid, x1, y1, x2, y2))

    return active
